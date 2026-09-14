"""Map the RVL-CDIP invoice ground truth onto the slotfill schema - as far as it honestly goes.

What the dataset actually contains, verified against all 520 documents:

* `*_gt.xml`  - PageXML with *region* boxes labelled by one of six entities. No text,
  no field values.
* `*_ocr.xml` - PageXML with ABBYY OCR words and per-word confidence. These are 1970s-90s
  litigation scans and the OCR is visibly noisy ("coNinoi.iNC.", "I5UO.OO" for 1500.00).

There is therefore no per-field ground truth value anywhere in this dataset. A `total`
region typically holds a whole column of figures, and an `invoice_info` region holds the
invoice number, the date and their printed labels as one run of text.

The honest consequence, and the contract of this module: RVL-CDIP scores *grounding*, not
accuracy. For the eight fields declared in `FIELD_SCORING` we can ask "does the predicted
value appear inside the correct annotated region?" and nothing stronger. Every other schema
field is declared unscoreable rather than given an invented label.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
_P = f"{{{PAGE_NS}}}"

#: The six entities annotated in `*_gt.xml`, in descending frequency across the 520 docs.
RVLCDIP_REGIONS: tuple[str, ...] = (
    "supplier",
    "receiver",
    "invoice_info",
    "positions",
    "total",
    "other",
)

#: Words on the same printed line wobble by a few pixels; group anything closer than this.
_LINE_TOLERANCE_PX = 12.0

BBox = tuple[float, float, float, float]


class ScoringMode(Enum):
    """How a field may be scored against this dataset. There is deliberately only one."""

    CONTAINMENT = "containment"


@dataclass(frozen=True, slots=True)
class RegionScoring:
    region: str
    mode: ScoringMode
    note: str


#: Schema field -> the region a correct answer must be drawn from. Anything absent from
#: this table is unscoreable on RVL-CDIP; see the mapping table in the README.
FIELD_SCORING = MappingProxyType(
    {
        "vendor.name": RegionScoring(
            "supplier", ScoringMode.CONTAINMENT, "supplier block mixes name, address and phone"
        ),
        "vendor.address": RegionScoring(
            "supplier", ScoringMode.CONTAINMENT, "not separable from vendor.name in the annotation"
        ),
        "buyer.name": RegionScoring(
            "receiver", ScoringMode.CONTAINMENT, "receiver block mixes name, address and attn line"
        ),
        "buyer.address": RegionScoring(
            "receiver", ScoringMode.CONTAINMENT, "not separable from buyer.name in the annotation"
        ),
        "invoice_number": RegionScoring(
            "invoice_info", ScoringMode.CONTAINMENT, "region holds the number, its label and dates"
        ),
        "invoice_date": RegionScoring(
            "invoice_info", ScoringMode.CONTAINMENT, "raw printed form only; no ISO normalisation"
        ),
        "due_date": RegionScoring(
            "invoice_info", ScoringMode.CONTAINMENT, "rarely present and never distinguished"
        ),
        "total_amount": RegionScoring(
            "total", ScoringMode.CONTAINMENT, "region often holds a whole column of figures"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Region:
    entity: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class OcrWord:
    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class RegionReference:
    """Everything RVL-CDIP can tell us about one document.

    Note what is *not* here: no invoice_number, no total_amount, no line items. Those
    would be guesses, and this project does not manufacture labels.
    """

    doc_id: str
    regions: MappingProxyType
    mean_ocr_confidence: float
    image: Path | None = None


def _bbox(points: str) -> BBox:
    """Axis-aligned bounds of a PageXML `Coords` polygon."""
    if not points.strip():
        raise ValueError("empty Coords points attribute - the PageXML region has no geometry")
    pairs = [p.split(",") for p in points.split()]
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    return min(xs), min(ys), max(xs), max(ys)


def parse_gt(path: Path) -> list[Region]:
    """Read the labelled regions from a `*_gt.xml` PageXML file, in document order."""
    root = ET.parse(path).getroot()
    regions: list[Region] = []
    for node in root.iter(f"{_P}TextRegion"):
        entity = next(
            (
                prop.get("value")
                for prop in node.findall(f"{_P}Property")
                if prop.get("key") == "entity"
            ),
            None,
        )
        coords = node.find(f"{_P}Coords")
        if entity is None or coords is None:
            continue
        if entity not in RVLCDIP_REGIONS:
            raise ValueError(f"{path.name}: unknown entity {entity!r}, mapping table is stale")
        regions.append(Region(entity=entity, bbox=_bbox(coords.get("points", ""))))
    return regions


def parse_ocr(path: Path) -> list[OcrWord]:
    """Read OCR words from a `*_ocr.xml` PageXML file. Empty transcriptions are dropped."""
    root = ET.parse(path).getroot()
    words: list[OcrWord] = []
    for node in root.iter(f"{_P}Word"):
        coords = node.find(f"{_P}Coords")
        equiv = node.find(f"{_P}TextEquiv")
        if coords is None or equiv is None:
            continue
        unicode_node = equiv.find(f"{_P}Unicode")
        if unicode_node is None or not (unicode_node.text or "").strip():
            continue
        words.append(
            OcrWord(
                text=unicode_node.text.strip(),
                bbox=_bbox(coords.get("points", "")),
                confidence=float(equiv.get("conf") or 0.0),
            )
        )
    return words


def _centre(bbox: BBox) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def words_in_region(words: Iterable[OcrWord], region: Region) -> list[OcrWord]:
    """Words whose centre falls inside the region box.

    Centre containment rather than full overlap: OCR boxes routinely poke a pixel or two
    past a hand-drawn region edge, and dropping those words would silently truncate values.
    """
    x0, y0, x1, y1 = region.bbox
    return [w for w in words if x0 <= _centre(w.bbox)[0] <= x1 and y0 <= _centre(w.bbox)[1] <= y1]


def region_text(words: Sequence[OcrWord]) -> str:
    """Join words in reading order: banded top to bottom, then left to right within a band."""
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: _centre(w.bbox)[1])
    lines: list[list[OcrWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        if _centre(word.bbox)[1] - _centre(lines[-1][0].bbox)[1] <= _LINE_TOLERANCE_PX:
            lines[-1].append(word)
        else:
            lines.append([word])
    return " ".join(
        w.text for line in lines for w in sorted(line, key=lambda w: _centre(w.bbox)[0])
    )


def field_scoring(field: str) -> RegionScoring | None:
    """How `field` may be scored on RVL-CDIP, or None if the dataset cannot score it."""
    return FIELD_SCORING.get(field)


def to_reference(directory: Path, doc_id: str) -> RegionReference:
    """Build the reference record for one document from its `_gt.xml` / `_ocr.xml` pair."""
    regions = parse_gt(directory / f"{doc_id}_gt.xml")
    words = parse_ocr(directory / f"{doc_id}_ocr.xml")

    # A document may carry several boxes for one entity (e.g. a split letterhead); join them.
    collected: dict[str, list[str]] = {}
    for region in regions:
        text = region_text(words_in_region(words, region))
        if text:
            collected.setdefault(region.entity, []).append(text)

    texts = {
        entity: " ".join(collected[entity]) if entity in collected else None
        for entity in RVLCDIP_REGIONS
    }

    confidences = [w.confidence for w in words]
    image = directory / f"{doc_id}.tif"
    return RegionReference(
        doc_id=doc_id,
        regions=MappingProxyType(texts),
        mean_ocr_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        image=image if image.exists() else None,
    )


def document_ids(directory: Path) -> list[str]:
    """Every document id in an extracted RVL-CDIP directory, sorted for reproducibility."""
    return sorted(p.name.removesuffix("_gt.xml") for p in directory.glob("*_gt.xml"))


def load_all(directory: Path) -> list[RegionReference]:
    """Reference records for every document in the directory."""
    return [to_reference(directory, doc_id) for doc_id in document_ids(directory)]


@dataclass(frozen=True, slots=True)
class OcrQuality:
    """Corpus-level OCR quality. Stated explicitly because the README quotes these numbers.

    `median` pools every word in the corpus rather than averaging per-document medians;
    the two differ here (0.515 against 0.519) and an unstated choice between them is how a
    reproducible number quietly becomes an unreproducible one.
    """

    documents: int
    words: int
    median: float
    mean: float
    zero_confidence_words: int


def ocr_quality(directory: Path) -> OcrQuality:
    """Pooled per-word OCR confidence across every document in an RVL-CDIP directory."""
    doc_ids = document_ids(directory)
    confidences = sorted(
        w.confidence for doc_id in doc_ids for w in parse_ocr(directory / f"{doc_id}_ocr.xml")
    )
    if not confidences:
        raise ValueError(f"no OCR words found under {directory}")
    count = len(confidences)
    median = (
        confidences[count // 2]
        if count % 2
        else (confidences[count // 2 - 1] + confidences[count // 2]) / 2
    )
    return OcrQuality(
        documents=len(doc_ids),
        words=count,
        median=median,
        mean=sum(confidences) / count,
        zero_confidence_words=sum(1 for c in confidences if c == 0.0),
    )
