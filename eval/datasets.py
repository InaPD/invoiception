"""Eval items: one page image, the shipped text layer, and the reference to score against.

Nothing here decides what counts as correct - that is `eval/scoring.py`. This module only
knows where each dataset keeps its pieces:

* FATURA lives in the Zenodo archive and is read straight out of it; the few hundred page
  images an eval set needs are extracted to `data/interim/` on first use.
* RVL-CDIP is already extracted by `data/download.py`.

The **text layer** each dataset ships is what the Path A pre-check consumes:

* FATURA's `OTHER` class is a text pass over the page. It is a stale one: on every
  document it names a different vendor than the image and the `SELLER_NAME` label do
  (measured: 0 of 360 agree; every other field agrees on 83-100%). A model reading it
  cannot be scored on `vendor.name` or `vendor.website`, so `TEXT_LAYER_EXCLUDED_FIELDS`
  takes those out of the text-path score and the README says why.
* RVL-CDIP's ABBYY OCR words, joined in reading order. Median confidence 0.51; that is the
  scanned-document case, warts included.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from data.download import RAW_DIR
from data.map_fatura import MappedRecord, map_annotation
from data.map_rvlcdip import RegionReference, document_ids, parse_ocr, region_text, to_reference
from data.sources import FATURA
from data.split import SPLIT_DIR, load_manifest

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

FATURA_ARCHIVE = RAW_DIR / FATURA.filename
RVLCDIP_DIR = RAW_DIR / "rvlcdip"
#: Present only if a budget constraint cut RVL-CDIP short of its full 520 documents. Every
#: condition scored afterward (this baseline, the Path A pre-check, the tuned adapter) reads
#: this same frozen subset, so "baseline vs adapter on RVL-CDIP" stays a fair comparison
#: instead of two different, silently-different document counts.
RVLCDIP_SAMPLE_PATH = SPLIT_DIR / "rvlcdip_sample.json"

_FATURA_ROOT = "invoices_dataset_final"
#: FATURA ships `images` and `colored_images`; the plain renders are the ones used throughout.
FATURA_IMAGE_VARIANT = "images"
_ANNOTATION = f"{_FATURA_ROOT}/Annotations/Original_Format/{{doc_id}}.json"
_IMAGE = f"{_FATURA_ROOT}/{FATURA_IMAGE_VARIANT}/{{doc_id}}.jpg"

FATURA_SETS: tuple[str, ...] = ("train", "dev_unseen", "test_seen", "test_unseen")
EVAL_SETS: tuple[str, ...] = ("dev_unseen", "test_seen", "test_unseen", "rvlcdip")
#: Never used for prompt iteration. `predict.py` demands an explicit flag to touch them.
FROZEN_SETS: frozenset[str] = frozenset({"test_seen", "test_unseen", "rvlcdip"})

#: Fields the FATURA text layer contradicts; see the module docstring.
TEXT_LAYER_EXCLUDED_FIELDS: frozenset[str] = frozenset({"vendor.name", "vendor.website"})


@dataclass(frozen=True, slots=True)
class EvalItem:
    doc_id: str
    eval_set: str
    image_path: Path
    text: str | None
    reference: MappedRecord | RegionReference


# --------------------------------------------------------------------------------------
# FATURA
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _archive(path: Path = FATURA_ARCHIVE) -> zipfile.ZipFile:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m data.download fatura` first")
    return zipfile.ZipFile(path)


def fatura_annotation(doc_id: str, *, archive: Path = FATURA_ARCHIVE) -> dict:
    return json.loads(_archive(archive).read(_ANNOTATION.format(doc_id=doc_id)))


def fatura_image(
    doc_id: str, *, archive: Path = FATURA_ARCHIVE, interim: Path = INTERIM_DIR
) -> Path:
    """Path to the page image, extracted from the archive on first use."""
    target = interim / "fatura" / FATURA_IMAGE_VARIANT / f"{doc_id}.jpg"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_archive(archive).read(_IMAGE.format(doc_id=doc_id)))
    return target


def fatura_text(annotation: dict) -> str | None:
    other = annotation.get("OTHER")
    text = other.get("text") if isinstance(other, dict) else None
    return text if isinstance(text, str) and text.strip() else None


def load_fatura(
    split_name: str,
    *,
    archive: Path = FATURA_ARCHIVE,
    interim: Path = INTERIM_DIR,
    split_dir: Path = SPLIT_DIR,
) -> list[EvalItem]:
    """Every document of a split, in manifest order, with its mapped reference."""
    if split_name not in FATURA_SETS:
        raise ValueError(f"unknown FATURA split {split_name!r}; expected one of {FATURA_SETS}")
    split = load_manifest(split_dir / f"{split_name}.json")
    items = []
    for doc_id in split.doc_ids:
        annotation = fatura_annotation(doc_id, archive=archive)
        items.append(
            EvalItem(
                doc_id=doc_id,
                eval_set=split_name,
                image_path=fatura_image(doc_id, archive=archive, interim=interim),
                text=fatura_text(annotation),
                reference=map_annotation(annotation, doc_id),
            )
        )
    return items


# --------------------------------------------------------------------------------------
# RVL-CDIP
# --------------------------------------------------------------------------------------


def rvlcdip_page_text(directory: Path, doc_id: str) -> str | None:
    """The whole page's OCR in reading order - the text layer a scanned PDF would carry."""
    text = region_text(parse_ocr(directory / f"{doc_id}_ocr.xml"))
    return text or None


def _sample_digest(doc_ids: list[str], reason: str) -> str:
    payload = json.dumps({"doc_ids": doc_ids, "reason": reason}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_rvlcdip_sample(
    doc_ids: list[str], reason: str, *, path: Path = RVLCDIP_SAMPLE_PATH
) -> None:
    """Freeze a subset of RVL-CDIP as the sample every later condition must use.

    Refuses to change an already-written sample, same as `data.split.write_manifests`
    refuses to move a frozen layout split - the point is that nothing can silently pick a
    different set of documents partway through a comparison.
    """
    doc_ids = sorted(doc_ids)
    payload = {"doc_ids": doc_ids, "reason": reason, "digest": _sample_digest(doc_ids, reason)}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("doc_ids") != doc_ids:
            raise RuntimeError(
                f"{path} already freezes a different RVL-CDIP sample. A comparison already "
                "in progress depends on that exact set; delete it deliberately to replace it."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_rvlcdip_sample(path: Path = RVLCDIP_SAMPLE_PATH) -> frozenset[str] | None:
    """The frozen RVL-CDIP subset, or None if no budget constraint ever cut it short."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _sample_digest(payload["doc_ids"], payload["reason"])
    if actual != payload["digest"]:
        raise ValueError(
            f"{path.name}: digest mismatch - it was edited by hand since being frozen. "
            "Every condition scored against RVL-CDIP depends on this exact set of documents."
        )
    return frozenset(payload["doc_ids"])


def load_rvlcdip(
    directory: Path = RVLCDIP_DIR, *, sample_path: Path = RVLCDIP_SAMPLE_PATH
) -> list[EvalItem]:
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} not found - run `python -m data.download rvlcdip` first"
        )
    sample = load_rvlcdip_sample(sample_path)
    doc_ids = document_ids(directory)
    if sample is not None:
        doc_ids = [d for d in doc_ids if d in sample]
    items = []
    for doc_id in doc_ids:
        reference = to_reference(directory, doc_id)
        if reference.image is None:
            raise FileNotFoundError(f"{doc_id}: no .tif next to its annotation in {directory}")
        items.append(
            EvalItem(
                doc_id=doc_id,
                eval_set="rvlcdip",
                image_path=reference.image,
                text=rvlcdip_page_text(directory, doc_id),
                reference=reference,
            )
        )
    return items


def load_eval_set(name: str) -> list[EvalItem]:
    if name == "rvlcdip":
        return load_rvlcdip()
    if name in FATURA_SETS:
        return load_fatura(name)
    raise ValueError(f"unknown eval set {name!r}; expected one of {EVAL_SETS}")


def excluded_fields_for(eval_set: str, input_kind: str) -> frozenset[str]:
    """Fields that cannot be scored for a given (dataset, input) pair."""
    if input_kind == "text" and eval_set in FATURA_SETS:
        return TEXT_LAYER_EXCLUDED_FIELDS
    return frozenset()
