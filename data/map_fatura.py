"""Map FATURA's annotation classes onto the slotfill schema.

Read this with the mapping table in the README - every lossy decision below is recorded
there, because the labelling process is part of the result.

Three things about the released data that the paper does not tell you, all verified
against all 10,000 annotation files:

1. There are **35** annotation keys, not the 24 the paper tabulates. The eleven-class gap:
   the seven `GST(<rate>%)` classes appear as a single "GST" row (-6), the three `GSTIN*`
   classes are absent (-3), and `INVOICE_INFO` and `OTHER` are not tabulated at all (-2),
   though both are present in every one of the 10,000 released files.
2. Every annotated value carries its **printed label**: "TOTAL : 972.30 EUR", not "972.30".
   Stripping those labels is the bulk of this module.
3. `TABLE` is a bare bounding box - a single `table` token in the LayoutLM view, with no
   cell text anywhere in the release. **Line items are therefore not annotated in FATURA**,
   and this mapper omits the key rather than emitting `[]`, which would be a label nobody
   wrote. See `MappedRecord.supervised_fields`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

# --------------------------------------------------------------------------------------
# Class vocabulary
# --------------------------------------------------------------------------------------

#: The seven rate-parameterised GST classes, which the paper reports as a single "GST".
GST_RATE_CLASSES: tuple[str, ...] = (
    "GST(1%)",
    "GST(5%)",
    "GST(7%)",
    "GST(9%)",
    "GST(12%)",
    "GST(18%)",
    "GST(20%)",
)

#: Class -> schema field. Dotted targets address a nested object.
CLASS_TO_FIELD = MappingProxyType(
    {
        "NUMBER": "invoice_number",
        "PO_NUMBER": "purchase_order_number",
        "DATE": "invoice_date",
        "DUE_DATE": "due_date",
        "SELLER_NAME": "vendor.name",
        "SELLER_ADDRESS": "vendor.address",
        "SELLER_EMAIL": "vendor.email",
        "SELLER_SITE": "vendor.website",
        "BUYER": "buyer",
        "BILL_TO": "buyer",
        "SEND_TO": "buyer",
        "SUB_TOTAL": "subtotal",
        "DISCOUNT": "discount",
        "TAX": "tax",
        "TOTAL": "total_amount",
        "AMOUNT_DUE": "amount_due",
        "CONDITIONS": "payment_terms",
        **dict.fromkeys(GST_RATE_CLASSES, "tax"),
    }
)

#: Class -> why it is not in the schema. Every one of these appears in the README table.
DROPPED_CLASSES = MappingProxyType(
    {
        "TABLE": "bounding box only - no cell text is released, so line items cannot be labelled",
        "INVOICE_INFO": "present as an empty list in all 10,000 files; carries no content",
        "LOGO": "graphical element, not an extractable field",
        "TITLE": "document heading ('INVOICE', 'COMMERCIAL INVOICE'), not invoice data",
        "OTHER": (
            "a text pass over the whole page; supervising on it would leak the answer, and it "
            "names a different vendor than the image on every document (it is the Path A text "
            "layer, see eval/datasets.py)"
        ),
        "NOTE": "free-text remarks and footers with no schema counterpart",
        "TOTAL_WORDS": "the total spelled out in words - redundant with total_amount",
        "PAYMENT_DETAILS": "bank account details; deliberately out of scope for this schema",
        "GSTIN": "a tax registration number, not a tax amount - no schema field for it",
        "GSTIN_SELLER": "a tax registration number, not a tax amount - no schema field for it",
        "GSTIN_BUYER": "a tax registration number, not a tax amount - no schema field for it",
    }
)

FATURA_CLASSES: frozenset[str] = frozenset(CLASS_TO_FIELD) | frozenset(DROPPED_CLASSES)

#: When several classes feed `buyer`, this is the order of preference. BILL_TO and BUYER
#: are the billed party; SEND_TO is the ship-to party and is only used as a last resort.
BUYER_PRECEDENCE: tuple[str, ...] = ("BUYER", "BILL_TO", "SEND_TO")

# --------------------------------------------------------------------------------------
# Value parsing
# --------------------------------------------------------------------------------------

_LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "NUMBER": re.compile(r"^\s*invoice\s*(?:id|number|no\.?|#)?\s*[:#]?\s*", re.IGNORECASE),
    "PO_NUMBER": re.compile(r"^\s*p\.?o\.?\s*(?:number|no\.?|#)?\s*[:#]?\s*", re.IGNORECASE),
    "SELLER_ADDRESS": re.compile(r"^\s*address\s*[:#]?\s*", re.IGNORECASE),
    "SELLER_EMAIL": re.compile(r"^\s*e-?mail\s*[:#]?\s*", re.IGNORECASE),
    "SELLER_SITE": re.compile(r"^\s*(?:site|web)\s*[:#]?\s*", re.IGNORECASE),
    "CONDITIONS": re.compile(r"^\s*terms\s*(?:&|and)\s*conditions\s*[:#]?\s*", re.IGNORECASE),
}

#: Party blocks end at the first contact line; those belong to neither name nor address.
_CONTACT_LINE = re.compile(r"^\s*(tel|phone|fax|e-?mail|site|web)\s*[:.]", re.IGNORECASE)
_PARTY_LABEL = re.compile(
    r"^\s*(?:bill\s*_?\s*to|ship\s*_?\s*to|send\s*_?\s*to|buyer|sold\s*to)\s*[:#]?\s*",
    re.IGNORECASE,
)

#: A bracketed rate such as "(4.96%)" is a rate, never the amount. Remove before parsing.
_RATE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CURRENCY_CODE = re.compile(r"\b(USD|EUR|GBP|INR|CAD|AUD|JPY|CHF)\b")

#: Symbols seen in the release. `$` is unqualified throughout FATURA and is read as USD;
#: that assumption is recorded in the README mapping table.
_CURRENCY_SYMBOLS = MappingProxyType({"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"})

_LAYOUT = re.compile(r"Template(\d+)_Instance", re.IGNORECASE)
_FATURA_DATE = "%d-%b-%Y"


@dataclass(frozen=True, slots=True)
class Party:
    name: str | None
    address: str | None


def layout_id(name: str) -> int:
    """Layout (template) id from a FATURA filename. The split is built on this, not on docs."""
    match = _LAYOUT.search(name)
    if not match:
        raise ValueError(f"no Template<N> in {name!r} - cannot place it in a layout split")
    return int(match.group(1))


def strip_label(raw: str, klass: str) -> str:
    """Remove the printed label FATURA bakes into the annotated text, and join wrapped lines."""
    text = _LABEL_PATTERNS[klass].sub("", raw) if klass in _LABEL_PATTERNS else raw
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ", ".join(lines) if len(lines) > 1 else (lines[0] if lines else "")


def parse_amount(raw: str) -> float | None:
    """Money from a labelled amount string, ignoring any bracketed percentage rate.

    The sign is preserved. Credit notes and negative adjustments are real, and flipping
    them silently would be worse than failing to parse them. Only `discount` is normalised
    to a positive amount, at its call site, because the schema documents that field as the
    amount deducted.
    """
    candidates = _NUMBER.findall(_RATE.sub(" ", raw))
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def parse_currency(raw: str) -> str | None:
    """ISO 4217 code from an amount string, by explicit code or by symbol."""
    match = _CURRENCY_CODE.search(raw.upper())
    if match:
        return match.group(1)
    return next((iso for sym, iso in _CURRENCY_SYMBOLS.items() if sym in raw), None)


def parse_date(raw: str) -> str | None:
    """Normalise FATURA's `14-Apr-2022` to an ISO calendar date. Unparseable -> None."""
    for token in re.findall(r"\d{1,2}-[A-Za-z]{3}-\d{4}", raw):
        try:
            return datetime.strptime(token, _FATURA_DATE).date().isoformat()
        except ValueError:
            continue
    return None


def parse_party(raw: str, klass: str) -> Party:
    """Split a buyer block into name and address.

    The block is `<label><name>\\n<address lines>\\n<contact lines>`. Contact lines are
    dropped: the schema has nowhere to put a buyer phone number, and silently gluing one
    onto the address would corrupt every address comparison in the eval.
    """
    text = _PARTY_LABEL.sub("", raw, count=1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return Party(name=None, address=None)

    name, rest = lines[0], lines[1:]
    address_lines = []
    for line in rest:
        if _CONTACT_LINE.match(line):
            break
        address_lines.append(line)
    return Party(name=name or None, address=", ".join(address_lines) or None)


# --------------------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappedRecord:
    """One FATURA document mapped onto the schema.

    `record` is deliberately **partial**: a key is present only when the dataset annotated
    it. Absent is not the same as null, and neither is the same as `[]`.
    """

    doc_id: str
    layout_id: int
    record: dict[str, Any]
    supervised_fields: frozenset[str]
    conflicts: tuple[str, ...]


def _set_nested(record: dict[str, Any], dotted: str, value: Any) -> None:
    head, _, tail = dotted.partition(".")
    if tail:
        record.setdefault(head, {})[tail] = value
    else:
        record[head] = value


def _text_of(value: Any) -> str | None:
    """FATURA values are either `{"bbox":..., "text":...}` or a bbox-only list."""
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) and text.strip() else None
    return None


def map_annotation(annotation: dict[str, Any], filename: str) -> MappedRecord:
    """Map one Original_Format annotation dict onto the schema."""
    unknown = set(annotation) - FATURA_CLASSES
    if unknown:
        raise ValueError(
            f"{filename}: unknown FATURA class(es) {sorted(unknown)} - the mapping table in "
            "data/map_fatura.py is stale and the README table would be wrong"
        )

    record: dict[str, Any] = {}
    supervised: set[str] = set()
    conflicts: list[str] = []
    currency_sources: list[str] = []

    # --- buyer: three classes compete for one field ------------------------------------
    present_buyer = [k for k in BUYER_PRECEDENCE if _text_of(annotation.get(k))]
    if present_buyer:
        if len(present_buyer) > 1:
            conflicts.append("buyer:multiple_sources")
        chosen = present_buyer[0]
        party = parse_party(_text_of(annotation[chosen]), chosen)
        record["buyer"] = {"name": party.name, "address": party.address}
        supervised.update({"buyer.name", "buyer.address"})

    # --- tax: VAT and rate-parameterised GST compete for one field ----------------------
    # Several GST lines on one document are a *rate card*, not a charge: Template25 prints
    # GST at 1/5/12/18/20% of the subtotal and its TOTAL matches none of them. There is no
    # printed tax figure to read, so `tax` goes unsupervised rather than being guessed at.
    gst_present = [k for k in GST_RATE_CLASSES if _text_of(annotation.get(k))]
    vat_text = _text_of(annotation.get("TAX"))
    if vat_text and gst_present:
        conflicts.append("tax:vat_and_gst")
    if len(gst_present) > 1 and not vat_text:
        conflicts.append("tax:multiple_gst_rates")
        gst_present = []
    tax_text = vat_text or (_text_of(annotation[gst_present[0]]) if gst_present else None)
    if tax_text:
        record["tax"] = parse_amount(tax_text)
        supervised.add("tax")
        currency_sources.append(tax_text)

    # --- everything else ----------------------------------------------------------------
    handled = {"TAX", *GST_RATE_CLASSES, *BUYER_PRECEDENCE}
    for klass, raw_value in annotation.items():
        if klass in DROPPED_CLASSES or klass in handled:
            continue
        text = _text_of(raw_value)
        if text is None:
            continue
        target = CLASS_TO_FIELD[klass]

        if target in ("invoice_date", "due_date"):
            value = parse_date(text)
        elif target in ("subtotal", "discount", "total_amount", "amount_due"):
            value = parse_amount(text)
            # "DISCOUNT(3.91%): (-)  51.2" marks the deduction with "(-)", which is not a
            # sign the number parser sees; the schema stores the amount deducted.
            if target == "discount" and value is not None:
                value = abs(value)
            currency_sources.append(text)
        else:
            value = strip_label(text, klass) or None

        _set_nested(record, target, value)
        supervised.add(target)

    # --- currency is not annotated; it is read off the amount strings -------------------
    currencies = [c for c in (parse_currency(t) for t in currency_sources) if c]
    if currencies:
        most_common, _ = Counter(currencies).most_common(1)[0]
        record["currency"] = most_common
        supervised.add("currency")
        if len(set(currencies)) > 1:
            conflicts.append("currency:mixed")

    return MappedRecord(
        doc_id=filename.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        layout_id=layout_id(filename),
        record=record,
        supervised_fields=frozenset(supervised),
        conflicts=tuple(conflicts),
    )


# --------------------------------------------------------------------------------------
# Reporting - this is what becomes the README's labelling table
# --------------------------------------------------------------------------------------

#: Schema fields FATURA can never supervise, and why. Sparse coverage is not the same
#: thing: `payment_terms` is annotated on 12 templates and simply absent on the rest.
UNSUPERVISED_FIELDS = MappingProxyType(
    {"line_items": "TABLE is a bounding box with no cell text; no line item is annotated"}
)


@dataclass
class MappingReport:
    """Coverage and conflict counts over a corpus, rendered as the README mapping table."""

    documents: int = 0
    field_coverage: Counter = field(default_factory=Counter)
    conflicts: Counter = field(default_factory=Counter)
    layouts: set[int] = field(default_factory=set)

    _ALL_FIELDS = (
        "invoice_number",
        "purchase_order_number",
        "invoice_date",
        "due_date",
        "vendor.name",
        "vendor.address",
        "vendor.email",
        "vendor.website",
        "buyer.name",
        "buyer.address",
        "currency",
        "subtotal",
        "discount",
        "tax",
        "total_amount",
        "amount_due",
        "payment_terms",
        "line_items",
    )

    def observe(self, mapped: MappedRecord) -> None:
        self.documents += 1
        self.layouts.add(mapped.layout_id)
        for name in self._ALL_FIELDS:
            self.field_coverage[name] += name in mapped.supervised_fields
        self.conflicts.update(mapped.conflicts)

    def to_markdown(self) -> str:
        """The class-mapping table, ready to paste into the README."""
        lines = [
            "| FATURA class | Schema field | Lossy? | Note |",
            "|---|---|---|---|",
        ]
        for klass in sorted(CLASS_TO_FIELD):
            target = CLASS_TO_FIELD[klass]
            if klass in GST_RATE_CLASSES:
                note = "rate-parameterised; VAT wins when both appear on one document"
                lossy = "yes"
            elif klass in BUYER_PRECEDENCE:
                note = f"precedence {' > '.join(BUYER_PRECEDENCE)}; SHIP_TO is not the billed party"
                lossy = "yes"
            elif klass == "DISCOUNT":
                note = "printed as '(-) 51.2'; stored as a positive amount deducted"
                lossy = "no"
            else:
                note = "printed label stripped"
                lossy = "no"
            lines.append(f"| `{klass}` | `{target}` | {lossy} | {note} |")
        for klass, reason in sorted(DROPPED_CLASSES.items()):
            lines.append(f"| `{klass}` | _dropped_ | yes | {reason} |")
        lines.append("")
        lines.append("| Schema field | Supervised by FATURA | Coverage |")
        lines.append("|---|---|---|")
        for name in self._ALL_FIELDS:
            count = self.field_coverage[name]
            if name in UNSUPERVISED_FIELDS:
                lines.append(f"| `{name}` | **no** - {UNSUPERVISED_FIELDS[name]} | 0 |")
            else:
                pct = 100 * count / self.documents if self.documents else 0
                lines.append(f"| `{name}` | yes | {count} / {self.documents} ({pct:.0f}%) |")
        return "\n".join(lines)
