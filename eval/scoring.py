"""Per-document scoring. Deterministic, and the one place field comparison rules live.

Two datasets, two honest questions:

* FATURA has field *values*, so a prediction is right or wrong. Only fields the dataset
  annotated for that document are scored - a prediction for an unannotated field has no
  label to be checked against, and inventing one is exactly what this project refuses to do.
* RVL-CDIP has *regions*, so the only question we can ask is "does the predicted value
  appear inside the correct annotated region?". A null prediction there is an abstention:
  neither right nor wrong, reported separately.

Both views treat an output that fails the schema as wrong on every scoreable field. The
evaluator also reports accuracy over valid outputs only; that second view lives in
`eval/evaluate.py`, not here.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Any

from data.map_fatura import MappedRecord
from data.map_rvlcdip import FIELD_SCORING, RegionReference
from schema.validate import parse_and_validate

#: Every scalar leaf of the schema, dotted. `line_items` is deliberately absent - neither
#: dataset annotates it, so nothing in this project scores it.
SCORED_FIELDS: tuple[str, ...] = (
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
)

MONEY_FIELDS: frozenset[str] = frozenset(
    {"subtotal", "discount", "tax", "total_amount", "amount_due"}
)
DATE_FIELDS: frozenset[str] = frozenset({"invoice_date", "due_date"})

#: Money matches when both sides agree once rounded (half up) to this many decimals: sub-cent
#: float noise is forgiven, a cent of difference is not. 392.905 matches 392.91; 392.92 does not.
MONEY_DECIMALS = 2

_SEPARATORS = re.compile(r"[\s,]+")
#: Everything except letters and digits, for the containment skeleton in `text_contains`.
_ALNUM = re.compile(r"[^a-z0-9]")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class DocScore:
    """One document's verdict. `fields` holds only the scoreable fields.

    A value of True/False is a scored field; None is an abstention (RVL-CDIP only: the
    model said null where the region exists, which containment cannot judge).
    """

    doc_id: str
    valid: bool
    fields: MappingProxyType
    errors: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        """Valid, at least one scoreable field, and nothing wrong. Abstentions block it."""
        return self.valid and bool(self.fields) and all(v is True for v in self.fields.values())


# --------------------------------------------------------------------------------------
# Normalisation and comparison
# --------------------------------------------------------------------------------------


def normalise_text(value: Any) -> str:
    """Case-fold, unify separators, drop a trailing full stop.

    Commas and newlines both become a single space, so an address the model wrapped on
    lines matches the reference joined with ", ". Content is otherwise untouched.
    """
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _SEPARATORS.sub(" ", text).strip().rstrip(".").strip()


def flatten(record: Any) -> dict[str, Any]:
    """Dotted scalar leaves of a record. Malformed nesting yields nothing rather than raising."""
    if not isinstance(record, dict):
        return {}
    flat: dict[str, Any] = {}
    for field in SCORED_FIELDS:
        head, _, tail = field.partition(".")
        if not tail:
            if head in record:
                flat[field] = record[head]
            continue
        nested = record.get(head)
        if isinstance(nested, dict) and tail in nested:
            flat[field] = nested[tail]
    return flat


def _cents(value: float | int) -> Decimal:
    quantum = Decimal(1).scaleb(-MONEY_DECIMALS)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def values_match(field: str, predicted: Any, reference: Any) -> bool:
    """Whether a predicted value equals the reference under that field's comparison rule."""
    if predicted is None or reference is None:
        return False
    if field in MONEY_FIELDS:
        if isinstance(predicted, bool) or not isinstance(predicted, int | float):
            return False
        return _cents(predicted) == _cents(reference)
    if field in DATE_FIELDS or field == "currency":
        return str(predicted).strip().upper() == str(reference).strip().upper()
    left, right = normalise_text(predicted), normalise_text(reference)
    return bool(left) and left == right


# --------------------------------------------------------------------------------------
# FATURA: field values
# --------------------------------------------------------------------------------------


def score_fatura(
    doc_id: str,
    raw_output: str | None,
    reference: MappedRecord,
    *,
    excluded_fields: frozenset[str] = frozenset(),
) -> DocScore:
    """Score one FATURA document.

    `excluded_fields` exists for the text path: FATURA's shipped text layer contradicts
    `vendor.name` on every document, so a text-conditioned model cannot be scored on it.
    """
    reference_values = flatten(reference.record)
    scoreable = [
        field
        for field in SCORED_FIELDS
        if field in reference.supervised_fields
        and field not in excluded_fields
        and reference_values.get(field) is not None
    ]

    outcome = parse_and_validate(raw_output or "")
    if not outcome.ok:
        return DocScore(
            doc_id=doc_id,
            valid=False,
            fields=MappingProxyType(dict.fromkeys(scoreable, False)),
            errors=tuple(str(e) for e in outcome.errors),
        )

    predicted = flatten(outcome.record)
    fields = {
        field: values_match(field, predicted.get(field), reference_values[field])
        for field in scoreable
    }
    return DocScore(doc_id=doc_id, valid=True, fields=MappingProxyType(fields))


# --------------------------------------------------------------------------------------
# RVL-CDIP: grounding in a region
# --------------------------------------------------------------------------------------


def date_variants(iso: str) -> frozenset[str]:
    """Printed forms a date might take on a 1980s-90s US invoice, normalised for containment."""
    try:
        d = date.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return frozenset()
    month, mon = _MONTHS[d.month - 1], _MONTHS[d.month - 1][:3]
    yy = f"{d.year % 100:02d}"
    forms = [
        f"{d.month}/{d.day}/{yy}",
        f"{d.month:02d}/{d.day:02d}/{yy}",
        f"{d.month}/{d.day}/{d.year}",
        f"{d.month:02d}/{d.day:02d}/{d.year}",
        f"{d.month}-{d.day}-{yy}",
        f"{d.month:02d}-{d.day:02d}-{yy}",
        f"{d.month:02d}-{d.day:02d}-{d.year}",
        f"{d.month:02d}{d.day:02d}{yy}",
        f"{month} {d.day}, {d.year}",
        f"{month} {d.day} {d.year}",
        f"{mon} {d.day}, {d.year}",
        f"{mon} {d.day} {d.year}",
        f"{mon}. {d.day}, {d.year}",
        f"{d.day} {month} {d.year}",
        f"{d.day} {mon} {d.year}",
        f"{d.day}-{mon}-{d.year}",
        f"{d.day}-{mon}-{yy}",
        d.isoformat(),
    ]
    return frozenset(normalise_text(f) for f in forms)


#: A printed symbol is evidence for the code; FATURA prints "$" for USD throughout.
_CURRENCY_SYMBOLS = MappingProxyType({"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥"})


def _money_variants(value: float) -> frozenset[str]:
    forms = {f"{value:.2f}", f"{value:,.2f}", f"{value:.2f}".rstrip("0").rstrip(".")}
    if float(value).is_integer():
        forms.update({f"{int(value)}", f"{int(value):,}"})
    return frozenset(normalise_text(f) for f in forms)


def text_contains(haystack: str, field: str, value: Any) -> bool:
    """Whether `value` for `field` appears in `haystack` under normalised containment.

    Money and dates are matched through the printed forms they could take; everything
    else is a normalised substring test. Substring, not equality: the region is a block
    of text and the value is one thing inside it.
    """
    if value is None or haystack is None:
        return False
    text = f" {normalise_text(haystack)} "
    if field in MONEY_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return any(f" {form} " in text or f"${form} " in text for form in _money_variants(value))
    if field in DATE_FIELDS:
        return any(form in text for form in date_variants(value))
    if field == "currency":
        code = str(value).strip().upper()
        symbol = _CURRENCY_SYMBOLS.get(code)
        return f" {code.casefold()} " in text or (symbol is not None and symbol in text)
    needle = normalise_text(value)
    if needle and needle in text:
        return True
    # OCR hyphenation and spacing are arbitrary ("Morris-USA" for "Morris -USA", a stray
    # period after an abbreviation), so containment also compares the alphanumeric
    # skeletons. Measured on the RVL-CDIP run, 14% of the cases where both conditions
    # produced the *same* value and both were marked wrong were this and nothing else.
    # This leniency is confined to region containment: `values_match`, which scores FATURA
    # against real field values, stays strict.
    skeleton = _ALNUM.sub("", needle)
    return bool(skeleton) and skeleton in _ALNUM.sub("", text)


def score_rvlcdip(doc_id: str, raw_output: str | None, reference: RegionReference) -> DocScore:
    """Score one RVL-CDIP document on grounding. Fields whose region has no text are skipped."""
    scoreable = [
        field for field, scoring in FIELD_SCORING.items() if reference.regions.get(scoring.region)
    ]

    outcome = parse_and_validate(raw_output or "")
    if not outcome.ok:
        return DocScore(
            doc_id=doc_id,
            valid=False,
            fields=MappingProxyType(dict.fromkeys(scoreable, False)),
            errors=tuple(str(e) for e in outcome.errors),
        )

    predicted = flatten(outcome.record)
    fields: dict[str, bool | None] = {}
    for field in scoreable:
        value = predicted.get(field)
        if value is None:
            fields[field] = None
            continue
        region_text = reference.regions[FIELD_SCORING[field].region]
        fields[field] = text_contains(region_text, field, value)
    return DocScore(doc_id=doc_id, valid=True, fields=MappingProxyType(fields))


def dumps_record(record: dict[str, Any]) -> str:
    """Canonical JSON for a record: sorted keys, so two equal records serialise identically."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False)
