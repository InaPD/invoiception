"""The field-accuracy scorer. Every number in the ship-gate table passes through here."""

from types import MappingProxyType

import pytest

from data.map_fatura import MappedRecord
from data.map_rvlcdip import RegionReference
from eval.scoring import (
    SCORED_FIELDS,
    date_variants,
    flatten,
    normalise_text,
    score_fatura,
    score_rvlcdip,
    text_contains,
    values_match,
)


def _fatura_ref(record, supervised):
    return MappedRecord(
        doc_id="Template1_Instance1",
        layout_id=1,
        record=record,
        supervised_fields=frozenset(supervised),
        conflicts=(),
    )


def _rvl_ref(**regions):
    texts = dict.fromkeys(("supplier", "receiver", "invoice_info", "positions", "total", "other"))
    texts.update(regions)
    return RegionReference(
        doc_id="0000000001", regions=MappingProxyType(texts), mean_ocr_confidence=0.5
    )


# --- normalisation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Brooks   LLC ", "brooks llc"),
        (
            "9950 Santos Squares,\nGarzafurt, MH 31937 US",
            "9950 santos squares garzafurt mh 31937 us",
        ),
        (
            "9950 Santos Squares, Garzafurt, MH 31937 US",
            "9950 santos squares garzafurt mh 31937 us",
        ),
        ("INV/79-83/438", "inv/79-83/438"),
        ("Net 30.", "net 30"),
    ],
)
def test_normalise_text_collapses_whitespace_commas_and_case(raw, expected):
    assert normalise_text(raw) == expected


def test_flatten_produces_dotted_leaves_and_ignores_line_items(valid_record):
    flat = flatten(valid_record)
    assert flat["vendor.name"] == "Acme Supplies Ltd"
    assert flat["buyer.address"] == "8 Harbour Way, Bristol BS1 5TY"
    assert flat["subtotal"] == 1200.0
    assert "line_items" not in flat
    assert set(flat) == set(SCORED_FIELDS)


def test_flatten_tolerates_missing_or_malformed_nesting():
    assert flatten({"vendor": "not an object", "buyer": None}) == {}


# --- value comparison ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "predicted", "reference", "expected"),
    [
        ("total_amount", 392.91, 392.91, True),
        ("total_amount", 392.905, 392.91, True),
        ("total_amount", 392.92, 392.91, False),
        ("total_amount", "392.91", 392.91, False),  # schema-valid output is numeric; strings fail
        ("total_amount", None, 392.91, False),
        ("discount", 51.2, 51.2, True),
        ("invoice_date", "2013-08-08", "2013-08-08", True),
        ("invoice_date", "08-Aug-2013", "2013-08-08", False),
        ("invoice_number", "inv/79-83/438", "INV/79-83/438", True),
        (
            "vendor.address",
            "9950 Santos Squares\nGarzafurt, MH 31937 US",
            "9950 Santos Squares, Garzafurt, MH 31937 US",
            True,
        ),
        ("vendor.email", "OSpencer@example.com", "ospencer@example.com", True),
        ("currency", "eur", "EUR", True),
        ("currency", "USD", "EUR", False),
        ("buyer.name", None, "Hannah Kim", False),
        ("buyer.name", "", "Hannah Kim", False),
    ],
)
def test_values_match(field, predicted, reference, expected):
    assert values_match(field, predicted, reference) is expected


# --- FATURA ----------------------------------------------------------------------------


REF_RECORD = {
    "invoice_number": "INV/79-83/438",
    "invoice_date": "2013-08-08",
    "vendor": {"name": "Brooks LLC", "address": "9950 Santos Squares, Garzafurt, MH 31937 US"},
    "subtotal": 381.59,
    "tax": None,  # annotated but unparseable: a labelling gap, not a model target
}
REF_SUPERVISED = {
    "invoice_number",
    "invoice_date",
    "vendor.name",
    "vendor.address",
    "subtotal",
    "tax",
}


def _output(**overrides):
    record = {
        "invoice_number": "INV/79-83/438",
        "purchase_order_number": None,
        "invoice_date": "2013-08-08",
        "due_date": None,
        "vendor": {
            "name": "Brooks LLC",
            "address": "9950 Santos Squares, Garzafurt, MH 31937 US",
            "email": None,
            "website": None,
        },
        "buyer": {"name": "Somebody", "address": None},
        "currency": "EUR",
        "subtotal": 381.59,
        "discount": None,
        "tax": 34.34,
        "total_amount": 392.91,
        "amount_due": None,
        "payment_terms": None,
        "line_items": [],
    }
    record.update(overrides)
    import json

    return json.dumps(record)


def test_score_fatura_only_scores_supervised_fields_with_a_reference_value():
    score = score_fatura("d", _output(), _fatura_ref(REF_RECORD, REF_SUPERVISED))
    assert score.valid is True
    # buyer.name was predicted but never annotated: no label, no score. tax is None: skipped.
    assert set(score.fields) == {
        "invoice_number",
        "invoice_date",
        "vendor.name",
        "vendor.address",
        "subtotal",
    }
    assert all(v is True for v in score.fields.values())
    assert score.exact is True


def test_score_fatura_marks_wrong_fields_and_breaks_exact_match():
    score = score_fatura("d", _output(subtotal=1.0), _fatura_ref(REF_RECORD, REF_SUPERVISED))
    assert score.fields["subtotal"] is False
    assert score.fields["invoice_number"] is True
    assert score.exact is False


def test_score_fatura_invalid_output_scores_every_field_wrong():
    score = score_fatura("d", "Sure! Here is the JSON: {}", _fatura_ref(REF_RECORD, REF_SUPERVISED))
    assert score.valid is False
    assert set(score.fields) == {
        "invoice_number",
        "invoice_date",
        "vendor.name",
        "vendor.address",
        "subtotal",
    }
    assert all(v is False for v in score.fields.values())
    assert score.exact is False
    assert score.errors


def test_score_fatura_schema_violation_is_invalid_even_if_fields_would_match():
    score = score_fatura(
        "d", _output(invoice_date="08-Aug-2013"), _fatura_ref(REF_RECORD, REF_SUPERVISED)
    )
    assert score.valid is False
    assert all(v is False for v in score.fields.values())


def test_score_fatura_excluded_fields_are_not_scored():
    ref = _fatura_ref(REF_RECORD, REF_SUPERVISED)
    score = score_fatura("d", _output(), ref, excluded_fields=frozenset({"vendor.name"}))
    assert "vendor.name" not in score.fields
    assert "vendor.address" in score.fields


def test_score_fatura_never_scores_line_items():
    ref = _fatura_ref({"line_items": [], "subtotal": 1.0}, {"line_items", "subtotal"})
    score = score_fatura("d", _output(subtotal=1.0), ref)
    assert set(score.fields) == {"subtotal"}


def test_score_fatura_with_nothing_scoreable_is_not_an_exact_match():
    score = score_fatura("d", _output(), _fatura_ref({}, set()))
    assert score.fields == {}
    assert score.exact is False


# --- RVL-CDIP --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "printed",
    [
        "3/12/85",
        "03/12/1985",
        "March 12, 1985",
        "MAR 12 1985",
        "12-Mar-85",
        "12 March 1985",
        "1985-03-12",
        "3-12-85",
    ],
)
def test_date_variants_cover_common_printed_forms(printed):
    assert normalise_text(printed) in date_variants("1985-03-12")


def test_date_variants_of_garbage_is_empty():
    assert date_variants("not a date") == frozenset()


@pytest.mark.parametrize(
    ("field", "value", "haystack", "expected"),
    [
        ("total_amount", 1500.0, "TOTAL DUE $1,500.00", True),
        ("total_amount", 1500.0, "TOTAL 1500.00", True),
        ("total_amount", 1500.0, "TOTAL 1500", True),
        ("total_amount", 1500.0, "TOTAL I5UO.OO", False),  # the OCR ceiling, counted honestly
        ("total_amount", 1500.5, "TOTAL 1500", False),
        ("discount", 51.2, "DISCOUNT(3.91%): (-)  51.2", True),
        ("currency", "USD", "SUB_TOTAL : 185.04 $", True),
        ("currency", "EUR", "TOTAL : 972.30 EUR", True),
        ("currency", "EUR", "TOTAL : 972.30 $", False),
        ("currency", "usd", "TOTAL : 972.30 USD", True),
        ("invoice_date", "1985-03-12", "INVOICE DATE 3/12/85 INVOICE NO 4471", True),
        ("invoice_date", "1985-03-12", "INVOICE DATE 3/12/86", False),
        ("invoice_number", "4471", "INVOICE NO 4471", True),
        (
            "invoice_number",
            "4471",
            "INVOICE NO 44710",
            True,
        ),  # substring: containment, not equality
        (
            "vendor.name",
            "Lorillard Tobacco Co.",
            "LORILLARD TOBACCO CO. ONE PARK AVE NEW YORK",
            True,
        ),
        ("vendor.name", "Lorillard", "PHILIP MORRIS", False),
        ("vendor.name", "", "anything", False),
    ],
)
def test_text_contains(field, value, haystack, expected):
    assert text_contains(haystack, field, value) is expected


def test_score_rvlcdip_grounds_each_field_in_its_region():
    ref = _rvl_ref(
        supplier="LORILLARD TOBACCO CO ONE PARK AVE",
        invoice_info="INVOICE 4471 DATE 3/12/85",
        total="TOTAL 1,500.00",
    )
    out = _output(
        **{
            "vendor": {
                "name": "Lorillard Tobacco Co",
                "address": "One Park Ave",
                "email": None,
                "website": None,
            }
        },
        invoice_number="4471",
        invoice_date="1985-03-12",
        due_date=None,
        total_amount=1500.0,
    )
    score = score_rvlcdip("d", out, ref)
    assert score.valid is True
    assert score.fields["vendor.name"] is True
    assert score.fields["vendor.address"] is True
    assert score.fields["invoice_number"] is True
    assert score.fields["invoice_date"] is True
    assert score.fields["total_amount"] is True
    assert (
        score.fields["due_date"] is None
    )  # abstained: a null prediction is neither right nor wrong here
    # receiver region has no recovered text: buyer fields are unscoreable, not wrong
    assert "buyer.name" not in score.fields
    assert "buyer.address" not in score.fields


def test_score_rvlcdip_wrong_region_is_not_grounding():
    ref = _rvl_ref(supplier="LORILLARD", receiver="PHILIP MORRIS")
    out = _output(
        **{"vendor": {"name": "Philip Morris", "address": None, "email": None, "website": None}}
    )
    score = score_rvlcdip("d", out, ref)
    assert score.fields["vendor.name"] is False
    assert score.fields["vendor.address"] is None


def test_score_rvlcdip_invalid_output_is_wrong_everywhere_scoreable():
    ref = _rvl_ref(supplier="X", total="Y")
    score = score_rvlcdip("d", "not json", ref)
    assert score.valid is False
    assert score.fields == {"vendor.name": False, "vendor.address": False, "total_amount": False}
