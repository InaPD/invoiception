"""The post-training smoke test decides whether an adapter is worth a held-out run."""

import json

from train.smoke import SmokeResult, score_smoke, summarise

TARGET = json.dumps(
    {
        "invoice_number": "INV-1",
        "purchase_order_number": None,
        "invoice_date": "2021-03-04",
        "due_date": None,
        "vendor": {"name": "Acme", "address": None, "email": None, "website": None},
        "buyer": {"name": None, "address": None},
        "currency": "USD",
        "subtotal": None,
        "discount": None,
        "tax": None,
        "total_amount": 10.5,
        "amount_due": None,
        "payment_terms": None,
        "line_items": None,
    }
)


def _output(**overrides):
    record = {**json.loads(TARGET), **overrides}
    return json.dumps(record)


def test_perfect_output_scores_every_non_null_target_field():
    result = score_smoke("doc", _output(), TARGET)
    assert result == SmokeResult("doc", valid=True, correct=5, scoreable=5)


def test_null_target_fields_are_not_scored():
    """The target's nulls are 'not printed'; predicting something there has no label."""
    result = score_smoke("doc", _output(due_date="2021-01-01"), TARGET)
    assert result.scoreable == 5 and result.correct == 5


def test_wrong_value_counts_against_the_field_only():
    result = score_smoke("doc", _output(total_amount=99.0), TARGET)
    assert result.valid and result.correct == 4 and result.scoreable == 5


def test_invalid_output_is_wrong_on_every_field():
    result = score_smoke("doc", "not json at all", TARGET)
    assert result == SmokeResult("doc", valid=False, correct=0, scoreable=5)


def test_money_tolerates_rounding_to_the_cent():
    assert score_smoke("doc", _output(total_amount=10.504), TARGET).correct == 5


def test_summary_reports_rates_the_way_the_evaluator_does():
    results = [
        SmokeResult("a", True, 5, 5),
        SmokeResult("b", True, 3, 5),
        SmokeResult("c", False, 0, 4),
    ]
    summary = summarise(results)
    assert summary["n_documents"] == 3
    assert summary["validity_rate"] == 2 / 3
    assert summary["field_accuracy_all"] == 8 / 14
    assert summary["exact_match_all"] == 1 / 3
