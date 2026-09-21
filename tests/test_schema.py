"""The schema is the correctness signal for the whole project, so it gets tested hard.

Two things are under test: that the schema file says what we think it says, and that
`validate_record` turns violations into structured errors rather than exceptions.
"""

import pytest

from schema.validate import (
    ValidationOutcome,
    load_schema,
    parse_and_validate,
    validate_record,
)


class TestSchemaShape:
    def test_schema_loads_and_is_strict(self):
        s = load_schema()
        assert s["additionalProperties"] is False
        assert s["type"] == "object"

    def test_every_documented_field_is_required(self):
        s = load_schema()
        assert set(s["required"]) == set(s["properties"]), (
            "every property is required - nullability is expressed by the null type, "
            "not by omitting the key, so that a missing key is always a real error"
        )

    def test_nested_objects_are_strict_too(self):
        s = load_schema()
        for path in ("vendor", "buyer"):
            assert s["properties"][path]["additionalProperties"] is False
        item = s["properties"]["line_items"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])


class TestValidateRecord:
    def test_accepts_a_complete_record(self, valid_record):
        outcome = validate_record(valid_record)
        assert outcome.ok
        assert outcome.errors == []

    def test_accepts_all_nulls(self, valid_record):
        record = dict.fromkeys(valid_record)
        record["vendor"] = {"name": None, "address": None, "email": None, "website": None}
        record["buyer"] = {"name": None, "address": None}
        assert validate_record(record).ok

    def test_accepts_an_empty_line_table(self, mutate):
        assert validate_record(mutate(line_items=[])).ok

    def test_rejects_unknown_top_level_field(self, mutate):
        outcome = validate_record(mutate(vendor_vat_number="GB123"))
        assert not outcome.ok
        assert any("additional" in e.message.lower() for e in outcome.errors)

    def test_rejects_missing_required_field(self, valid_record):
        record = {k: v for k, v in valid_record.items() if k != "currency"}
        outcome = validate_record(record)
        assert not outcome.ok
        assert any("currency" in e.message for e in outcome.errors)

    @pytest.mark.parametrize(
        "bad_date", ["04/03/2021", "2021-3-4", "March 4 2021", "2021-03-04T00:00:00Z", ""]
    )
    def test_rejects_non_iso_dates(self, mutate, bad_date):
        assert not validate_record(mutate(invoice_date=bad_date)).ok

    def test_rejects_impossible_calendar_date(self, mutate):
        assert not validate_record(mutate(invoice_date="2021-02-30")).ok, (
            "format: date must be enforced, not just the pattern"
        )

    @pytest.mark.parametrize("bad_currency", ["usd", "US", "USDD", "$", ""])
    def test_rejects_non_iso_currency(self, mutate, bad_currency):
        assert not validate_record(mutate(currency=bad_currency)).ok

    def test_rejects_amount_as_string(self, mutate):
        assert not validate_record(mutate(total_amount="1440.00")).ok

    def test_rejects_empty_string_where_null_is_meant(self, mutate):
        assert not validate_record(mutate(invoice_number="")).ok, (
            "an empty string is a silent pass pretending to be an answer; null is the honest value"
        )

    def test_accepts_line_items_as_null(self, mutate):
        """null is 'this extractor does not produce line items' - the tuned adapter's honest
        answer, since no training dataset annotates them. [] means a table was looked for
        and not found; the two are different claims and the schema keeps both."""
        assert validate_record(mutate(line_items=None)).ok

    def test_errors_carry_a_json_path(self, mutate):
        outcome = validate_record(mutate(total_amount="1440.00"))
        assert outcome.errors[0].path == "total_amount"

    def test_nested_error_path_points_at_the_line_item(self, mutate):
        record = mutate(
            line_items=[{"description": "ok", "quantity": "two", "unit_price": 1.0, "amount": 2.0}]
        )
        outcome = validate_record(record)
        assert not outcome.ok
        assert outcome.errors[0].path == "line_items.0.quantity"

    def test_reports_every_violation_not_just_the_first(self, mutate):
        outcome = validate_record(mutate(currency="dollars", total_amount="lots"))
        assert len({e.path for e in outcome.errors}) == 2

    def test_outcome_is_immutable(self, valid_record):
        outcome = validate_record(valid_record)
        assert isinstance(outcome, ValidationOutcome)
        with pytest.raises((AttributeError, TypeError)):
            outcome.ok = False


class TestParseAndValidate:
    """Model output arrives as text. Fences are tolerated, nothing else is."""

    def test_parses_plain_json(self, valid_record):
        import json

        assert parse_and_validate(json.dumps(valid_record)).ok

    def test_tolerates_markdown_fence(self, valid_record):
        import json

        text = "```json\n" + json.dumps(valid_record) + "\n```"
        assert parse_and_validate(text).ok

    def test_tolerates_bare_fence(self, valid_record):
        import json

        assert parse_and_validate("```\n" + json.dumps(valid_record) + "\n```").ok

    def test_rejects_prose_around_the_json(self, valid_record):
        import json

        text = "Here is the invoice:\n" + json.dumps(valid_record) + "\nHope that helps!"
        outcome = parse_and_validate(text)
        assert not outcome.ok, "chatty output is a failure mode we must be able to count"

    def test_unparseable_text_is_a_structured_error_not_an_exception(self):
        outcome = parse_and_validate("I could not read this invoice, sorry.")
        assert not outcome.ok
        assert outcome.errors[0].path == "$"
        assert "json" in outcome.errors[0].message.lower()

    def test_json_that_is_not_an_object_is_rejected(self):
        assert not parse_and_validate("[1, 2, 3]").ok

    def test_empty_output_is_rejected(self):
        assert not parse_and_validate("").ok
