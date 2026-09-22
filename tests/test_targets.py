"""Training targets: the JSON string the adapter learns to emit for a page.

A silent bug here would teach the model the wrong schema on every one of 1,750 documents,
so the invariants are pinned: every target validates, keys sit in schema order, an
unannotated field is null, `line_items` is null, and no held-out layout can ever get in.
"""

import json

import pytest

from data.map_fatura import MappedRecord, map_annotation
from data.split import DEV_LAYOUTS, HELD_OUT_LAYOUTS, Split
from eval.datasets import EvalItem
from eval.prompt import INSTRUCTION
from schema.validate import load_schema, validate_record
from train.targets import (
    TRAINABLE_SPLITS,
    LeakError,
    TrainingExample,
    dataset_digest,
    guard_training_split,
    target_json,
    to_example,
    training_target,
)

BBOX = [[0, 0], [1, 1]]
ANNOTATION = {
    "NUMBER": {"bbox": BBOX, "text": "INVOICE # INV/79-83/438"},
    "DATE": {"bbox": BBOX, "text": "Invoice Date: 08-Aug-2013"},
    "SELLER_NAME": {"bbox": BBOX, "text": "Brooks LLC"},
    "BILL_TO": {"bbox": BBOX, "text": "Bill to:Hannah Kim\n7094 Joseph Ports\nAmandaville, AR"},
    "SUB_TOTAL": {"bbox": BBOX, "text": "SUB_TOTAL : 381.59 EUR"},
    "GST(9%)": {"bbox": BBOX, "text": "GST(9%) : 34.34"},
    "TOTAL": {"bbox": BBOX, "text": "BALANCE DUE : 392.91 EUR"},
    "TABLE": BBOX,
    "OTHER": {"text": "..."},
    "INVOICE_INFO": [],
}


@pytest.fixture
def mapped() -> MappedRecord:
    return map_annotation(ANNOTATION, "Template48_Instance97")


class TestTrainingTarget:
    def test_validates_against_the_schema(self, mapped):
        outcome = validate_record(training_target(mapped))
        assert outcome.ok, [str(e) for e in outcome.errors]

    def test_keeps_every_annotated_value(self, mapped):
        target = training_target(mapped)
        assert target["invoice_number"] == "INV/79-83/438"
        assert target["invoice_date"] == "2013-08-08"
        assert target["vendor"]["name"] == "Brooks LLC"
        assert target["buyer"] == {
            "name": "Hannah Kim",
            "address": "7094 Joseph Ports, Amandaville, AR",
        }
        assert target["subtotal"] == 381.59
        assert target["tax"] == 34.34
        assert target["total_amount"] == 392.91
        assert target["currency"] == "EUR"

    def test_unannotated_fields_are_null_not_missing(self, mapped):
        """FATURA annotates every field a template prints (34 of 35 training layouts have
        a constant supervised set), so 'not annotated' is 'not printed' and null is the
        label the dataset wrote, not one we invented."""
        target = training_target(mapped)
        for key in ("purchase_order_number", "due_date", "discount", "amount_due", "payment_terms"):
            assert key in target and target[key] is None
        assert target["vendor"]["email"] is None and target["vendor"]["website"] is None

    def test_line_items_is_null_never_an_empty_list(self, mapped):
        """No dataset annotates line items. null is 'not extracted'; [] would claim the page
        has no line table, which is false on every FATURA page."""
        assert training_target(mapped)["line_items"] is None

    def test_keys_follow_schema_order_at_every_level(self, mapped):
        schema = load_schema()
        target = training_target(mapped)
        assert list(target) == list(schema["properties"])
        for nested in ("vendor", "buyer"):
            assert list(target[nested]) == list(schema["properties"][nested]["properties"])

    def test_does_not_mutate_the_mapped_record(self, mapped):
        before = json.dumps(mapped.record, sort_keys=True)
        training_target(mapped)
        assert json.dumps(mapped.record, sort_keys=True) == before


class TestTargetJson:
    def test_is_compact_and_round_trips(self, mapped):
        target = training_target(mapped)
        text = target_json(target)
        assert text == json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        assert len(text) < len(json.dumps(target, indent=1))
        assert json.loads(text) == target

    def test_keeps_non_ascii_characters_as_is(self):
        assert target_json({"vendor": "Müller & Söhne"}) == '{"vendor":"Müller & Söhne"}'

    def test_is_deterministic(self, mapped):
        assert target_json(training_target(mapped)) == target_json(training_target(mapped))


class TestExample:
    def test_carries_the_fixed_instruction_and_the_page(self, mapped, tmp_path):
        image = tmp_path / "Template48_Instance97.jpg"
        image.write_bytes(b"jpeg")
        item = EvalItem("Template48_Instance97", "train", image, None, mapped)
        example = to_example(item)
        assert isinstance(example, TrainingExample)
        assert example.doc_id == "Template48_Instance97"
        assert example.layout_id == 48
        assert example.image_path == image
        assert example.instruction == INSTRUCTION
        assert json.loads(example.target_json)["invoice_number"] == "INV/79-83/438"

    def test_dataset_digest_changes_with_any_target(self, mapped, tmp_path):
        image = tmp_path / "x.jpg"
        a = to_example(EvalItem("Template48_Instance97", "train", image, None, mapped))
        other = map_annotation(
            {**ANNOTATION, "NUMBER": {"bbox": BBOX, "text": "INVOICE # 2"}}, "Template48_Instance97"
        )
        b = to_example(EvalItem("Template48_Instance97", "train", image, None, other))
        assert dataset_digest([a]) != dataset_digest([b])
        assert dataset_digest([a]) == dataset_digest([a])
        assert len(dataset_digest([a])) == 64


class TestLeakGuard:
    def test_only_the_training_splits_are_trainable(self):
        assert set(TRAINABLE_SPLITS) == {"train", "train_4k"}

    @pytest.mark.parametrize("name", ["test_seen", "test_unseen", "dev_unseen", "rvlcdip"])
    def test_refuses_any_split_that_is_not_a_training_split(self, name):
        split = Split(name, frozenset({1}), ("Template1_Instance1",), frozen=False)
        with pytest.raises(LeakError):
            guard_training_split(split)

    @pytest.mark.parametrize("layout", [HELD_OUT_LAYOUTS[0], DEV_LAYOUTS[0]])
    def test_refuses_a_training_split_that_carries_an_unseen_layout(self, layout):
        """Belt and braces: even a split *named* train is rejected if a held-out or dev
        layout has crept into its documents."""
        split = Split(
            "train",
            frozenset({1, layout}),
            ("Template1_Instance1", f"Template{layout}_Instance1"),
            False,
        )
        with pytest.raises(LeakError, match=str(layout)):
            guard_training_split(split)

    def test_accepts_a_clean_training_split(self):
        guard_training_split(Split("train", frozenset({1}), ("Template1_Instance1",), False))


def test_a_training_target_scores_perfectly_against_its_own_reference(mapped):
    """The bridge between what the adapter is taught and how it is graded.

    If `train/targets.py` and `eval/scoring.py` ever drift apart - a normalisation on one
    side only, a field completed differently - a perfect adapter would be marked wrong and
    the ship-gate table would understate it. Echoing the target back must score 100%.
    """
    from eval.scoring import score_fatura

    score = score_fatura(mapped.doc_id, target_json(training_target(mapped)), mapped)
    assert score.valid
    assert score.fields, "the fixture must supervise something for this to mean anything"
    assert all(score.fields.values()), [f for f, ok in score.fields.items() if not ok]
    assert score.exact
