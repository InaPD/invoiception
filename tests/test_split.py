"""The layout split.

This is the file that decides whether the project measures generalisation or template
memorisation, so the invariants are tested rather than assumed: no layout may straddle
train and an unseen-layout eval set, no document may appear twice, and regenerating the
split must reproduce it byte for byte.
"""

import json

import pytest

from data.split import (
    DEV_LAYOUTS,
    FROZEN_SPLITS,
    HELD_OUT_LAYOUTS,
    TRAIN_LAYOUTS,
    Split,
    build_splits,
    load_manifest,
    manifest_digest,
    write_manifests,
)

ALL_DOCS = [f"Template{layout}_Instance{i}" for layout in range(1, 51) for i in range(200)]


@pytest.fixture
def splits():
    return build_splits(ALL_DOCS)


class TestLayoutPartition:
    def test_the_three_layout_groups_cover_all_fifty_exactly_once(self):
        groups = [set(TRAIN_LAYOUTS), set(DEV_LAYOUTS), set(HELD_OUT_LAYOUTS)]
        assert set().union(*groups) == set(range(1, 51))
        assert sum(len(g) for g in groups) == 50, "a layout may not appear in two groups"

    def test_held_out_is_the_ten_official_strat2_layouts(self):
        assert set(HELD_OUT_LAYOUTS) == {50, 7, 32, 39, 2, 12, 4, 49, 10, 25}

    def test_dev_layouts_are_carved_from_the_official_train_layouts(self):
        """The released split reuses dev as test. Iterating a prompt on it would burn
        the held-out set, so dev comes out of the training layouts instead."""
        assert not set(DEV_LAYOUTS) & set(HELD_OUT_LAYOUTS)
        assert len(DEV_LAYOUTS) == 5

    def test_train_keeps_the_bulk_of_the_layouts(self):
        assert len(TRAIN_LAYOUTS) == 35


class TestSplitContents:
    def test_produces_the_four_expected_splits(self, splits):
        assert set(splits) == {"train", "dev_unseen", "test_seen", "test_unseen"}

    def test_no_document_appears_in_two_splits(self, splits):
        seen = set()
        for split in splits.values():
            assert not seen & set(split.doc_ids), f"{split.name} overlaps an earlier split"
            seen |= set(split.doc_ids)

    def test_train_and_test_unseen_share_no_layout(self, splits):
        assert not splits["train"].layouts & splits["test_unseen"].layouts

    def test_train_and_dev_share_no_layout(self, splits):
        assert not splits["train"].layouts & splits["dev_unseen"].layouts

    def test_test_seen_reuses_the_training_layouts_by_design(self, splits):
        """Seen-vs-unseen is the whole generalisation story; test_seen must be same-layout."""
        assert splits["test_seen"].layouts == splits["train"].layouts

    def test_test_seen_documents_are_disjoint_from_training_documents(self, splits):
        assert not set(splits["test_seen"].doc_ids) & set(splits["train"].doc_ids)

    def test_training_set_is_in_the_planned_size_range(self, splits):
        assert 1500 <= len(splits["train"].doc_ids) <= 2000

    def test_unseen_test_set_is_a_few_hundred_documents(self, splits):
        assert 200 <= len(splits["test_unseen"].doc_ids) <= 600

    @pytest.mark.parametrize("name", ["train", "dev_unseen", "test_seen", "test_unseen"])
    def test_every_split_is_stratified_evenly_across_its_layouts(self, splits, name):
        split = splits[name]
        counts = {
            layout: sum(1 for d in split.doc_ids if d.startswith(f"Template{layout}_"))
            for layout in split.layouts
        }
        assert len(set(counts.values())) == 1, f"{name} is lopsided: {counts}"

    def test_doc_ids_are_sorted_for_reproducibility(self, splits):
        for split in splits.values():
            assert list(split.doc_ids) == sorted(split.doc_ids)

    def test_split_is_immutable(self, splits):
        with pytest.raises((AttributeError, TypeError)):
            splits["train"].name = "tampered"


class TestDeterminism:
    def test_same_seed_gives_the_same_split(self):
        assert build_splits(ALL_DOCS) == build_splits(ALL_DOCS)

    def test_different_seed_gives_a_different_sample_but_the_same_layouts(self):
        other = build_splits(ALL_DOCS, seed=99)
        assert other["train"].doc_ids != build_splits(ALL_DOCS)["train"].doc_ids
        assert other["train"].layouts == build_splits(ALL_DOCS)["train"].layouts

    def test_input_order_does_not_change_the_result(self, splits):
        shuffled = list(reversed(ALL_DOCS))
        assert build_splits(shuffled) == splits

    def test_digest_is_stable_across_runs(self, splits):
        assert manifest_digest(splits["train"]) == manifest_digest(build_splits(ALL_DOCS)["train"])

    def test_digest_changes_if_one_document_changes(self, splits):
        train = splits["train"]
        tampered = Split(
            name=train.name,
            layouts=train.layouts,
            doc_ids=(*train.doc_ids[:-1], "Template1_Instance999"),
            frozen=train.frozen,
        )
        assert manifest_digest(tampered) != manifest_digest(train)


class TestFreezing:
    def test_the_held_out_sets_are_declared_frozen(self, splits):
        assert FROZEN_SPLITS == ("test_seen", "test_unseen")
        for name in FROZEN_SPLITS:
            assert splits[name].frozen is True

    def test_train_and_dev_are_not_frozen(self, splits):
        assert splits["train"].frozen is False
        assert splits["dev_unseen"].frozen is False

    def test_manifests_round_trip(self, splits, tmp_path):
        write_manifests(splits, tmp_path)
        for name, split in splits.items():
            assert load_manifest(tmp_path / f"{name}.json") == split

    def test_manifest_records_its_own_digest(self, splits, tmp_path):
        write_manifests(splits, tmp_path)
        payload = json.loads((tmp_path / "test_unseen.json").read_text())
        assert payload["digest"] == manifest_digest(splits["test_unseen"])

    def test_loading_a_tampered_manifest_fails_loudly(self, splits, tmp_path):
        write_manifests(splits, tmp_path)
        path = tmp_path / "test_unseen.json"
        payload = json.loads(path.read_text())
        payload["doc_ids"].append("Template2_Instance999")
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="digest"):
            load_manifest(path)

    def test_rewriting_a_frozen_manifest_with_different_content_is_refused(self, splits, tmp_path):
        write_manifests(splits, tmp_path)
        moved = build_splits(ALL_DOCS, seed=1234)
        with pytest.raises(RuntimeError, match="frozen"):
            write_manifests(moved, tmp_path)

    def test_rewriting_an_identical_frozen_manifest_is_allowed(self, splits, tmp_path):
        write_manifests(splits, tmp_path)
        write_manifests(build_splits(ALL_DOCS), tmp_path)


class TestLayoutIdSafety:
    def test_a_document_with_no_layout_is_rejected(self):
        with pytest.raises(ValueError):
            build_splits([*ALL_DOCS, "loose_invoice_7"])

    def test_missing_layouts_are_reported_rather_than_silently_shrinking_a_split(self):
        without_layout_3 = [d for d in ALL_DOCS if not d.startswith("Template3_")]
        with pytest.raises(ValueError, match="3"):
            build_splits(without_layout_3)
