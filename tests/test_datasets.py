"""Eval items carry the right image, text layer and reference for each dataset."""

import json
import zipfile

import pytest
from PIL import Image

from data.split import Split, manifest_digest
from eval.datasets import (
    TEXT_LAYER_EXCLUDED_FIELDS,
    _archive,
    excluded_fields_for,
    fatura_text,
    load_fatura,
    load_rvlcdip,
)
from tests.test_map_rvlcdip import GT_XML, OCR_XML

ANNOTATION = {
    "NUMBER": {"bbox": [[0, 0], [1, 1]], "text": "INVOICE # INV/1"},
    "TOTAL": {"bbox": [[0, 0], [1, 1]], "text": "TOTAL : 10.00 USD"},
    "OTHER": {"text": "INVOICE # INV/1\nTOTAL : 10.00 USD"},
}


@pytest.fixture
def fatura(tmp_path):
    archive = tmp_path / "FATURA2.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for doc_id in ("Template1_Instance1", "Template1_Instance2"):
            zf.writestr(
                f"invoices_dataset_final/Annotations/Original_Format/{doc_id}.json",
                json.dumps(ANNOTATION),
            )
            image = tmp_path / "tmp.jpg"
            Image.new("RGB", (10, 10)).save(image)
            zf.write(image, f"invoices_dataset_final/images/{doc_id}.jpg")
    split = Split(
        "dev_unseen", frozenset({1}), ("Template1_Instance1", "Template1_Instance2"), False
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "dev_unseen.json").write_text(
        json.dumps(
            {
                "name": "dev_unseen",
                "frozen": False,
                "layouts": [1],
                "n_documents": 2,
                "doc_ids": list(split.doc_ids),
                "digest": manifest_digest(split),
            }
        )
    )
    _archive.cache_clear()
    yield {"archive": archive, "interim": tmp_path / "interim", "split_dir": split_dir}
    _archive.cache_clear()


def test_load_fatura_extracts_images_and_maps_references(fatura):
    items = load_fatura("dev_unseen", **fatura)
    assert [i.doc_id for i in items] == ["Template1_Instance1", "Template1_Instance2"]
    item = items[0]
    assert item.eval_set == "dev_unseen"
    assert item.image_path.exists() and item.image_path.suffix == ".jpg"
    assert item.text == "INVOICE # INV/1\nTOTAL : 10.00 USD"
    assert item.reference.record["invoice_number"] == "INV/1"
    assert item.reference.record["total_amount"] == 10.0


def test_load_fatura_rejects_unknown_split(fatura):
    with pytest.raises(ValueError, match="unknown FATURA split"):
        load_fatura("held_out_by_another_name", **fatura)


def test_fatura_text_is_none_when_absent_or_blank():
    assert fatura_text({}) is None
    assert fatura_text({"OTHER": {"text": "  "}}) is None
    assert fatura_text({"OTHER": []}) is None


def test_load_rvlcdip_builds_page_text_and_region_reference(tmp_path):
    (tmp_path / "doc1_gt.xml").write_text(GT_XML, encoding="utf-8")
    (tmp_path / "doc1_ocr.xml").write_text(OCR_XML, encoding="utf-8")
    (tmp_path / "doc1.tif").write_bytes(b"tif")
    # Isolated from the real project's frozen sample - this fake "doc1" would otherwise
    # be filtered out by it, since that sample names 433 real RVL-CDIP document ids.
    items = load_rvlcdip(tmp_path, sample_path=tmp_path / "no_sample.json")
    assert len(items) == 1
    item = items[0]
    assert item.eval_set == "rvlcdip"
    assert item.image_path == tmp_path / "doc1.tif"
    assert item.text == "Acme Supplies Woking AMOUNT $11,780.63 Footer"
    assert item.reference.regions["supplier"] == "Acme Supplies Woking"


def test_load_rvlcdip_requires_the_image(tmp_path):
    (tmp_path / "doc1_gt.xml").write_text(GT_XML, encoding="utf-8")
    (tmp_path / "doc1_ocr.xml").write_text(OCR_XML, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no .tif"):
        load_rvlcdip(tmp_path, sample_path=tmp_path / "no_sample.json")


def test_text_path_on_fatura_excludes_the_contradicted_fields():
    assert excluded_fields_for("test_unseen", "text") == TEXT_LAYER_EXCLUDED_FIELDS
    assert excluded_fields_for("test_unseen", "image") == frozenset()
    assert excluded_fields_for("rvlcdip", "text") == frozenset()


# --------------------------------------------------------------------------------------
# RVL-CDIP frozen sample (a budget-constrained subset, once one exists)
# --------------------------------------------------------------------------------------


def test_load_rvlcdip_sample_is_none_when_no_budget_constraint_ever_cut_it_short(tmp_path):
    from eval.datasets import load_rvlcdip_sample

    assert load_rvlcdip_sample(tmp_path / "no_such_file.json") is None


def test_write_and_load_rvlcdip_sample_roundtrips(tmp_path):
    from eval.datasets import load_rvlcdip_sample, write_rvlcdip_sample

    path = tmp_path / "rvlcdip_sample.json"
    write_rvlcdip_sample(["b", "a", "c"], "ran out of budget at document 433", path=path)
    assert load_rvlcdip_sample(path) == frozenset({"a", "b", "c"})


def test_load_rvlcdip_sample_rejects_hand_editing(tmp_path):
    from eval.datasets import load_rvlcdip_sample, write_rvlcdip_sample

    path = tmp_path / "rvlcdip_sample.json"
    write_rvlcdip_sample(["a", "b"], "reason", path=path)
    payload = json.loads(path.read_text())
    payload["doc_ids"].append("c")  # slip in an extra document by hand
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_rvlcdip_sample(path)


def test_write_rvlcdip_sample_refuses_to_change_an_already_frozen_one(tmp_path):
    from eval.datasets import write_rvlcdip_sample

    path = tmp_path / "rvlcdip_sample.json"
    write_rvlcdip_sample(["a", "b"], "first reason", path=path)
    with pytest.raises(RuntimeError, match="already freezes"):
        write_rvlcdip_sample(["a", "b", "c"], "second reason", path=path)


def test_write_rvlcdip_sample_is_a_no_op_when_rewritten_identically(tmp_path):
    from eval.datasets import write_rvlcdip_sample

    path = tmp_path / "rvlcdip_sample.json"
    write_rvlcdip_sample(["a", "b"], "reason", path=path)
    write_rvlcdip_sample(["b", "a"], "reason", path=path)  # same set, different order: fine
    assert json.loads(path.read_text())["doc_ids"] == ["a", "b"]


def test_load_rvlcdip_filters_to_the_frozen_sample_when_one_exists(tmp_path):
    from eval.datasets import load_rvlcdip, write_rvlcdip_sample

    (tmp_path / "doc1_gt.xml").write_text(GT_XML, encoding="utf-8")
    (tmp_path / "doc1_ocr.xml").write_text(OCR_XML, encoding="utf-8")
    (tmp_path / "doc1.tif").write_bytes(b"tif")

    sample_path = tmp_path / "sample.json"
    write_rvlcdip_sample([], "excludes doc1 entirely", path=sample_path)
    items = load_rvlcdip(tmp_path, sample_path=sample_path)
    assert items == []


def test_load_rvlcdip_loads_everything_when_no_sample_is_frozen(tmp_path):
    from eval.datasets import load_rvlcdip

    (tmp_path / "doc1_gt.xml").write_text(GT_XML, encoding="utf-8")
    (tmp_path / "doc1_ocr.xml").write_text(OCR_XML, encoding="utf-8")
    (tmp_path / "doc1.tif").write_bytes(b"tif")

    items = load_rvlcdip(tmp_path, sample_path=tmp_path / "never_written.json")
    assert [i.doc_id for i in items] == ["doc1"]
