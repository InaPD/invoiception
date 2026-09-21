"""The training bundle is what leaves this machine for a GPU box; it must be self-describing."""

import json
import tarfile

import pytest

from data.split import HELD_OUT_LAYOUTS
from train.export import BUNDLE_FILE, export_bundle, load_bundle_split, tar_bundle
from train.targets import LeakError, TrainingExample, dataset_digest

TARGET = '{"invoice_number":"INV-1","line_items":null}'


def _examples(tmp_path, split, doc_ids):
    examples = []
    for doc_id in doc_ids:
        image = tmp_path / "src" / f"{doc_id}.jpg"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"\xff\xd8" + doc_id.encode())
        examples.append(TrainingExample(doc_id, 1, image, "Extract.", TARGET))
    return examples


@pytest.fixture
def bundle(tmp_path):
    out = tmp_path / "bundle"
    splits = {
        "train": _examples(tmp_path, "train", ["Template1_Instance1", "Template1_Instance2"]),
        "dev_unseen": _examples(tmp_path, "dev", ["Template5_Instance7"]),
    }
    export_bundle(splits, out, trainable=("train",))
    return out, splits


def test_bundle_manifest_describes_every_split(bundle):
    out, splits = bundle
    manifest = json.loads((out / BUNDLE_FILE).read_text())
    assert manifest["instruction"] == "Extract."
    assert manifest["splits"]["train"] == {
        "n_examples": 2,
        "digest": dataset_digest(splits["train"]),
        "trainable": True,
        "layouts": [1],
    }
    assert manifest["splits"]["dev_unseen"]["trainable"] is False
    assert len(manifest["schema_digest"]) == 64


def test_bundle_copies_images_once_and_references_them_relatively(bundle):
    out, _ = bundle
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "Template1_Instance1.jpg",
        "Template1_Instance2.jpg",
        "Template5_Instance7.jpg",
    ]
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert rows[0] == {
        "doc_id": "Template1_Instance1",
        "layout_id": 1,
        "image": "images/Template1_Instance1.jpg",
        "instruction": "Extract.",
        "target": TARGET,
    }


def test_load_bundle_split_round_trips_and_checks_the_digest(bundle):
    out, splits = bundle
    rows = load_bundle_split(out, "train")
    assert [r["doc_id"] for r in rows] == [e.doc_id for e in splits["train"]]
    assert all((out / r["image"]).exists() for r in rows)


def test_load_bundle_split_refuses_a_tampered_split(bundle):
    out, _ = bundle
    path = out / "train.jsonl"
    path.write_text(path.read_text().replace("INV-1", "INV-2"))
    with pytest.raises(ValueError, match="digest"):
        load_bundle_split(out, "train")


def test_load_bundle_split_refuses_to_train_on_a_non_trainable_split(bundle):
    out, _ = bundle
    with pytest.raises(ValueError, match="trainable"):
        load_bundle_split(out, "dev_unseen", for_training=True)
    assert load_bundle_split(out, "dev_unseen", for_training=False)


def test_tar_bundle_packs_the_directory(bundle, tmp_path):
    out, _ = bundle
    archive = tar_bundle(out, tmp_path / "bundle.tar.gz")
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
    assert "bundle/bundle.json" in names
    assert "bundle/images/Template1_Instance1.jpg" in names


def test_export_bundle_refuses_to_mark_a_leaking_split_trainable(tmp_path):
    """The trainable flag is the only thing the GPU side trusts, so export_bundle must
    earn it itself rather than rely on the CLI having checked the manifest."""
    layout = HELD_OUT_LAYOUTS[0]
    image = tmp_path / "x.jpg"
    image.write_bytes(b"jpeg")
    leaking = [TrainingExample(f"Template{layout}_Instance1", layout, image, "Extract.", TARGET)]
    with pytest.raises(LeakError, match=str(layout)):
        export_bundle({"train": leaking}, tmp_path / "out", trainable=("train",))


def test_export_bundle_refuses_to_mark_a_non_training_split_trainable(tmp_path):
    image = tmp_path / "x.jpg"
    image.write_bytes(b"jpeg")
    examples = [TrainingExample("Template1_Instance1", 1, image, "Extract.", TARGET)]
    with pytest.raises(LeakError, match="not a training split"):
        export_bundle({"dev_unseen": examples}, tmp_path / "out", trainable=("dev_unseen",))
