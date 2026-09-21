"""Pack the training data into a self-describing bundle for a GPU box.

    python -m train.export                       # data/interim/slotfill-bundle/ + .tar.gz
    python -m train.export --splits train dev_unseen

Training happens on Colab/Kaggle (a 4GB local GPU cannot fine-tune a 3B VLM), so the
page images and their targets have to travel. The bundle is:

    slotfill-bundle/
      bundle.json          instruction, schema digest, per-split example count + digest
      images/<doc_id>.jpg  each page once, shared between splits (train is inside train_4k)
      <split>.jsonl        one example per line: doc_id, layout_id, image, instruction, target

`bundle.json` records a digest per split. `load_bundle_split` recomputes it on the GPU
side, so the run config's `dataset_digest` provably names the data the adapter saw, and a
split not marked `trainable` (dev_unseen, shipped for the post-training smoke test)
cannot be trained on by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.split import SPLIT_DIR, load_manifest
from eval.datasets import FROZEN_SETS, INTERIM_DIR, load_fatura
from schema.validate import SCHEMA_PATH
from train.targets import (
    TRAINABLE_SPLITS,
    LeakError,
    TrainingExample,
    dataset_digest,
    guard_training_examples,
    guard_training_split,
    to_example,
)

BUNDLE_FILE = "bundle.json"
IMAGES_DIR = "images"
DEFAULT_BUNDLE_DIR = INTERIM_DIR / "slotfill-bundle"
#: dev_unseen rides along for the smoke test only; the guard in `load_bundle_split` and
#: the `trainable` flag written next to it both say so.
DEFAULT_SPLITS: tuple[str, ...] = ("train", "train_4k", "dev_unseen")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def schema_digest() -> str:
    return hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()


def _row(example: TrainingExample) -> dict[str, Any]:
    return {
        "doc_id": example.doc_id,
        "layout_id": example.layout_id,
        "image": f"{IMAGES_DIR}/{example.image_path.name}",
        "instruction": example.instruction,
        "target": example.target_json,
    }


def _row_digest(rows: Iterable[dict[str, Any]]) -> str:
    """The same digest `dataset_digest` computes, from rows read back off disk."""
    payload = [(r["doc_id"], r["instruction"], r["target"]) for r in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def export_bundle(
    splits: dict[str, Sequence[TrainingExample]],
    out_dir: Path,
    *,
    trainable: Sequence[str] = TRAINABLE_SPLITS,
) -> Path:
    """Write the bundle directory. Returns `out_dir`."""
    images = out_dir / IMAGES_DIR
    images.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "schema_digest": schema_digest(),
        "instruction": None,
        "splits": {},
    }
    for name, examples in splits.items():
        # The `trainable` flag is what the GPU side trusts, so it is earned here, not declared.
        if name in trainable:
            guard_training_examples(name, examples)
        instructions = {e.instruction for e in examples}
        if len(instructions) != 1:
            raise ValueError(f"{name}: one fixed instruction expected, found {len(instructions)}")
        manifest["instruction"] = instructions.pop()
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as sink:
            for example in examples:
                target = images / example.image_path.name
                if not target.exists():
                    shutil.copyfile(example.image_path, target)
                sink.write(json.dumps(_row(example), ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "n_examples": len(examples),
            "digest": dataset_digest(examples),
            "trainable": name in trainable,
            "layouts": sorted({e.layout_id for e in examples}),
        }
    (out_dir / BUNDLE_FILE).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return out_dir


def load_bundle_split(
    bundle_dir: Path, name: str, *, for_training: bool = False
) -> list[dict[str, Any]]:
    """Read one split back, verifying its digest against `bundle.json`.

    `for_training=True` additionally refuses a split the bundle did not mark trainable -
    the GPU-side half of the leak guard.
    """
    manifest = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    entry = manifest["splits"].get(name)
    if entry is None:
        raise ValueError(f"{name!r} is not in this bundle: {sorted(manifest['splits'])}")
    if for_training and not entry["trainable"]:
        raise ValueError(
            f"{name} is not marked trainable in {BUNDLE_FILE}; it is shipped for the smoke "
            "test only. Training on it would put unseen layouts into the adapter."
        )
    with (bundle_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    actual = _row_digest(rows)
    if actual != entry["digest"]:
        raise ValueError(
            f"{name}.jsonl: digest mismatch - recorded {entry['digest'][:12]}, computed "
            f"{actual[:12]}. The split was edited after export."
        )
    if len(rows) != entry["n_examples"]:
        raise ValueError(f"{name}.jsonl: {len(rows)} rows, manifest says {entry['n_examples']}")
    return rows


def tar_bundle(bundle_dir: Path, archive: Path) -> Path:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--no-tar", action="store_true", help="leave the directory unpacked")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = {}
    for name in args.splits:
        # Frozen sets have no business on a training box, not even as a smoke test.
        if name in FROZEN_SETS:
            raise LeakError(f"{name} is a frozen held-out set and is never exported for training")
        if name in TRAINABLE_SPLITS:
            guard_training_split(load_manifest(SPLIT_DIR / f"{name}.json"))
        items = load_fatura(name)
        splits[name] = [to_example(item) for item in items]
        print(f"{name:12s} {len(items):5d} examples")
    export_bundle(splits, args.out)
    print(f"bundle written to {args.out}")
    if not args.no_tar:
        archive = tar_bundle(args.out, args.out.with_suffix(".tar.gz"))
        print(f"archive: {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
