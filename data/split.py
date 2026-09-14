"""The layout-ID split. This file decides what the project's numbers mean.

Splitting FATURA by document would put images from the same template in both train and
test, and the resulting accuracy would measure template memorisation. Everything here is
therefore keyed on the layout (template) id parsed out of the filename, never on the
document.

Layout groups
-------------
The release ships `Strat2_Split.txt`, an inter-template split of 40 train layouts and 10
held-out layouts - but it sets `test_inds = dev_inds`, so its dev set *is* its test set.
Iterating a prompt against that would burn the held-out set on the first afternoon. We
keep its 10 held-out layouts untouched and carve a 5-layout dev set out of its 40 training
layouts instead, leaving:

    35 train layouts | 5 dev layouts (unseen) | 10 held-out layouts (unseen, frozen)

Eval sets
---------
`test_seen` reuses the training layouts with documents disjoint from training; `test_unseen`
uses the 10 held-out layouts. The gap between the two is the generalisation story the README
reports. Both are frozen: `write_manifests` refuses to change them once written.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

SPLIT_DIR = Path(__file__).resolve().parent / "splits"

#: Inter-template split shipped as `invoices_dataset_final/Strat2_Split.txt`.
STRAT2_TRAIN_LAYOUTS: tuple[int, ...] = (
    3, 11, 30, 24, 40, 48, 41, 22, 27, 19, 45, 1, 29, 44, 9, 47, 36, 23, 18, 42,
    15, 14, 28, 43, 33, 6, 38, 26, 13, 34, 17, 37, 5, 8, 21, 35, 16, 20, 31, 46,
)  # fmt: skip

#: The released held-out layouts. Never trained on, never used for prompt iteration.
HELD_OUT_LAYOUTS: tuple[int, ...] = (2, 4, 7, 10, 12, 25, 32, 39, 49, 50)

#: Carved from the released *training* layouts so prompt iteration never touches the above.
#: Fixed by hand rather than sampled, so the group cannot drift when a seed changes.
DEV_LAYOUTS: tuple[int, ...] = (5, 14, 26, 38, 47)

TRAIN_LAYOUTS: tuple[int, ...] = tuple(
    layout for layout in STRAT2_TRAIN_LAYOUTS if layout not in DEV_LAYOUTS
)

#: Documents sampled per layout, per split. Train lands at 35 x 50 = 1,750 documents,
#: inside the 1,500-2,000 band the plan asks for.
DOCS_PER_LAYOUT = {"train": 50, "dev_unseen": 40, "test_seen": 10, "test_unseen": 30}

#: Frozen the moment they are written. Changing one invalidates every number in the README.
FROZEN_SPLITS: tuple[str, ...] = ("test_seen", "test_unseen")

DEFAULT_SEED = 20260916

_LAYOUT = re.compile(r"Template(\d+)_Instance", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Split:
    name: str
    layouts: frozenset[int]
    doc_ids: tuple[str, ...]
    frozen: bool


def layout_of(doc_id: str) -> int:
    match = _LAYOUT.search(doc_id)
    if not match:
        raise ValueError(f"{doc_id!r} carries no Template<N> - it cannot be placed in a split")
    return int(match.group(1))


def group_by_layout(doc_ids: Iterable[str]) -> dict[int, list[str]]:
    """Bucket documents by layout id, each bucket sorted so ordering never depends on input."""
    buckets: dict[int, list[str]] = {}
    for doc_id in doc_ids:
        buckets.setdefault(layout_of(doc_id), []).append(doc_id)
    return {layout: sorted(docs) for layout, docs in sorted(buckets.items())}


def _sample(
    buckets: dict[int, list[str]],
    layouts: Sequence[int],
    count: int,
    rng: Random,
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Take `count` documents from each layout, stratified and reproducible."""
    picked: list[str] = []
    for layout in sorted(layouts):
        available = [d for d in buckets[layout] if d not in exclude]
        if len(available) < count:
            raise ValueError(
                f"layout {layout} has {len(available)} documents available, need {count}"
            )
        picked.extend(Random(rng.randrange(2**32)).sample(available, count))
    return tuple(sorted(picked))


def build_splits(doc_ids: Iterable[str], seed: int = DEFAULT_SEED) -> dict[str, Split]:
    """Build all four splits. Deterministic in `seed` and independent of input order."""
    buckets = group_by_layout(doc_ids)

    missing = set(range(1, 51)) - set(buckets)
    if missing:
        raise ValueError(
            f"missing layouts {sorted(missing)} - a split built on a partial corpus would "
            "silently under-represent them"
        )

    rng = Random(seed)
    train = _sample(buckets, TRAIN_LAYOUTS, DOCS_PER_LAYOUT["train"], rng)
    dev = _sample(buckets, DEV_LAYOUTS, DOCS_PER_LAYOUT["dev_unseen"], rng)
    test_seen = _sample(
        buckets, TRAIN_LAYOUTS, DOCS_PER_LAYOUT["test_seen"], rng, exclude=frozenset(train)
    )
    test_unseen = _sample(buckets, HELD_OUT_LAYOUTS, DOCS_PER_LAYOUT["test_unseen"], rng)

    return {
        "train": Split("train", frozenset(TRAIN_LAYOUTS), train, frozen=False),
        "dev_unseen": Split("dev_unseen", frozenset(DEV_LAYOUTS), dev, frozen=False),
        "test_seen": Split("test_seen", frozenset(TRAIN_LAYOUTS), test_seen, frozen=True),
        "test_unseen": Split("test_unseen", frozenset(HELD_OUT_LAYOUTS), test_unseen, frozen=True),
    }


def manifest_digest(split: Split) -> str:
    """SHA-256 over the split's identity. The freeze is auditable in git history through this."""
    payload = json.dumps(
        {
            "name": split.name,
            "layouts": sorted(split.layouts),
            "doc_ids": list(split.doc_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_payload(split: Split) -> dict:
    return {
        "name": split.name,
        "frozen": split.frozen,
        "layouts": sorted(split.layouts),
        "n_documents": len(split.doc_ids),
        "doc_ids": list(split.doc_ids),
        "digest": manifest_digest(split),
    }


def load_manifest(path: Path) -> Split:
    """Read a manifest and verify it has not been edited since it was written."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = Split(
        name=payload["name"],
        layouts=frozenset(payload["layouts"]),
        doc_ids=tuple(payload["doc_ids"]),
        frozen=payload["frozen"],
    )
    actual = manifest_digest(split)
    if actual != payload["digest"]:
        raise ValueError(
            f"{path.name}: digest mismatch - recorded {payload['digest'][:12]}, computed "
            f"{actual[:12]}. The manifest was edited by hand; the split is no longer the one "
            "the published numbers were measured on."
        )
    return split


def write_manifests(splits: dict[str, Split], directory: Path = SPLIT_DIR) -> None:
    """Write every split to disk. A frozen split that already exists may not change."""
    directory.mkdir(parents=True, exist_ok=True)

    for name in FROZEN_SPLITS:
        path = directory / f"{name}.json"
        if not path.exists():
            continue
        existing = load_manifest(path)
        if manifest_digest(existing) != manifest_digest(splits[name]):
            raise RuntimeError(
                f"{name} is frozen and already written, but the new split differs. Held-out "
                "sets never move: delete the manifest deliberately and re-baseline every "
                "published number, or leave it alone."
            )

    for name, split in splits.items():
        (directory / f"{name}.json").write_text(
            json.dumps(_to_payload(split), indent=1) + "\n", encoding="utf-8"
        )
