"""Training targets: the JSON string the adapter learns to emit for a page.

The mapper (`data/map_fatura.py`) produces a deliberately *partial* record - a key is
present only when FATURA annotated it. The schema, on the other hand, requires every key.
This module bridges the two, and the bridge is where a training label could quietly be
invented, so the rules are spelled out:

* **An unannotated scalar field becomes `null`.** FATURA annotates every field a template
  prints: 34 of the 35 training layouts have exactly one supervised field set across all
  50 of their documents (the 35th differs on a single document). "Not annotated" is
  therefore "not printed on this template", and `null` - the schema's word for "the page
  does not state it" - is the label the dataset wrote, not one we filled in.
* **`line_items` is `null`, never `[]`.** No dataset here annotates line items (FATURA's
  `TABLE` is a bare bounding box). The schema distinguishes `null` ("this extractor does
  not produce line items") from `[]` ("a line table was looked for and not found"); every
  FATURA page has a line table, so `[]` would be a false label on every document.
* **Keys are emitted in schema order, compactly.** Compact JSON is roughly half the output
  tokens of pretty-printed JSON, and output tokens are what the adapter's cost per invoice
  is made of. Formatting is invisible to the scorer, which parses first.

The leak guard is the other job here. The plan's non-negotiable is that held-out layouts
never touch training; `guard_training_split` makes that a hard error rather than a
convention, and it checks the *documents*, not just the split's name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.map_fatura import MappedRecord
from data.split import DEV_LAYOUTS, HELD_OUT_LAYOUTS, Split, layout_of
from eval.datasets import EvalItem
from eval.prompt import INSTRUCTION
from schema.validate import load_schema, validate_record

#: The only manifests an adapter may be trained on. Everything else is an eval set.
TRAINABLE_SPLITS: tuple[str, ...] = ("train", "train_4k")

#: Layouts that must never appear in a training example, whatever the split is called.
UNSEEN_LAYOUTS: frozenset[int] = frozenset(HELD_OUT_LAYOUTS) | frozenset(DEV_LAYOUTS)


class LeakError(RuntimeError):
    """A held-out or dev layout was about to enter training. Nothing downstream survives that."""


@dataclass(frozen=True, slots=True)
class TrainingExample:
    doc_id: str
    layout_id: int
    image_path: Path
    instruction: str
    target_json: str


def _complete(properties: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    """Every schema key in schema order; absent keys become null, objects recurse."""
    completed: dict[str, Any] = {}
    for key, spec in properties.items():
        value = partial.get(key)
        if spec.get("type") == "object":
            completed[key] = _complete(spec["properties"], value or {})
        else:
            completed[key] = value
    return completed


def training_target(mapped: MappedRecord) -> dict[str, Any]:
    """The full schema record for one training document. Validates or raises."""
    target = _complete(load_schema()["properties"], mapped.record)
    target["line_items"] = None
    outcome = validate_record(target)
    if not outcome.ok:
        errors = "; ".join(str(e) for e in outcome.errors)
        raise ValueError(f"{mapped.doc_id}: training target violates the schema: {errors}")
    return target


def target_json(record: dict[str, Any]) -> str:
    """Canonical serialisation: compact, key order preserved, non-ASCII kept as is."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def to_example(item: EvalItem) -> TrainingExample:
    if not isinstance(item.reference, MappedRecord):
        raise TypeError(f"{item.doc_id}: only FATURA documents carry field values to train on")
    return TrainingExample(
        doc_id=item.doc_id,
        layout_id=item.reference.layout_id,
        image_path=item.image_path,
        instruction=INSTRUCTION,
        target_json=target_json(training_target(item.reference)),
    )


def guard_training_split(split: Split) -> None:
    """Refuse anything that is not a training split, or that smuggles in an unseen layout."""
    if split.name not in TRAINABLE_SPLITS:
        raise LeakError(
            f"{split.name!r} is not a training split ({TRAINABLE_SPLITS}). Held-out and dev "
            "layouts never touch training - not once."
        )
    leaked = ({layout_of(d) for d in split.doc_ids} | split.layouts) & UNSEEN_LAYOUTS
    if leaked:
        raise LeakError(
            f"{split.name} carries unseen layout(s) {sorted(leaked)}: training on them would "
            "turn the generalisation measurement into memorisation."
        )


def guard_training_examples(name: str, examples: Iterable[TrainingExample]) -> None:
    """The same guard, on the examples themselves - for the bundle writer, which never
    sees a `Split` and must not trust its caller to have checked one."""
    if name not in TRAINABLE_SPLITS:
        raise LeakError(f"{name!r} is not a training split ({TRAINABLE_SPLITS})")
    leaked = {e.layout_id for e in examples} & UNSEEN_LAYOUTS
    if leaked:
        raise LeakError(f"{name} carries unseen layout(s) {sorted(leaked)}; refusing to bundle")


def dataset_digest(examples: Iterable[TrainingExample]) -> str:
    """SHA-256 over (doc_id, instruction, target) - what the model actually learns from.

    Written into the adapter's run config so a held-out result can be traced to the exact
    training set, the same way `prompt_digest` traces a baseline run to its prompt.
    """
    payload = [(e.doc_id, e.instruction, e.target_json) for e in examples]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
