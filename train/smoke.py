"""Score a freshly trained adapter on a handful of dev_unseen pages, on the GPU box.

This is M3's "in-schema output on a smoke test": enough to tell a working adapter from a
broken run before spending a held-out evaluation on it. It reuses the evaluator's field
comparison rules (`eval/scoring.py`) so the number means the same thing here as in the
ship-gate table, but it is scored against the bundle's *targets* rather than the mapped
references, because that is what travels with the images.

Not a substitute for `eval/evaluate.py` on the frozen sets, and never reported as one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from eval.scoring import SCORED_FIELDS, flatten, values_match
from schema.validate import parse_and_validate


@dataclass(frozen=True, slots=True)
class SmokeResult:
    doc_id: str
    valid: bool
    correct: int
    scoreable: int

    @property
    def exact(self) -> bool:
        return self.valid and self.scoreable > 0 and self.correct == self.scoreable


def score_smoke(doc_id: str, raw_output: str | None, target_json: str) -> SmokeResult:
    """Fields with a non-null target value are scoreable; an invalid output misses them all."""
    target = flatten(json.loads(target_json))
    scoreable = [f for f in SCORED_FIELDS if target.get(f) is not None]
    outcome = parse_and_validate(raw_output or "")
    if not outcome.ok:
        return SmokeResult(doc_id, valid=False, correct=0, scoreable=len(scoreable))
    predicted = flatten(outcome.record)
    correct = sum(values_match(f, predicted.get(f), target[f]) for f in scoreable)
    return SmokeResult(doc_id, valid=True, correct=correct, scoreable=len(scoreable))


def summarise(results: Iterable[SmokeResult]) -> dict[str, Any]:
    results = list(results)
    n = len(results)
    scoreable = sum(r.scoreable for r in results)
    return {
        "n_documents": n,
        "validity_rate": sum(r.valid for r in results) / n if n else None,
        "field_accuracy_all": sum(r.correct for r in results) / scoreable if scoreable else None,
        "exact_match_all": sum(r.exact for r in results) / n if n else None,
    }
