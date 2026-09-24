"""Why the RVL-CDIP numbers are a floor: measure the ground truth, not just the models.

    python -m eval.agreement runs/vision-frontier/rvlcdip runs/vision-adapter/rvlcdip

RVL-CDIP has no field values. A prediction is scored by whether it appears inside the
right annotated region of an ABBYY OCR pass over 1970s-90s microfilm, median per-word
confidence 0.51. When the OCR is wrong, a correct answer scores wrong, and no amount of
model quality fixes that.

This script bounds the effect without inventing labels. It looks at the field instances
where **two independent conditions produced the identical value and both were marked
wrong**. Two different models agreeing character for character is a weak signal that they
read the page correctly, so those cases are sorted by what the region text looks like:

    punctuation/spacing only  the scorer's fault (`text_contains` now folds these)
    close / partial match     the value is in the region, OCR damaged it
    no resemblance            the region genuinely lacks the value: the OCR pass dropped
                              it (logos and letterheads are routinely missed), it is
                              annotated to a different region, or both models are wrong

Only the first is fixable and the last is ambiguous, so this produces a range, not a
corrected score. The README quotes it as the reason the RVL-CDIP column is a floor.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.datasets import load_eval_set
from eval.predict import PREDICTIONS_FILE, load_config, load_predictions
from eval.scoring import (
    DATE_FIELDS,
    FIELD_SCORING,
    MONEY_FIELDS,
    flatten,
    normalise_text,
    text_contains,
)
from schema.validate import parse_and_validate

_ALNUM = re.compile(r"[^a-z0-9]")

#: Similarity of the alphanumeric skeletons above which the value is judged present but
#: OCR-damaged. 0.8 tolerates a few wrong characters in a short string; 0.6 tolerates a lot.
CLOSE_MATCH = 0.8
PARTIAL_MATCH = 0.6

SCORER_GAP = "punctuation/spacing only (scorer, now folded)"
OCR_CLOSE = "value present, OCR garbled a few characters"
OCR_PARTIAL = "value present, heavier OCR damage"
ABSENT = "no resemblance: region lacks the value"


@dataclass(frozen=True, slots=True)
class Agreement:
    """What the two conditions agreed on, and what the ground truth made of it."""

    n_compared: int
    n_agreed_wrong: int
    kinds: Counter


def _records(run_dir: Path) -> dict[str, dict[str, Any]]:
    config = load_config(run_dir)
    if config.eval_set != "rvlcdip":
        raise ValueError(f"{run_dir} is a {config.eval_set} run; this analysis is RVL-CDIP only")
    records = {}
    for doc_id, prediction in load_predictions(run_dir / PREDICTIONS_FILE).items():
        outcome = parse_and_validate(prediction.raw_output or "")
        if outcome.ok:
            records[doc_id] = flatten(outcome.record)
    return records


def skeleton(value: Any) -> str:
    return _ALNUM.sub("", normalise_text(value))


def best_window_ratio(needle: str, haystack: str) -> float:
    """Best similarity of `needle` against any same-length window of `haystack`."""
    if not needle or not haystack:
        return 0.0
    size = len(needle)
    step = max(1, size // 4)
    return max(
        (
            difflib.SequenceMatcher(None, needle, haystack[i : i + size]).ratio()
            for i in range(0, max(1, len(haystack) - size + 1), step)
        ),
        default=0.0,
    )


def classify(field: str, value: Any, region_text: str) -> str:
    needle, hay = skeleton(value), skeleton(region_text)
    # Money, dates and currency codes are matched by their printed forms, not as text, so
    # a skeleton hit on them says nothing about punctuation - a bare digit run inside a
    # longer one would match by accident. Only text fields can land in SCORER_GAP.
    is_text_field = field not in MONEY_FIELDS and field not in DATE_FIELDS and field != "currency"
    if needle and needle in hay and is_text_field:
        return SCORER_GAP
    ratio = best_window_ratio(needle, hay)
    if ratio >= CLOSE_MATCH:
        return OCR_CLOSE
    if ratio >= PARTIAL_MATCH:
        return OCR_PARTIAL
    return ABSENT


def compare(run_a: Path, run_b: Path) -> Agreement:
    items = {item.doc_id: item for item in load_eval_set("rvlcdip")}
    a, b = _records(run_a), _records(run_b)

    compared = 0
    kinds: Counter = Counter()
    for doc_id in sorted(set(a) & set(b)):
        reference = items[doc_id].reference
        for field, scoring in FIELD_SCORING.items():
            region_text = reference.regions.get(scoring.region)
            if not region_text:
                continue
            value_a, value_b = a[doc_id].get(field), b[doc_id].get(field)
            if value_a is None or value_b is None:
                continue
            if normalise_text(value_a) != normalise_text(value_b):
                continue
            compared += 1
            if text_contains(region_text, field, value_a):
                continue
            kinds[classify(field, value_a, region_text)] += 1
    return Agreement(compared, sum(kinds.values()), kinds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)

    result = compare(args.run_a, args.run_b)
    agreed_ok = result.n_compared - result.n_agreed_wrong
    print(
        f"{result.n_compared} field instances where both conditions returned the same value\n"
        f"  {agreed_ok} scored correct, {result.n_agreed_wrong} scored wrong\n"
    )
    print("Of the ones scored wrong:")
    for kind, count in result.kinds.most_common():
        print(f"  {count:5d} ({count / result.n_agreed_wrong:5.1%})  {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
