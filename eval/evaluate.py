"""Score a run and produce the ship-gate numbers.

    python -m eval.evaluate runs/vision-frontier/test_unseen

Reads `run_config.json` and `predictions.jsonl`, scores every prediction against the eval
set's references, writes `metrics.json` next to them and prints the table row plus a
per-field breakdown.

Metric definitions (the same for every condition, which is the point of one harness):

* **schema validity** - parses as JSON and validates, markdown fence tolerated.
* **field accuracy, all outputs** - correct field instances over scoreable ones, with an
  invalid output counted wrong on every field. This is the headline view.
* **field accuracy, valid only** - the same over valid outputs, so "how good is it when it
  obeys the schema" can be read separately from "how often does it obey the schema".
* **exact match** - documents where every scoreable field is right.
* **cost per 1,000 invoices** - measured tokens x published prices; None for a model with no
  published price (self-hosted cost is measured at the serving layer).
* **p95 latency** - nearest-rank over the successful requests of this run.

On RVL-CDIP "correct" means grounded in the right region and a null prediction is an
abstention, excluded from the denominator and reported as `abstain_rate`.

A text-input run on FATURA also reports the **text-layer ceiling**: the share of reference
values that appear in the input text at all. The text path cannot beat it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.map_fatura import MappedRecord
from eval.datasets import FATURA_SETS, EvalItem, load_eval_set
from eval.predict import (
    PREDICTIONS_FILE,
    Prediction,
    RunConfig,
    Session,
    load_config,
    load_predictions,
    load_sessions,
)
from eval.pricing import cost_usd
from eval.scoring import (
    SCORED_FIELDS,
    DocScore,
    flatten,
    score_fatura,
    score_rvlcdip,
    text_contains,
)

METRICS_FILE = "metrics.json"


@dataclass(frozen=True, slots=True)
class FieldTally:
    scoreable: int = 0
    correct: int = 0
    wrong: int = 0  # wrong from a valid output
    invalid: int = 0  # wrong because the output failed the schema
    abstained: int = 0

    @property
    def accuracy_all(self) -> float | None:
        denominator = self.scoreable - self.abstained
        return self.correct / denominator if denominator else None

    @property
    def accuracy_valid(self) -> float | None:
        denominator = self.correct + self.wrong
        return self.correct / denominator if denominator else None

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "accuracy_all": self.accuracy_all,
            "accuracy_valid": self.accuracy_valid,
        }


@dataclass(frozen=True, slots=True)
class Metrics:
    condition: str
    eval_set: str
    model: str
    n_documents: int
    n_predicted: int
    n_errors: int
    n_valid: int
    validity_rate: float | None
    field_accuracy_all: float | None
    field_accuracy_valid: float | None
    exact_match_all: float | None
    exact_match_valid: float | None
    abstain_rate: float | None
    per_field: dict[str, dict[str, Any]]
    latency_p50_s: float | None
    latency_p95_s: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_cache_read_tokens: float | None
    cost_per_1k_usd: float | None
    #: Self-hosted only: measured GPU seconds and what they bought.
    gpu_wall_clock_s: float | None
    throughput_docs_per_hour: float | None
    text_layer_ceiling: dict[str, float] | None
    truncated_outputs: int
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


# --------------------------------------------------------------------------------------
# Scoring a run
# --------------------------------------------------------------------------------------


def score_prediction(prediction: Prediction, item: EvalItem, excluded: frozenset[str]) -> DocScore:
    if isinstance(item.reference, MappedRecord):
        return score_fatura(
            item.doc_id, prediction.raw_output, item.reference, excluded_fields=excluded
        )
    return score_rvlcdip(item.doc_id, prediction.raw_output, item.reference)


def score_run(
    predictions: dict[str, Prediction], items: Sequence[EvalItem], excluded: frozenset[str]
) -> list[DocScore]:
    """One score per predicted item, in eval-set order. Unpredicted items are not scored."""
    return [
        score_prediction(predictions[item.doc_id], item, excluded)
        for item in items
        if item.doc_id in predictions
    ]


def percentile(values: Iterable[float], p: float) -> float | None:
    """Nearest-rank percentile; None on empty input."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def text_layer_ceiling(items: Sequence[EvalItem], excluded: frozenset[str]) -> dict[str, float]:
    """Per field: share of reference values findable in the shipped text layer."""
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for item in items:
        if not isinstance(item.reference, MappedRecord) or item.text is None:
            continue
        values = flatten(item.reference.record)
        for name in SCORED_FIELDS:
            if name in excluded or name not in item.reference.supervised_fields:
                continue
            if values.get(name) is None:
                continue
            total[name] += 1
            found[name] += text_contains(item.text, name, values[name])
    return {name: found[name] / total[name] for name in SCORED_FIELDS if total[name]}


def _tally(scores: Sequence[DocScore]) -> dict[str, FieldTally]:
    counts: dict[str, Counter[str]] = {}
    for score in scores:
        for name, verdict in score.fields.items():
            tally = counts.setdefault(name, Counter())
            tally["scoreable"] += 1
            if verdict is None:
                tally["abstained"] += 1
            elif verdict:
                tally["correct"] += 1
            elif score.valid:
                tally["wrong"] += 1
            else:
                tally["invalid"] += 1
    return {name: FieldTally(**counts[name]) for name in SCORED_FIELDS if name in counts}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _overall(per_field: dict[str, FieldTally]) -> FieldTally:
    return FieldTally(
        scoreable=sum(t.scoreable for t in per_field.values()),
        correct=sum(t.correct for t in per_field.values()),
        wrong=sum(t.wrong for t in per_field.values()),
        invalid=sum(t.invalid for t in per_field.values()),
        abstained=sum(t.abstained for t in per_field.values()),
    )


def _exact_match(
    scores: Sequence[DocScore], *, has_values: bool
) -> tuple[float | None, float | None]:
    """(all outputs, valid only). Undefined for a region-only dataset."""
    if not has_values:
        return None, None
    exactable = [s for s in scores if s.fields]
    exactable_valid = [s for s in exactable if s.valid]
    return (
        sum(s.exact for s in exactable) / len(exactable) if exactable else None,
        sum(s.exact for s in exactable_valid) / len(exactable_valid) if exactable_valid else None,
    )


def throughput_per_hour(wall_clock_s: float, n_predicted: int) -> float | None:
    """Documents per hour of measured GPU wall clock. The self-hosted throughput number."""
    if wall_clock_s <= 0 or n_predicted <= 0:
        return None
    return n_predicted / (wall_clock_s / 3600)


def self_hosted_cost_per_1k(
    wall_clock_s: float, n_predicted: int, usd_per_hour: float | None
) -> float | None:
    """GPU $/hour / measured throughput. None unless a rate and real measured time exist.

    Deliberately not a table lookup: a model we host has no published per-token price, and
    inventing one from someone else's API rate would be a different measurement wearing
    this one's label. The rate is passed in by whoever rented the GPU.
    """
    rate = throughput_per_hour(wall_clock_s, n_predicted)
    if rate is None or usd_per_hour is None:
        return None
    return usd_per_hour / rate * 1000


def _cost_per_1k(model: str, successful: Sequence[Prediction]) -> float | None:
    """Mean measured cost x 1000, or None if any request could not be priced."""
    costs = [cost_usd(model, p.usage) for p in successful if p.usage]
    if not costs or any(c is None for c in costs):
        return None
    return sum(costs) / len(costs) * 1000  # type: ignore[arg-type]


def aggregate(
    config: RunConfig,
    items: Sequence[EvalItem],
    predictions: dict[str, Prediction],
    scores: Sequence[DocScore],
    *,
    sessions: Sequence[Session] = (),
    usd_per_hour: float | None = None,
) -> Metrics:
    per_field = _tally(scores)
    overall = _overall(per_field)
    has_values = config.eval_set in FATURA_SETS
    exact_all, exact_valid = _exact_match(scores, has_values=has_values)

    predicted = [predictions[item.doc_id] for item in items if item.doc_id in predictions]
    successful = [p for p in predicted if p.ok]
    latencies = [p.latency_s for p in successful if p.latency_s is not None]
    usages = [p.usage for p in successful if p.usage]
    n_valid = sum(s.valid for s in scores)

    excluded = frozenset(config.excluded_fields)
    ceiling = (
        text_layer_ceiling(items, excluded) if config.input_kind == "text" and has_values else None
    )

    # A published per-token price wins where one exists; a self-hosted model has none, and
    # its cost is the GPU time this run actually spent divided by what it got through.
    wall_clock = sum(s.wall_clock_s for s in sessions)
    n_session_docs = sum(s.n_predicted for s in sessions)
    api_cost = _cost_per_1k(config.model, successful)
    gpu_cost = self_hosted_cost_per_1k(wall_clock, n_session_docs, usd_per_hour)

    return Metrics(
        condition=config.condition,
        eval_set=config.eval_set,
        model=config.model,
        n_documents=len(items),
        n_predicted=len(predicted),
        n_errors=len(predicted) - len(successful),
        n_valid=n_valid,
        validity_rate=n_valid / len(scores) if scores else None,
        field_accuracy_all=overall.accuracy_all,
        field_accuracy_valid=overall.accuracy_valid,
        exact_match_all=exact_all,
        exact_match_valid=exact_valid,
        abstain_rate=(overall.abstained / overall.scoreable) if overall.scoreable else None,
        per_field={name: tally.as_dict() for name, tally in per_field.items()},
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        mean_input_tokens=_mean([u.input_tokens for u in usages]),
        mean_output_tokens=_mean([u.output_tokens for u in usages]),
        mean_cache_read_tokens=_mean([u.cache_read_tokens for u in usages]),
        cost_per_1k_usd=api_cost if api_cost is not None else gpu_cost,
        gpu_wall_clock_s=wall_clock or None,
        throughput_docs_per_hour=throughput_per_hour(wall_clock, n_session_docs),
        text_layer_ceiling=ceiling,
        truncated_outputs=sum(p.stop_reason in ("max_tokens", "length") for p in successful),
        config=asdict(config),
    )


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _usd(value: float | None) -> str:
    return "-" if value is None else f"${value:,.2f}"


def _secs(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


TABLE_HEADER = (
    "| Condition | Eval set | n | Schema validity | Field acc. (all) | Field acc. (valid) "
    "| Exact match | Cost / 1k | p95 latency |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def table_row(m: Metrics) -> str:
    return (
        f"| {m.condition} | {m.eval_set} | {m.n_predicted} | {_pct(m.validity_rate)} "
        f"| {_pct(m.field_accuracy_all)} | {_pct(m.field_accuracy_valid)} "
        f"| {_pct(m.exact_match_all)} | {_usd(m.cost_per_1k_usd)} | {_secs(m.latency_p95_s)} |"
    )


def per_field_table(m: Metrics) -> str:
    lines = [
        "| Field | Scoreable | Acc. (all) | Acc. (valid) | Abstained | Text-layer ceiling |",
        "|---|---|---|---|---|---|",
    ]
    for name, tally in m.per_field.items():
        ceiling = _pct(m.text_layer_ceiling.get(name)) if m.text_layer_ceiling else "-"
        lines.append(
            f"| `{name}` | {tally['scoreable']} | {_pct(tally['accuracy_all'])} "
            f"| {_pct(tally['accuracy_valid'])} | {tally['abstained']} | {ceiling} |"
        )
    return "\n".join(lines)


def evaluate_run(directory: Path, *, usd_per_hour: float | None = None) -> Metrics:
    config = load_config(directory)
    items = load_eval_set(config.eval_set)
    if config.limit:
        items = items[: config.limit]
    predictions = load_predictions(directory / PREDICTIONS_FILE)
    scores = score_run(predictions, items, frozenset(config.excluded_fields))
    return aggregate(
        config,
        items,
        predictions,
        scores,
        sessions=load_sessions(directory),
        usd_per_hour=usd_per_hour,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument(
        "--gpu-usd-per-hour",
        type=float,
        help="rented GPU rate, for self-hosted conditions: cost/1k = rate / measured "
        "throughput. Without it a self-hosted run's cost column reads '-' rather than "
        "borrowing an API price that does not apply to it.",
    )
    args = parser.parse_args(argv)

    print(TABLE_HEADER)
    all_metrics = []
    for directory in args.run_dir:
        metrics = evaluate_run(directory, usd_per_hour=args.gpu_usd_per_hour)
        (directory / METRICS_FILE).write_text(metrics.to_json() + "\n", encoding="utf-8")
        all_metrics.append(metrics)
        print(table_row(metrics))

    for metrics in all_metrics:
        print()
        print(f"### {metrics.condition} on {metrics.eval_set} ({metrics.model})")
        missing = metrics.n_documents - metrics.n_predicted
        notes = []
        if missing:
            notes.append(f"{missing} documents not yet predicted")
        if metrics.n_errors:
            notes.append(f"{metrics.n_errors} request errors (scored as invalid)")
        if metrics.truncated_outputs:
            notes.append(f"{metrics.truncated_outputs} outputs truncated at max_tokens")
        if metrics.abstain_rate:
            notes.append(f"abstain rate {_pct(metrics.abstain_rate)}")
        if notes:
            print("; ".join(notes))
        print(per_field_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
