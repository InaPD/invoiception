"""Aggregation: the ship-gate table must be derivable by hand from the per-document scores."""

import json
from types import MappingProxyType

import pytest

from data.map_fatura import MappedRecord
from data.map_rvlcdip import RegionReference
from eval.backends import Usage
from eval.datasets import EvalItem
from eval.evaluate import aggregate, percentile, score_run, table_row, text_layer_ceiling
from eval.predict import Prediction, RunConfig

REF = {"invoice_number": "INV-1", "total_amount": 10.0, "vendor": {"name": "Acme"}}
SUPERVISED = frozenset({"invoice_number", "total_amount", "vendor.name"})


def _item(doc_id, text="INVOICE INV-1 TOTAL 10.00 Acme"):
    return EvalItem(doc_id, "dev_unseen", None, text, MappedRecord(doc_id, 1, REF, SUPERVISED, ()))


def _record(**overrides):
    record = {
        "invoice_number": "INV-1",
        "purchase_order_number": None,
        "invoice_date": None,
        "due_date": None,
        "vendor": {"name": "Acme", "address": None, "email": None, "website": None},
        "buyer": {"name": None, "address": None},
        "currency": None,
        "subtotal": None,
        "discount": None,
        "tax": None,
        "total_amount": 10.0,
        "amount_due": None,
        "payment_terms": None,
        "line_items": [],
    }
    record.update(overrides)
    return json.dumps(record)


def _prediction(doc_id, raw, latency=1.0, tokens=(1000, 100, 0)):
    return Prediction(
        doc_id, "dev_unseen", "c", "claude-haiku-4-5", raw,
        usage=Usage(*tokens), latency_s=latency, stop_reason="end_turn",
    )  # fmt: skip


def _config(**overrides):
    base = dict(
        condition="c", eval_set="dev_unseen", input_kind="image", backend="anthropic",
        model="claude-haiku-4-5", prompt_digest="x", git_commit=None,
    )  # fmt: skip
    base.update(overrides)
    return RunConfig(**base)


@pytest.mark.parametrize(("p", "expected"), [(50, 2), (95, 4), (100, 4), (1, 1)])
def test_percentile_nearest_rank(p, expected):
    assert percentile([4, 1, 3, 2], p) == expected


def test_percentile_of_nothing_is_none():
    assert percentile([], 95) is None


def test_aggregate_matches_hand_computation():
    items = [_item("a"), _item("b"), _item("c"), _item("d")]
    predictions = {
        "a": _prediction("a", _record(), latency=1.0),  # 3/3
        "b": _prediction("b", _record(total_amount=11.0), latency=2.0),  # 2/3
        "c": _prediction("c", "not json", latency=3.0),  # invalid: 0/3
        # d: not predicted
    }
    config = _config()
    scores = score_run(predictions, items, frozenset())
    m = aggregate(config, items, predictions, scores)

    assert m.n_documents == 4 and m.n_predicted == 3 and m.n_errors == 0
    assert m.n_valid == 2
    assert m.validity_rate == pytest.approx(2 / 3)
    assert m.field_accuracy_all == pytest.approx(5 / 9)
    assert m.field_accuracy_valid == pytest.approx(5 / 6)
    assert m.exact_match_all == pytest.approx(1 / 3)
    assert m.exact_match_valid == pytest.approx(1 / 2)
    assert m.abstain_rate == 0
    assert m.per_field["total_amount"]["correct"] == 1
    assert m.per_field["total_amount"]["wrong"] == 1
    assert m.per_field["total_amount"]["invalid"] == 1
    assert m.latency_p50_s == 2.0 and m.latency_p95_s == 3.0
    assert m.mean_input_tokens == 1000 and m.mean_output_tokens == 100
    # haiku: $1 in + $5 out per Mtok -> (1000*1 + 100*5)/1e6 = $0.0015 per doc -> $1.50 per 1k
    assert m.cost_per_1k_usd == pytest.approx(1.5)
    assert m.text_layer_ceiling is None
    assert m.config["condition"] == "c"


def test_request_errors_count_as_invalid_and_carry_no_latency_or_cost():
    items = [_item("a"), _item("b")]
    predictions = {
        "a": _prediction("a", _record()),
        "b": Prediction("b", "dev_unseen", "c", "claude-haiku-4-5", None, error="HTTP 529"),
    }
    m = aggregate(_config(), items, predictions, score_run(predictions, items, frozenset()))
    assert m.n_errors == 1
    assert m.validity_rate == 0.5
    assert m.field_accuracy_all == 0.5
    assert m.latency_p95_s == 1.0
    assert m.cost_per_1k_usd == pytest.approx(1.5)


def test_unpriced_model_reports_no_cost():
    items = [_item("a")]
    predictions = {"a": _prediction("a", _record())}
    m = aggregate(
        _config(model="qwen-lora"), items, predictions, score_run(predictions, items, frozenset())
    )
    assert m.cost_per_1k_usd is None
    assert "| - |" in table_row(m)


def test_text_layer_ceiling_counts_reference_values_present_in_the_text():
    items = [_item("a", text="INVOICE INV-1 TOTAL 10.00 Acme"), _item("b", text="garbage")]
    ceiling = text_layer_ceiling(items, frozenset({"vendor.name"}))
    assert ceiling == {"invoice_number": 0.5, "total_amount": 0.5}


def test_text_run_on_fatura_reports_ceiling_and_respects_exclusions():
    items = [_item("a")]
    predictions = {
        "a": _prediction(
            "a",
            _record(
                **{"vendor": {"name": "WRONG", "address": None, "email": None, "website": None}}
            ),
        )
    }
    config = _config(input_kind="text", excluded_fields=("vendor.name",))
    scores = score_run(predictions, items, frozenset(config.excluded_fields))
    m = aggregate(config, items, predictions, scores)
    assert "vendor.name" not in m.per_field
    assert m.field_accuracy_all == 1.0
    assert m.text_layer_ceiling == {"invoice_number": 1.0, "total_amount": 1.0}


def test_rvlcdip_abstentions_leave_the_denominator():
    regions = MappingProxyType(
        {
            "supplier": "ACME CORP",
            "receiver": None,
            "invoice_info": "NO 77",
            "positions": None,
            "total": None,
            "other": None,
        }
    )
    ref = RegionReference("r1", regions, 0.5)
    items = [EvalItem("r1", "rvlcdip", None, None, ref)]
    raw = _record(
        **{"vendor": {"name": "Acme Corp", "address": None, "email": None, "website": None}},
        invoice_number="77",
        invoice_date=None,
        due_date=None,
    )
    predictions = {"r1": _prediction("r1", raw)}
    config = _config(eval_set="rvlcdip")
    m = aggregate(config, items, predictions, score_run(predictions, items, frozenset()))
    # scoreable: vendor.name, vendor.address, invoice_number, invoice_date, due_date (5)
    # asserted: vendor.name (grounded), invoice_number (grounded); the other three are null
    assert m.per_field["vendor.name"]["correct"] == 1
    assert m.per_field["vendor.address"]["abstained"] == 1
    assert m.abstain_rate == pytest.approx(3 / 5)
    assert m.field_accuracy_all == 1.0
    assert m.exact_match_all is None  # regions, not values: exact match is undefined here


def test_table_row_formats_every_column():
    items = [_item("a")]
    predictions = {"a": _prediction("a", _record(), latency=0.42)}
    m = aggregate(_config(), items, predictions, score_run(predictions, items, frozenset()))
    row = table_row(m)
    assert row.startswith(
        "| c | dev_unseen | 1 | 100.0% | 100.0% | 100.0% | 100.0% | $1.50 | 0.4s |"
    )


def test_cli_scores_a_run_directory_and_writes_metrics(tmp_path, monkeypatch, capsys):
    from eval import evaluate, predict

    items = [_item("a"), _item("b")]
    monkeypatch.setattr(evaluate, "load_eval_set", lambda name: items)
    run_dir = tmp_path / "c" / "dev_unseen"
    run_dir.mkdir(parents=True)
    (run_dir / predict.CONFIG_FILE).write_text(_config(limit=1).to_json())
    (run_dir / predict.PREDICTIONS_FILE).write_text(_prediction("a", _record()).to_json() + "\n")

    assert evaluate.main([str(run_dir)]) == 0
    metrics = json.loads((run_dir / evaluate.METRICS_FILE).read_text())
    assert metrics["n_documents"] == 1  # the run's --limit is honoured
    assert metrics["field_accuracy_all"] == 1.0
    out = capsys.readouterr().out
    assert "| c | dev_unseen | 1 | 100.0%" in out
    assert "| `invoice_number` | 1 | 100.0%" in out


def test_cli_notes_missing_errored_and_truncated_documents(tmp_path, monkeypatch, capsys):
    from eval import evaluate, predict

    items = [_item("a"), _item("b"), _item("c")]
    monkeypatch.setattr(evaluate, "load_eval_set", lambda name: items)
    run_dir = tmp_path / "c" / "dev_unseen"
    run_dir.mkdir(parents=True)
    (run_dir / predict.CONFIG_FILE).write_text(_config().to_json())
    truncated = Prediction(
        "a", "dev_unseen", "c", "claude-haiku-4-5", "{", stop_reason="max_tokens"
    )
    errored = Prediction("b", "dev_unseen", "c", "claude-haiku-4-5", None, error="HTTP 500")
    (run_dir / predict.PREDICTIONS_FILE).write_text(
        truncated.to_json() + "\n" + errored.to_json() + "\n"
    )
    evaluate.main([str(run_dir)])
    out = capsys.readouterr().out
    assert "1 documents not yet predicted" in out
    assert "1 request errors" in out
    assert "1 outputs truncated" in out


class TestSelfHostedCost:
    """An adapter we host has no per-token price: its cost is GPU time / documents."""

    def test_cost_per_1k_from_gpu_time(self):
        from eval.evaluate import self_hosted_cost_per_1k

        # 1,083 documents in 40 minutes on a $0.35/hour T4.
        cost = self_hosted_cost_per_1k(wall_clock_s=2400.0, n_predicted=1083, usd_per_hour=0.35)
        assert cost == pytest.approx(0.35 * (2400 / 3600) / 1083 * 1000, rel=1e-9)
        assert cost == pytest.approx(0.2154, abs=1e-4)

    def test_is_none_without_a_price_or_without_measured_time(self):
        from eval.evaluate import self_hosted_cost_per_1k

        assert self_hosted_cost_per_1k(2400.0, 1083, None) is None
        assert self_hosted_cost_per_1k(0.0, 1083, 0.35) is None
        assert self_hosted_cost_per_1k(2400.0, 0, 0.35) is None

    def test_throughput_is_documents_per_hour(self):
        from eval.evaluate import throughput_per_hour

        assert throughput_per_hour(3600.0, 500) == pytest.approx(500.0)
        assert throughput_per_hour(0.0, 500) is None


def test_measured_gpu_time_replaces_the_token_price_for_a_self_hosted_model():
    """An adapter served locally has no entry in PRICES; with a GPU rate and measured
    sessions its cost column is real rather than absent, and the throughput is reported
    next to it so the reader can see what the number is made of."""
    from eval.predict import Session

    items = [_item("a"), _item("b")]
    predictions = {"a": _prediction("a", _record()), "b": _prediction("b", _record())}
    config = _config(model="slotfill-lora")
    scores = score_run(predictions, items, frozenset())
    sessions = [Session("2026-09-22T10:00:00+00:00", 3600.0, 2, 16)]

    m = aggregate(config, items, predictions, scores, sessions=sessions, usd_per_hour=0.35)
    assert m.throughput_docs_per_hour == pytest.approx(2.0)
    assert m.cost_per_1k_usd == pytest.approx(0.35 / 2 * 1000)
    assert m.gpu_wall_clock_s == pytest.approx(3600.0)

    # Without a GPU rate the column is honestly absent, not guessed.
    assert aggregate(config, items, predictions, scores, sessions=sessions).cost_per_1k_usd is None


def test_a_priced_api_model_ignores_the_gpu_rate():
    items = [_item("a")]
    predictions = {"a": _prediction("a", _record())}
    config = _config()  # claude-haiku-4-5, which has a published price
    scores = score_run(predictions, items, frozenset())
    from eval.predict import Session

    m = aggregate(
        config,
        items,
        predictions,
        scores,
        sessions=[Session("t", 3600.0, 1, 4)],
        usd_per_hour=0.35,
    )
    assert m.cost_per_1k_usd == pytest.approx(1.5)
