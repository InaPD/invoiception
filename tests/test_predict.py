"""The runner records what happened, resumes cleanly, and never touches a frozen set by accident."""

import json

import pytest
from PIL import Image

from data.map_fatura import MappedRecord
from eval.backends import BackendError, Completion, Usage
from eval.datasets import EvalItem
from eval.predict import (
    PREDICTIONS_FILE,
    HeldOutError,
    Prediction,
    RunConfig,
    RunOptions,
    load_config,
    load_predictions,
    main,
    predict_item,
    run,
)

QUIET = RunOptions(log=lambda _: None)


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, answers, failing=()):
        self.answers = answers
        self.failing = set(failing)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        last_text = request.messages[-1].parts[-1].text
        # the target part precedes the instruction; find which doc this is by its page part
        page = request.messages[-1].parts[0]
        key = getattr(page, "text", None) or page.data
        doc_id = next(d for d, marker in self.answers if marker in key)
        if doc_id in self.failing:
            self.failing.discard(doc_id)
            raise BackendError("HTTP 529: overloaded", retryable=True)
        assert "Reply with the JSON object only" in last_text
        return Completion(
            f'{{"doc": "{doc_id}"}}', Usage(10, 5, 100, 0), 0.25, "fake-model", "end_turn", "req"
        )


@pytest.fixture
def items(tmp_path):
    built = []
    for index in range(3):
        doc_id = f"Template1_Instance{index}"
        image = tmp_path / f"{doc_id}.jpg"
        Image.new("RGB", (8, 8), color=index * 40).save(image)
        built.append(
            EvalItem(
                doc_id=doc_id,
                eval_set="dev_unseen",
                image_path=image,
                text=f"page text {doc_id}",
                reference=MappedRecord(doc_id, 1, {}, frozenset(), ()),
            )
        )
    return built


def _config(**overrides):
    base = dict(
        condition="test-cond", eval_set="dev_unseen", input_kind="text", backend="fake",
        model="fake-model", prompt_digest="abc", git_commit=None,
    )  # fmt: skip
    base.update(overrides)
    return RunConfig(**base)


def _backend(items, failing=()):
    return FakeBackend([(i.doc_id, i.doc_id) for i in items], failing=failing)


def test_run_writes_config_and_one_prediction_per_document(tmp_path, items):
    out = tmp_path / "run"
    predictions = run(
        _backend(items),
        items,
        _config(),
        out,
        options=RunOptions(concurrency=2, log=lambda _: None),
    )
    assert [p.doc_id for p in predictions] == [i.doc_id for i in items]
    assert all(p.ok for p in predictions)
    assert predictions[0].usage == Usage(10, 5, 100, 0)
    assert predictions[0].latency_s == 0.25
    assert json.loads(predictions[1].raw_output) == {"doc": items[1].doc_id}
    assert load_config(out).condition == "test-cond"
    on_disk = load_predictions(out / PREDICTIONS_FILE)
    assert set(on_disk) == {i.doc_id for i in items}


def test_backend_failure_is_recorded_not_raised(tmp_path, items):
    out = tmp_path / "run"
    predictions = run(
        _backend(items, failing=[items[1].doc_id]), items, _config(), out, options=QUIET
    )
    failed = predictions[1]
    assert failed.ok is False
    assert "overloaded" in failed.error
    assert failed.raw_output is None


def test_resume_skips_answered_documents_and_retries_errors(tmp_path, items):
    out = tmp_path / "run"
    backend = _backend(items, failing=[items[1].doc_id])
    run(backend, items, _config(), out, options=QUIET)
    assert len(backend.requests) == 3

    backend = _backend(items)
    predictions = run(backend, items, _config(), out, options=QUIET)
    assert len(backend.requests) == 1  # only the errored document is retried
    assert all(p.ok for p in predictions)
    assert load_predictions(out / PREDICTIONS_FILE)[items[1].doc_id].ok


def test_resume_refuses_a_changed_prompt_or_model(tmp_path, items):
    out = tmp_path / "run"
    run(_backend(items), items, _config(), out, options=QUIET)
    with pytest.raises(RuntimeError, match="prompt_digest"):
        run(_backend(items), items, _config(prompt_digest="changed"), out, options=QUIET)
    with pytest.raises(RuntimeError, match="model"):
        run(_backend(items), items, _config(model="other"), out, options=QUIET)


def test_resume_refuses_a_changed_max_tokens(tmp_path, items):
    out = tmp_path / "run"
    run(_backend(items), items, _config(), out, options=QUIET)
    with pytest.raises(RuntimeError, match="max_tokens"):
        run(_backend(items), items, _config(max_tokens=99), out, options=QUIET)


def test_library_refuses_a_frozen_set_without_the_opt_in(tmp_path, items):
    frozen = [EvalItem(i.doc_id, "test_unseen", i.image_path, i.text, i.reference) for i in items]
    backend = _backend(frozen)
    with pytest.raises(HeldOutError, match="frozen held-out set"):
        run(backend, frozen, _config(eval_set="test_unseen"), tmp_path / "run", options=QUIET)
    assert backend.requests == []
    assert not (tmp_path / "run").exists()

    run(
        backend,
        frozen,
        _config(eval_set="test_unseen"),
        tmp_path / "run",
        options=RunOptions(held_out=True, log=lambda _: None),
    )
    assert len(backend.requests) == 3


def test_fresh_discards_the_old_run(tmp_path, items):
    out = tmp_path / "run"
    run(_backend(items), items, _config(), out, options=QUIET)
    backend = _backend(items)
    run(
        backend,
        items,
        _config(prompt_digest="changed"),
        out,
        options=RunOptions(fresh=True, log=lambda _: None),
    )
    assert len(backend.requests) == 3
    assert load_config(out).prompt_digest == "changed"


def test_resume_keeps_the_original_start_time(tmp_path, items):
    out = tmp_path / "run"
    run(
        _backend(items),
        items,
        _config(started_at="2026-01-01T00:00:00+00:00"),
        out,
        options=QUIET,
    )
    run(
        _backend(items),
        items,
        _config(started_at="2026-02-02T00:00:00+00:00"),
        out,
        options=QUIET,
    )
    assert load_config(out).started_at == "2026-01-01T00:00:00+00:00"


def test_text_input_without_a_text_layer_is_a_recorded_error(items):
    item = EvalItem(items[0].doc_id, "dev_unseen", items[0].image_path, None, items[0].reference)
    prediction = predict_item(_backend(items), item, _config(), example_image=None)
    assert prediction.ok is False
    assert "no text layer" in prediction.error


def test_prediction_round_trips_through_json():
    prediction = Prediction("d", "s", "c", "m", "{}", usage=Usage(1, 2, 3, 4), latency_s=1.5)
    assert Prediction.from_json(prediction.to_json()) == prediction
    errored = Prediction("d", "s", "c", "m", None, error="boom")
    assert Prediction.from_json(errored.to_json()) == errored


def test_frozen_sets_require_the_explicit_flag(capsys):
    code = main(["--set", "test_unseen", "--input", "image", "--condition", "x", "--model", "m"])
    assert code == 2
    assert "frozen held-out set" in capsys.readouterr().err


def test_openai_backend_requires_a_base_url(capsys):
    code = main(
        [
            "--set",
            "dev_unseen",
            "--input",
            "image",
            "--condition",
            "x",
            "--model",
            "m",
            "--backend",
            "openai",
        ]
    )
    assert code == 2
    assert "--base-url" in capsys.readouterr().err


def test_cli_runs_a_dev_set_end_to_end(tmp_path, items, monkeypatch, capsys):
    from eval import predict

    backend = _backend(items)
    monkeypatch.setattr(predict, "load_eval_set", lambda name: items)
    monkeypatch.setattr(predict, "make_backend", lambda args: backend)
    code = main(
        [
            "--set", "dev_unseen", "--input", "text", "--condition", "text-small",
            "--model", "fake-model", "--limit", "2", "--runs-dir", str(tmp_path), "--effort", "low",
        ]
    )  # fmt: skip
    assert code == 0
    assert len(backend.requests) == 2
    assert backend.requests[0].effort == "low"
    config = load_config(tmp_path / "text-small" / "dev_unseen")
    assert config.limit == 2 and config.input_kind == "text"
    assert len(config.prompt_digest) == 64
    assert "2 predictions" in capsys.readouterr().out


def test_cli_exit_code_reflects_request_errors(tmp_path, items, monkeypatch):
    from eval import predict

    backend = _backend(items, failing=[items[0].doc_id])
    monkeypatch.setattr(predict, "load_eval_set", lambda name: items)
    monkeypatch.setattr(predict, "make_backend", lambda args: backend)
    code = main(
        ["--set", "dev_unseen", "--input", "text", "--condition", "x", "--model", "m",
         "--runs-dir", str(tmp_path)]
    )  # fmt: skip
    assert code == 1


def test_resolve_api_key_prefers_explicit_flag_over_env(monkeypatch):
    from argparse import Namespace

    from eval.predict import resolve_api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
    assert resolve_api_key(Namespace(api_key="from-flag")) == "from-flag"


def test_resolve_api_key_falls_back_to_openrouter_then_openai_env(monkeypatch):
    from argparse import Namespace

    from eval.predict import resolve_api_key

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key(Namespace(api_key=None)) is None

    monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
    assert resolve_api_key(Namespace(api_key=None)) == "from-openai"

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-openrouter")
    assert resolve_api_key(Namespace(api_key=None)) == "from-openrouter"


def test_make_backend_openai_reads_base_url_and_resolved_key(monkeypatch):
    from argparse import Namespace

    from eval.predict import make_backend

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    args = Namespace(
        backend="openai", model="deepseek/deepseek-v4-flash-vision-exp",
        base_url="https://openrouter.ai/api/v1", api_key=None,
    )  # fmt: skip
    backend = make_backend(args)
    assert backend.model == "deepseek/deepseek-v4-flash-vision-exp"
    assert backend._client.api_key == "or-key"
    assert backend._client.base_url is not None and "openrouter.ai" in str(backend._client.base_url)


def test_run_config_records_the_schema_digest_and_older_configs_load_as_none():
    from eval.predict import RunConfig

    config = RunConfig("c", "dev_unseen", "image", "openai", "m", "p" * 64)
    assert len(config.schema_digest) == 64
    payload = json.loads(config.to_json())
    del payload["schema_digest"]
    assert RunConfig.from_json(json.dumps(payload)).schema_digest is None
    assert RunConfig.from_json(config.to_json()) == config
