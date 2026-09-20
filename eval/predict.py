"""Run one condition on one eval set and record every raw output, token count and latency.

    # prompt iteration happens on the dev set, never on a held-out one
    python -m eval.predict --set dev_unseen --input image --model claude-opus-5 \\
        --condition vision-frontier --limit 20

    # the frontier baseline on the held-out sets (explicit flag required)
    python -m eval.predict --set test_unseen --input image --model claude-opus-5 \\
        --condition vision-frontier --held-out
    python -m eval.predict --set rvlcdip --input image --model claude-opus-5 \\
        --condition vision-frontier --held-out

    # Path A pre-check: shipped text layer -> small model
    python -m eval.predict --set test_unseen --input text --model claude-haiku-4-5 \\
        --condition text-small --held-out

    # the tuned adapter behind vLLM (phase 5) through the same harness
    python -m eval.predict --set test_unseen --input image --backend openai \\
        --base-url http://localhost:8000/v1 --model slotfill-lora \\
        --condition vision-adapter --held-out

Outputs land in `runs/<condition>/<set>/`: `run_config.json` (model, prompt digest, every
knob) and `predictions.jsonl` (one line per document). Runs resume: documents already
answered are skipped, documents that errored are retried. A run whose prompt or model has
changed since it was started is refused rather than silently mixed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.backends import (
    DEFAULT_MAX_TOKENS,
    AnthropicBackend,
    Backend,
    BackendError,
    OpenAICompatibleBackend,
    Usage,
)
from eval.datasets import (
    EVAL_SETS,
    FROZEN_SETS,
    EvalItem,
    excluded_fields_for,
    fatura_image,
    load_eval_set,
)
from eval.pages import ImagePart, encode_image
from eval.prompt import WORKED_EXAMPLE_DOC_ID, InputKind, build_request, prompt_digest, text_part

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
CONFIG_FILE = "run_config.json"
PREDICTIONS_FILE = "predictions.jsonl"
DEFAULT_CONCURRENCY = 4

#: Changing any of these mid-run would mix two experiments under one name.
_IDENTITY_KEYS = (
    "condition",
    "eval_set",
    "input_kind",
    "backend",
    "model",
    "prompt_digest",
    "effort",
    "max_tokens",
    "excluded_fields",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True, slots=True)
class RunConfig:
    condition: str
    eval_set: str
    input_kind: str
    backend: str
    model: str
    prompt_digest: str
    with_example: bool = True
    effort: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    excluded_fields: tuple[str, ...] = ()
    limit: int | None = None
    started_at: str = field(default_factory=_now)
    git_commit: str | None = field(default_factory=_git_commit)

    def identity(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _IDENTITY_KEYS}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> RunConfig:
        payload = json.loads(text)
        payload["excluded_fields"] = tuple(payload.get("excluded_fields", ()))
        return cls(**payload)


class HeldOutError(RuntimeError):
    """Raised when a frozen set is about to be run without an explicit, deliberate opt-in."""


@dataclass(frozen=True, slots=True)
class RunOptions:
    """How a run executes, as opposed to what it is (that is `RunConfig`)."""

    concurrency: int = DEFAULT_CONCURRENCY
    fresh: bool = False
    #: The only way to run a frozen set. Prompt iteration belongs on dev_unseen.
    held_out: bool = False
    log: Callable[[str], None] = print


DEFAULT_OPTIONS = RunOptions()


@dataclass(frozen=True, slots=True)
class Prediction:
    doc_id: str
    eval_set: str
    condition: str
    model: str
    raw_output: str | None
    error: str | None = None
    stop_reason: str | None = None
    usage: Usage | None = None
    latency_s: float | None = None
    request_id: str | None = None
    started_at: str = field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_output is not None

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Prediction:
        payload = json.loads(text)
        if payload.get("usage") is not None:
            payload["usage"] = Usage(**payload["usage"])
        return cls(**payload)


# --------------------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------------------


def predict_item(
    backend: Backend,
    item: EvalItem,
    config: RunConfig,
    *,
    example_image: ImagePart | None,
) -> Prediction:
    """Ask the model about one page. A backend failure is recorded, not raised."""
    if config.input_kind == "image":
        target = encode_image(item.image_path)
    else:
        if item.text is None:
            return Prediction(
                item.doc_id, item.eval_set, config.condition, config.model, None,
                error="no text layer for this document",
            )  # fmt: skip
        target = text_part(item.text)

    request = build_request(
        target,
        input_kind=config.input_kind,  # type: ignore[arg-type]
        example_image=example_image,
        with_example=config.with_example,
        effort=config.effort,
        max_tokens=config.max_tokens,
    )
    started = _now()
    try:
        completion = backend.complete(request)
    except BackendError as exc:
        return Prediction(
            item.doc_id, item.eval_set, config.condition, config.model, None,
            error=str(exc), started_at=started,
        )  # fmt: skip
    return Prediction(
        doc_id=item.doc_id,
        eval_set=item.eval_set,
        condition=config.condition,
        model=completion.model,
        raw_output=completion.text,
        stop_reason=completion.stop_reason,
        usage=completion.usage,
        latency_s=completion.latency_s,
        request_id=completion.request_id,
        started_at=started,
    )


# --------------------------------------------------------------------------------------
# A run
# --------------------------------------------------------------------------------------


def run_dir(condition: str, eval_set: str, runs_dir: Path = RUNS_DIR) -> Path:
    return runs_dir / condition / eval_set


def load_predictions(path: Path) -> dict[str, Prediction]:
    """Predictions keyed by document, last write wins (a retried document replaces its error)."""
    if not path.exists():
        return {}
    predictions: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                prediction = Prediction.from_json(line)
                predictions[prediction.doc_id] = prediction
    return predictions


def load_config(directory: Path) -> RunConfig:
    return RunConfig.from_json((directory / CONFIG_FILE).read_text(encoding="utf-8"))


def prepare_run_dir(
    config: RunConfig, directory: Path, *, fresh: bool = False
) -> dict[str, Prediction]:
    """Create or resume the run directory. Returns the predictions already on disk."""
    config_path = directory / CONFIG_FILE
    predictions_path = directory / PREDICTIONS_FILE
    if fresh:
        config_path.unlink(missing_ok=True)
        predictions_path.unlink(missing_ok=True)

    directory.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        existing = load_config(directory)
        if existing.identity() != config.identity():
            changed = [k for k in _IDENTITY_KEYS if getattr(existing, k) != getattr(config, k)]
            raise RuntimeError(
                f"{directory} holds a run with a different {', '.join(changed)}. Resuming would "
                "mix two experiments under one name; pass --fresh to discard it or use a new "
                "--condition."
            )
        # Keep the original start time and commit: the run is the same run.
        config = replace(config, started_at=existing.started_at, git_commit=existing.git_commit)
    config_path.write_text(config.to_json() + "\n", encoding="utf-8")
    return load_predictions(predictions_path)


def guard_held_out(eval_set: str, *, held_out: bool) -> None:
    """The library-level gate: the CLI flag is only one way to reach it."""
    if eval_set in FROZEN_SETS and not held_out:
        raise HeldOutError(
            f"{eval_set} is a frozen held-out set. Prompt iteration happens on dev_unseen; "
            "opt in with held_out=True (--held-out) only for a final, un-iterated run."
        )


def run(
    backend: Backend,
    items: Iterable[EvalItem],
    config: RunConfig,
    directory: Path,
    *,
    example_image: ImagePart | None = None,
    options: RunOptions = DEFAULT_OPTIONS,
) -> list[Prediction]:
    """Predict every item not already answered, appending to predictions.jsonl as it goes."""
    guard_held_out(config.eval_set, held_out=options.held_out)
    if example_image is None and config.input_kind == "image" and config.with_example:
        example_image = encode_image(fatura_image(WORKED_EXAMPLE_DOC_ID))
    done = prepare_run_dir(config, directory, fresh=options.fresh)
    pending = [item for item in items if not (item.doc_id in done and done[item.doc_id].ok)]
    options.log(
        f"{config.condition} on {config.eval_set}: {len(done)} on disk, {len(pending)} to run"
    )

    results = dict(done)
    with (
        (directory / PREDICTIONS_FILE).open("a", encoding="utf-8") as sink,
        ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool,
    ):
        futures = {
            pool.submit(predict_item, backend, item, config, example_image=example_image): item
            for item in pending
        }
        # Futures complete on worker threads; everything below runs on this thread only.
        for index, future in enumerate(as_completed(futures), start=1):
            prediction = future.result()
            sink.write(prediction.to_json() + "\n")
            sink.flush()
            results[prediction.doc_id] = prediction
            status = "ok" if prediction.ok else f"ERROR {prediction.error}"
            latency = f"{prediction.latency_s:.1f}s" if prediction.latency_s else "-"
            options.log(f"  [{index}/{len(pending)}] {prediction.doc_id} {latency} {status}")
    return [results[item.doc_id] for item in items if item.doc_id in results]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


#: Read in this order so a stray shell-history key never wins over an explicit --api-key,
#: and OpenRouter (this project's day-to-day choice: one balance, many models) is tried
#: before a generic OPENAI_API_KEY. A local vLLM server needs neither - it ignores auth.
_API_KEY_ENV_VARS: tuple[str, ...] = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


def resolve_api_key(args: argparse.Namespace) -> str | None:
    """The key for an OpenAI-compatible backend: --api-key, else the environment.

    Never put a real key on the command line - it lands in shell history and `ps`.
    `.env.example` documents `OPENROUTER_API_KEY` / `OPENAI_API_KEY` for this reason.
    """
    if args.api_key:
        return args.api_key
    for name in _API_KEY_ENV_VARS:
        if value := os.environ.get(name):
            return value
    return None


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "anthropic":
        return AnthropicBackend(args.model, cache=not args.no_cache)
    return OpenAICompatibleBackend(
        args.model, base_url=args.base_url, api_key=resolve_api_key(args)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--set", dest="eval_set", required=True, choices=EVAL_SETS)
    parser.add_argument("--input", dest="input_kind", required=True, choices=("image", "text"))
    parser.add_argument("--condition", required=True, help="run label, e.g. vision-frontier")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible server, e.g. http://localhost:8000/v1"
    )
    parser.add_argument(
        "--api-key",
        help="for --backend openai; defaults to $OPENROUTER_API_KEY or $OPENAI_API_KEY "
        "(the Anthropic SDK reads its own env var)",
    )
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-example", action="store_true", help="drop the worked example")
    parser.add_argument("--no-cache", action="store_true", help="disable prompt caching")
    parser.add_argument("--limit", type=int, help="only the first N documents (dev iteration)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--fresh", action="store_true", help="discard an existing run directory")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="required to run on a frozen set; iterate on dev_unseen instead",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        guard_held_out(args.eval_set, held_out=args.held_out)
    except HeldOutError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.backend == "openai" and not args.base_url:
        print("--backend openai needs --base-url", file=sys.stderr)
        return 2

    input_kind: InputKind = args.input_kind
    config = RunConfig(
        condition=args.condition,
        eval_set=args.eval_set,
        input_kind=input_kind,
        backend=args.backend,
        model=args.model,
        prompt_digest=prompt_digest(input_kind, with_example=not args.no_example),
        with_example=not args.no_example,
        effort=args.effort,
        max_tokens=args.max_tokens,
        excluded_fields=tuple(sorted(excluded_fields_for(args.eval_set, input_kind))),
        limit=args.limit,
    )

    items = load_eval_set(args.eval_set)
    if args.limit:
        items = items[: args.limit]
    directory = run_dir(args.condition, args.eval_set, args.runs_dir)
    predictions = run(
        make_backend(args),
        items,
        config,
        directory,
        options=RunOptions(concurrency=args.concurrency, fresh=args.fresh, held_out=args.held_out),
    )
    errors = sum(not p.ok for p in predictions)
    print(f"{len(predictions)} predictions in {directory} ({errors} errors)")
    print(f"next: python -m eval.evaluate {directory}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
