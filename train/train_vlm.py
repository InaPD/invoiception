"""Path B: Unsloth vision LoRA on a small VLM. Page image + fixed instruction -> JSON.

Runs on a GPU box (Colab/Kaggle T4 or a rented L4/A10), never locally:

    python -m train.train_vlm --bundle /content/slotfill-bundle --split train \\
        --out /content/adapters/qwen25vl3b-r16-train

    # the data-mix ablation: same knobs, the 4,025-document split
    python -m train.train_vlm --bundle /content/slotfill-bundle --split train_4k \\
        --out /content/adapters/qwen25vl3b-r16-train4k

    # smoke-test an adapter that already exists
    python -m train.train_vlm --bundle ... --out <adapter dir> --smoke-only

What lands in `--out`:

    adapter/            LoRA weights + adapter_config.json, loadable by vLLM `--enable-lora`
    run_config.json     every knob (r, alpha, lr, epochs, seed, base model, ...), the
                        dataset digest, library versions, git commit, wall-clock time
    train_log.json      the trainer's per-step log. Loss lives here and nowhere else - it is
                        never a headline result (CLAUDE.md)
    smoke.json          validity / field accuracy on N dev_unseen pages, via train/smoke.py

Design notes, each of which is a plan decision or a serving constraint:

* The prompt is `INSTRUCTION` and nothing else - no schema, no worked example. The schema
  goes into the weights; that is the whole cost argument. `eval/predict.py --no-example
  --no-schema` evaluates the adapter on exactly this prompt.
* Image first, then the instruction, matching `eval/prompt.build_request`. The chat
  template adds Qwen's default system turn on both sides, so training and serving see the
  same token stream.
* Loss on the assistant turn only (`train_on_responses_only`): the instruction is fixed,
  learning to predict it is wasted capacity.
* Vision layers are frozen by default. vLLM's LoRA support for multimodal models covers
  the language model; an adapter with weights in the vision tower may not load there, and
  a 3B model that cannot be served is not a result. `--tune-vision` exists for a
  comparison run that is scored offline.
* The bundle's `trainable` flag is enforced (`load_bundle_split(for_training=True)`) and
  the dataset digest is recorded, so no held-out layout can be trained on and every
  adapter names the exact examples it saw.
* Checkpoints every `--save-steps` optimizer steps into `<out>/trainer/checkpoint-N`, and
  a rerun with the same `--out` resumes from the latest one. Free Colab reclaims idle
  sessions and Kaggle wipes the working disk on a kernel crash, so `--out` should be on
  storage that survives the session (Drive, or Kaggle's saved output). `--fresh` starts
  over on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from train.export import BUNDLE_FILE, load_bundle_split
from train.smoke import score_smoke, summarise

#: Qwen2.5-VL 3B in Unsloth's repo; `load_in_4bit` quantises on the fly, so the adapter is
#: trained against weights that are name-compatible with `Qwen/Qwen2.5-VL-3B-Instruct`,
#: which is what vLLM serves as the base in phase 5.
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct"

#: Qwen2 chat template markers, for response-only loss.
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"

RUN_CONFIG_FILE = "run_config.json"
TRAIN_LOG_FILE = "train_log.json"
SMOKE_FILE = "smoke.json"
ADAPTER_DIR = "adapter"
#: HF Trainer checkpoints (`checkpoint-<step>/`: LoRA weights, optimizer, scheduler, RNG).
TRAINER_DIR = "trainer"
#: ~50 steps is ~400 examples at the default batch: 10-20 minutes of T4 time at risk.
DEFAULT_SAVE_STEPS = 50
SMOKE_SPLIT = "dev_unseen"
#: A 595x841 page is ~640 visual tokens; the instruction and a full record fit well inside.
DEFAULT_MAX_SEQ_LENGTH = 2048
#: Compact JSON records are ~250 tokens; this leaves room and still stops a runaway.
SMOKE_MAX_NEW_TOKENS = 768


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Every knob the plan asks to log, plus the ones that affect the result."""

    base_model: str = DEFAULT_BASE_MODEL
    split: str = "train"
    r: int = 16
    alpha: int = 16
    lr: float = 2e-4
    epochs: float = 1.0
    seed: int = 3407
    lora_dropout: float = 0.0
    batch_size: int = 2
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    load_in_4bit: bool = True
    tune_vision: bool = False
    save_steps: int = DEFAULT_SAVE_STEPS
    resume: bool = True  # pick up from the latest checkpoint in --out, if there is one
    smoke_n: int = 20
    limit: int | None = None  # a handful of examples, to shake the pipeline down


def to_conversation(row: dict[str, Any], image: Any) -> list[dict[str, Any]]:
    """Unsloth's vision message format. Image before text, as in `eval/prompt.py`."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["instruction"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]},
    ]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in ("unsloth", "unsloth_zoo", "torch", "transformers", "trl", "peft"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def build_run_config(
    config: TrainConfig,
    bundle_manifest: dict[str, Any],
    *,
    n_examples: int,
    started_at: str,
    train_seconds: float | None,
    gpu: str | None,
    resumed_from: str | None = None,
) -> dict[str, Any]:
    """`run_config.json`: the plan's (r, alpha, lr, epochs, seed) and everything around it."""
    split = bundle_manifest["splits"][config.split]
    return {
        **asdict(config),
        "n_examples": n_examples,
        "resumed_from": resumed_from,
        "dataset_digest": split["digest"],
        "schema_digest": bundle_manifest["schema_digest"],
        "instruction": bundle_manifest["instruction"],
        "bundle_created_at": bundle_manifest["created_at"],
        "bundle_git_commit": bundle_manifest["git_commit"],
        "git_commit": _git_commit(),
        "started_at": started_at,
        "train_seconds": train_seconds,
        "gpu": gpu,
        "versions": _versions(),
    }


# --------------------------------------------------------------------------------------
# GPU side. Nothing below imports cleanly without CUDA + unsloth; keep it behind functions.
# --------------------------------------------------------------------------------------


def _load_model(config: TrainConfig):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        config.base_model,
        load_in_4bit=config.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=config.max_seq_length,
    )
    return model, tokenizer


def _add_lora(model, config: TrainConfig):
    from unsloth import FastVisionModel

    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.tune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )


def _dataset(bundle_dir: Path, rows: list[dict[str, Any]]):
    """A list of conversations with the image loaded; ~2k pages fits in RAM comfortably."""
    from PIL import Image

    conversations = []
    for row in rows:
        with Image.open(bundle_dir / row["image"]) as page:
            image = page.convert("RGB")
        conversations.append({"messages": to_conversation(row, image)})
    return conversations


def sequence_length_kwarg(config_cls: Any, value: int) -> dict[str, int]:
    """TRL renamed `max_seq_length` to `max_length` (0.20); pass whichever this TRL has."""
    from dataclasses import fields

    names = {f.name for f in fields(config_cls)}
    key = "max_length" if "max_length" in names else "max_seq_length"
    return {key: value}


def latest_checkpoint(out_dir: Path) -> Path | None:
    """The highest-numbered `trainer/checkpoint-N` under `out_dir`, or None."""
    trainer_dir = out_dir / TRAINER_DIR
    if not trainer_dir.is_dir():
        return None
    checkpoints = [
        (int(p.name.rsplit("-", 1)[1]), p)
        for p in trainer_dir.glob("checkpoint-*")
        if p.is_dir() and p.name.rsplit("-", 1)[1].isdigit()
    ]
    return max(checkpoints)[1] if checkpoints else None


def processor_kwarg(trainer_cls: Any, processor: Any) -> dict[str, Any]:
    """TRL renamed `SFTTrainer(tokenizer=...)` to `processing_class=...`; pass the right one."""
    import inspect

    params = inspect.signature(trainer_cls.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return {key: processor}


def warmup_steps(config: TrainConfig, *, n_examples: int) -> int:
    """`warmup_ratio` is deprecated in TRL; the same share, as a step count, at least 1."""
    per_epoch = math.ceil(n_examples / (config.batch_size * config.grad_accum))
    total = math.ceil(per_epoch * config.epochs)
    return max(1, round(config.warmup_ratio * total))


def train(config: TrainConfig, bundle_dir: Path, out_dir: Path) -> Path:
    # Unsloth patches trl/transformers/peft and must be imported before them.
    # isort: off
    from unsloth import is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTConfig, SFTTrainer
    # isort: on

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    rows = load_bundle_split(bundle_dir, config.split, for_training=True)
    if config.limit:
        rows = rows[: config.limit]
    digest = manifest["splits"][config.split]["digest"]
    print(f"{config.split}: {len(rows)} examples, digest {digest[:12]}")

    model, tokenizer = _load_model(config)
    model = _add_lora(model, config)
    dataset = _dataset(bundle_dir, rows)

    collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        # The default ("min") reads the model's image size, finds none for Qwen2.5-VL and
        # shrinks every page to 512 on one side. Inference sees the full 595x841 page, so
        # training must too; "max" means no resizing.
        resize="max",
        train_on_responses_only=True,
        instruction_part=QWEN_INSTRUCTION_PART,
        response_part=QWEN_RESPONSE_PART,
    )
    trainer = SFTTrainer(
        model=model,
        **processor_kwarg(SFTTrainer, tokenizer),
        data_collator=collator,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            num_train_epochs=config.epochs,
            learning_rate=config.lr,
            warmup_steps=warmup_steps(config, n_examples=len(rows)),
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler,
            seed=config.seed,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            optim="adamw_8bit",
            logging_steps=5,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=2,
            report_to="none",
            output_dir=str(out_dir / TRAINER_DIR),
            # Required for vision: the collator does the tokenising, not the trainer.
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            **sequence_length_kwarg(SFTConfig, config.max_seq_length),
        ),
    )

    if not config.resume and (out_dir / TRAINER_DIR).exists():
        import shutil

        shutil.rmtree(out_dir / TRAINER_DIR)  # --fresh means it: no stale checkpoints
    checkpoint = latest_checkpoint(out_dir) if config.resume else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"training on {gpu}; {len(dataset)} examples, {config.epochs} epoch(s)")
    if checkpoint:
        print(f"resuming from {checkpoint}")
    tic = time.perf_counter()
    stats = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    # Wall-clock of *this* session only; a resumed run's total is the sum over sessions.
    train_seconds = time.perf_counter() - tic

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / ADAPTER_DIR))
    tokenizer.save_pretrained(str(out_dir / ADAPTER_DIR))
    (out_dir / TRAIN_LOG_FILE).write_text(
        json.dumps({"metrics": stats.metrics, "log_history": trainer.state.log_history}, indent=1)
    )
    run_config = build_run_config(
        config,
        manifest,
        n_examples=len(rows),
        started_at=started_at,
        train_seconds=train_seconds,
        gpu=gpu,
        resumed_from=str(checkpoint.relative_to(out_dir)) if checkpoint else None,
    )
    (out_dir / RUN_CONFIG_FILE).write_text(json.dumps(run_config, indent=1) + "\n")
    print(f"adapter saved to {out_dir / ADAPTER_DIR} after {train_seconds / 60:.1f} min")

    smoke(model, tokenizer, bundle_dir, out_dir, n=config.smoke_n)
    return out_dir


def _generate(model, tokenizer, image, instruction: str) -> str:
    """Greedy decoding on one page with the minimal prompt, the way vLLM will see it."""
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs, max_new_tokens=SMOKE_MAX_NEW_TOKENS, do_sample=False, use_cache=True
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def smoke(model, tokenizer, bundle_dir: Path, out_dir: Path, *, n: int) -> dict[str, Any]:
    """Generate for the first `n` dev_unseen pages and score them. Writes smoke.json."""
    from PIL import Image

    rows = load_bundle_split(bundle_dir, SMOKE_SPLIT, for_training=False)[:n]
    results, outputs = [], []
    for row in rows:
        with Image.open(bundle_dir / row["image"]) as page:
            image = page.convert("RGB")
        tic = time.perf_counter()
        text = _generate(model, tokenizer, image, row["instruction"])
        latency = time.perf_counter() - tic
        result = score_smoke(row["doc_id"], text, row["target"])
        results.append(result)
        outputs.append({**asdict(result), "latency_s": latency, "raw_output": text})
        mark = "ok " if result.valid else "BAD"
        print(f"  {mark} {row['doc_id']} {result.correct}/{result.scoreable} {latency:.1f}s")

    summary = {"split": SMOKE_SPLIT, **summarise(results), "documents": outputs}
    (out_dir / SMOKE_FILE).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(
        f"smoke: validity {summary['validity_rate']:.0%}, "
        f"field accuracy {summary['field_accuracy_all']:.0%}, "
        f"exact match {summary['exact_match_all']:.0%} over {summary['n_documents']} pages"
    )
    return summary


def smoke_only(config: TrainConfig, bundle_dir: Path, out_dir: Path) -> dict[str, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        str(out_dir / ADAPTER_DIR),
        load_in_4bit=config.load_in_4bit,
        max_seq_length=config.max_seq_length,
    )
    return smoke(model, tokenizer, bundle_dir, out_dir, n=config.smoke_n)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True, help="unpacked slotfill-bundle/")
    parser.add_argument("--out", type=Path, required=True, help="adapter + logs land here")
    parser.add_argument("--split", default=defaults.split, help="train or train_4k")
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--r", type=int, default=defaults.r)
    parser.add_argument("--alpha", type=int, default=defaults.alpha)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--epochs", type=float, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--no-4bit", action="store_true", help="16-bit base (needs >16GB)")
    parser.add_argument(
        "--tune-vision", action="store_true", help="also adapt the vision tower (see docstring)"
    )
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument(
        "--fresh", action="store_true", help="ignore checkpoints in --out and start over"
    )
    parser.add_argument("--smoke-n", type=int, default=defaults.smoke_n)
    parser.add_argument("--limit", type=int, help="train on the first N examples only")
    parser.add_argument("--smoke-only", action="store_true", help="skip training; score --out")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        base_model=args.base_model,
        split=args.split,
        r=args.r,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        tune_vision=args.tune_vision,
        save_steps=args.save_steps,
        resume=not args.fresh,
        smoke_n=args.smoke_n,
        limit=args.limit,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.smoke_only:
        smoke_only(config, args.bundle, args.out)
    else:
        train(config, args.bundle, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
