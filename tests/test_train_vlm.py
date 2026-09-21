"""The training script's GPU-free parts: what it logs and the prompt it trains on."""

from dataclasses import dataclass

from eval.prompt import INSTRUCTION
from train.train_vlm import (
    DEFAULT_BASE_MODEL,
    TrainConfig,
    build_parser,
    build_run_config,
    config_from_args,
    sequence_length_kwarg,
    to_conversation,
)

ROW = {
    "doc_id": "Template1_Instance1",
    "layout_id": 1,
    "image": "images/x.jpg",
    "instruction": INSTRUCTION,
    "target": '{"invoice_number":"1","line_items":null}',
}
MANIFEST = {
    "created_at": "2026-09-20T00:00:00+00:00",
    "git_commit": "abc1234",
    "schema_digest": "s" * 64,
    "instruction": INSTRUCTION,
    "splits": {"train": {"n_examples": 2, "digest": "d" * 64, "trainable": True, "layouts": [1]}},
}


def test_conversation_is_image_then_instruction_then_the_target_verbatim():
    image = object()
    user, assistant = to_conversation(ROW, image)
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "image", "image": image},
        {"type": "text", "text": INSTRUCTION},
    ]
    assert assistant == {"role": "assistant", "content": [{"type": "text", "text": ROW["target"]}]}


def test_defaults_are_the_plans_starting_point():
    config = TrainConfig()
    assert (config.r, config.alpha, config.lr) == (16, 16, 2e-4)
    assert 1 <= config.epochs <= 2
    assert config.base_model == DEFAULT_BASE_MODEL
    assert config.tune_vision is False, "vLLM serves language-side LoRA; see the docstring"


def test_run_config_logs_every_knob_and_the_data_digest():
    config = TrainConfig(r=64, alpha=64, epochs=2, seed=7)
    run = build_run_config(
        config, MANIFEST, n_examples=2, started_at="t0", train_seconds=12.5, gpu="Tesla T4"
    )
    for key in ("r", "alpha", "lr", "epochs", "seed", "base_model", "split"):
        assert run[key] == getattr(config, key)
    assert run["dataset_digest"] == "d" * 64
    assert run["schema_digest"] == "s" * 64
    assert run["instruction"] == INSTRUCTION
    assert run["n_examples"] == 2 and run["gpu"] == "Tesla T4" and run["train_seconds"] == 12.5
    assert "versions" in run and "git_commit" in run


def test_cli_round_trips_into_the_config():
    args = build_parser().parse_args(
        ["--bundle", "b", "--out", "o", "--split", "train_4k", "--r", "4", "--alpha", "8",
         "--epochs", "2", "--no-4bit", "--tune-vision", "--limit", "8"]
    )  # fmt: skip
    config = config_from_args(args)
    assert config.split == "train_4k" and config.r == 4 and config.alpha == 8
    assert config.epochs == 2 and config.load_in_4bit is False and config.tune_vision is True
    assert config.limit == 8


def test_sequence_length_kwarg_follows_the_installed_trl():
    @dataclass
    class NewTrl:
        max_length: int = 0

    @dataclass
    class OldTrl:
        max_seq_length: int = 0

    assert sequence_length_kwarg(NewTrl, 2048) == {"max_length": 2048}
    assert sequence_length_kwarg(OldTrl, 2048) == {"max_seq_length": 2048}


def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    from train.train_vlm import TRAINER_DIR, latest_checkpoint

    assert latest_checkpoint(tmp_path) is None
    trainer = tmp_path / TRAINER_DIR
    for step in (50, 100, 150):
        (trainer / f"checkpoint-{step}").mkdir(parents=True)
    (trainer / "checkpoint-tmp").mkdir()  # a half-written or foreign dir is ignored
    assert latest_checkpoint(tmp_path) == trainer / "checkpoint-150"


def test_run_config_records_where_a_run_resumed_from():
    config = TrainConfig()
    run = build_run_config(
        config,
        MANIFEST,
        n_examples=2,
        started_at="t0",
        train_seconds=1.0,
        gpu=None,
        resumed_from="trainer/checkpoint-100",
    )
    assert run["resumed_from"] == "trainer/checkpoint-100"
    assert run["save_steps"] == config.save_steps


def test_cli_can_refuse_to_resume():
    args = build_parser().parse_args(
        ["--bundle", "b", "--out", "o", "--fresh", "--save-steps", "25"]
    )
    config = config_from_args(args)
    assert config.resume is False and config.save_steps == 25
