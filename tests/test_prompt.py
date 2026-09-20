"""The prompt is part of the experiment; its example must be legal and its digest stable."""

import json

import pytest

from data.split import SPLIT_DIR, load_manifest
from eval.backends import TextPart
from eval.pages import ImagePart
from eval.prompt import (
    EXAMPLE_RECORD,
    EXAMPLE_TEXT,
    INSTRUCTION,
    WORKED_EXAMPLE_DOC_ID,
    build_request,
    prompt_digest,
    system_prompt,
    worked_example,
)
from schema.validate import validate_record

IMAGE = ImagePart("image/jpeg", "QUJD", 595, 841)


def test_example_record_validates_against_the_schema():
    outcome = validate_record(EXAMPLE_RECORD)
    assert outcome.ok, [str(e) for e in outcome.errors]


def test_example_comes_from_the_train_split_never_a_held_out_one():
    train = load_manifest(SPLIT_DIR / "train.json")
    assert WORKED_EXAMPLE_DOC_ID in train.doc_ids
    for frozen in ("test_seen", "test_unseen", "dev_unseen"):
        assert WORKED_EXAMPLE_DOC_ID not in load_manifest(SPLIT_DIR / f"{frozen}.json").doc_ids


def test_example_text_agrees_with_the_example_record():
    assert EXAMPLE_RECORD["vendor"]["name"] in EXAMPLE_TEXT
    assert EXAMPLE_RECORD["invoice_number"] in EXAMPLE_TEXT
    assert f"{EXAMPLE_RECORD['total_amount']:.2f}" in EXAMPLE_TEXT
    for item in EXAMPLE_RECORD["line_items"]:
        assert item["description"] in EXAMPLE_TEXT


def test_system_prompt_embeds_the_schema_and_the_no_fence_rule():
    prompt = system_prompt()
    assert '"additionalProperties": false' in prompt
    assert "line_items" in prompt
    assert "no markdown fence" in prompt


def test_image_request_shape():
    request = build_request(IMAGE, input_kind="image", example_image=IMAGE, effort="low")
    assert request.system == system_prompt()
    assert request.effort == "low"
    example_user, example_answer, target = request.messages
    assert example_user.role == "user" and example_user.parts[0] is IMAGE
    assert example_user.parts[1] == TextPart(INSTRUCTION)
    assert example_answer.role == "assistant" and example_answer.cache_breakpoint is True
    assert json.loads(example_answer.parts[0].text) == EXAMPLE_RECORD
    assert target.role == "user" and target.cache_breakpoint is False


def test_text_request_uses_the_transcription_not_an_image():
    request = build_request(TextPart("<page_text>\nx\n</page_text>"), input_kind="text")
    example_user = request.messages[0]
    assert isinstance(example_user.parts[0], TextPart)
    assert "Brooks LLC" in example_user.parts[0].text
    assert "<page_text>" in example_user.parts[0].text


def test_image_example_requires_the_page():
    with pytest.raises(ValueError, match="page image"):
        worked_example("image")


def test_request_without_example_has_a_single_turn():
    request = build_request(IMAGE, input_kind="image", with_example=False)
    assert len(request.messages) == 1


def test_prompt_digest_is_stable_and_distinguishes_variants():
    assert prompt_digest("image") == prompt_digest("image")
    assert prompt_digest("image") != prompt_digest("text")
    assert prompt_digest("image") != prompt_digest("image", with_example=False)
    assert len(prompt_digest("image")) == 64
