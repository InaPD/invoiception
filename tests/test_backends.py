"""Backends translate one request shape faithfully and record usage and latency."""

from types import SimpleNamespace

import pytest

from eval.backends import (
    AnthropicBackend,
    BackendError,
    Message,
    OpenAICompatibleBackend,
    Request,
    TextPart,
    Usage,
)
from eval.pages import ImagePart

IMAGE = ImagePart("image/png", "AAAA", 1, 1)
REQUEST = Request(
    system="SYSTEM",
    messages=(
        Message("user", (IMAGE, TextPart("extract"))),
        Message("assistant", (TextPart("{}"),), cache_breakpoint=True),
        Message("user", (IMAGE, TextPart("extract"))),
    ),
    effort="medium",
)


def test_usage_adds():
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)


class _FakeAnthropic:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


def _anthropic_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text='{"a": 1}'),
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=0,
        ),
        model="claude-opus-5",
        stop_reason="end_turn",
        _request_id="req_1",
    )


def test_anthropic_builds_blocks_with_cache_breakpoints():
    fake = _FakeAnthropic(_anthropic_response())
    kwargs = AnthropicBackend("claude-opus-5", client=fake).build_kwargs(REQUEST)
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert kwargs["output_config"] == {"effort": "medium"}
    user, assistant, target = kwargs["messages"]
    assert user["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    assert assistant["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in target["content"][-1]
    assert "thinking" not in kwargs and "temperature" not in kwargs


def test_anthropic_cache_can_be_switched_off():
    kwargs = AnthropicBackend("claude-opus-5", client=_FakeAnthropic(), cache=False).build_kwargs(
        REQUEST
    )
    assert "cache_control" not in kwargs["system"][0]
    assert all("cache_control" not in b for m in kwargs["messages"] for b in m["content"])


def test_anthropic_completion_keeps_text_only_and_records_usage():
    fake = _FakeAnthropic(_anthropic_response())
    completion = AnthropicBackend("claude-opus-5", client=fake).complete(REQUEST)
    assert completion.text == '{"a": 1}'
    assert completion.usage == Usage(100, 20, cache_read_tokens=900, cache_write_tokens=0)
    assert completion.stop_reason == "end_turn"
    assert completion.request_id == "req_1"
    assert completion.latency_s >= 0


def test_anthropic_errors_become_backend_errors():
    import anthropic
    import httpx2  # anthropic 1.x's transport; an SDK upgrade that swaps it breaks only this test

    response = httpx2.Response(status_code=529, request=httpx2.Request("POST", "http://x"))
    err = anthropic.APIStatusError("overloaded", response=response, body=None)
    fake = _FakeAnthropic(error=err)
    with pytest.raises(BackendError) as info:
        AnthropicBackend("claude-opus-5", client=fake).complete(REQUEST)
    assert info.value.retryable is True


def test_rate_limit_and_client_errors_are_classified():
    import anthropic
    import httpx2

    def status(code):
        return httpx2.Response(status_code=code, request=httpx2.Request("POST", "http://x"))

    limited = anthropic.RateLimitError("slow down", response=status(429), body=None)
    with pytest.raises(BackendError) as info:
        AnthropicBackend("claude-opus-5", client=_FakeAnthropic(error=limited)).complete(REQUEST)
    assert info.value.retryable is True and "rate limited" in str(info.value)

    bad = anthropic.BadRequestError("nope", response=status(400), body=None)
    with pytest.raises(BackendError) as info:
        AnthropicBackend("claude-opus-5", client=_FakeAnthropic(error=bad)).complete(REQUEST)
    assert info.value.retryable is False and "HTTP 400" in str(info.value)


class _FakeOpenAI:
    def __init__(self, response):
        self.calls = []
        self._response = response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_openai_compatible_builds_data_urls_and_records_usage():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        ),
        model="adapter",
        id="chatcmpl-1",
    )
    fake = _FakeOpenAI(response)
    backend = OpenAICompatibleBackend("adapter", client=fake)
    completion = backend.complete(REQUEST)
    kwargs = fake.calls[0]
    assert kwargs["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert kwargs["messages"][1]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    assert kwargs["temperature"] == 0
    assert completion.usage == Usage(600, 5, cache_read_tokens=400)
    assert completion.stop_reason == "stop"
    assert completion.model == "adapter"


def test_openai_compatible_folds_hidden_reasoning_tokens_into_output():
    """Some providers (observed: Gemini via its OpenAI-compat endpoint) spend tokens on
    reasoning that never appears in `completion_tokens` - only `total_tokens` reflects it,
    and it is billed at the completion rate. Measured live: a 1-token visible reply that
    actually cost 78 total tokens. Dropping that gap understates cost by a wide margin."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=6, completion_tokens=1, total_tokens=78),
        model="gemini-3.8-flash",
        id="resp-1",
    )
    completion = OpenAICompatibleBackend("gemini-3.8-flash", client=_FakeOpenAI(response)).complete(
        REQUEST
    )
    assert completion.usage.input_tokens == 6
    assert completion.usage.output_tokens == 72  # 1 visible + 71 hidden reasoning


def test_openai_compatible_without_a_total_tokens_field_assumes_no_hidden_gap():
    """A provider that never reports `total_tokens` (or reports it as exactly
    prompt + completion) must not have output inflated - there is nothing hidden to fold in."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        model="adapter",
        id="resp-2",
    )
    completion = OpenAICompatibleBackend("adapter", client=_FakeOpenAI(response)).complete(REQUEST)
    assert completion.usage == Usage(100, 20)


MINIMAL_REQUEST = Request(system="", messages=(Message("user", (IMAGE, TextPart("extract"))),))


def test_openai_compatible_omits_an_empty_system_message():
    """An empty system message is not 'no system prompt': the server's chat template would
    render an empty system turn, which the adapter never saw in training."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        model="adapter",
        id="chatcmpl-2",
    )
    fake = _FakeOpenAI(response)
    OpenAICompatibleBackend("adapter", client=fake).complete(MINIMAL_REQUEST)
    roles = [m["role"] for m in fake.calls[0]["messages"]]
    assert roles == ["user"]


def test_anthropic_omits_an_empty_system_prompt():
    fake = _FakeAnthropic(_anthropic_response())
    AnthropicBackend("claude", client=fake).complete(MINIMAL_REQUEST)
    assert "system" not in fake.calls[0]


def test_openai_compatible_sends_no_extra_body_when_unconstrained():
    """An unconstrained request must stay byte-identical to what the committed runs sent,
    so turning the feature on cannot quietly re-shape every other condition."""
    kwargs = OpenAICompatibleBackend("adapter", client=_FakeOpenAI(None)).build_kwargs(REQUEST)
    assert "extra_body" not in kwargs


def test_openai_compatible_passes_the_schema_as_guided_json():
    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    backend = OpenAICompatibleBackend("adapter", client=_FakeOpenAI(None), guided_json=schema)
    kwargs = backend.build_kwargs(REQUEST)
    assert kwargs["extra_body"] == {"guided_json": schema}
    assert kwargs["temperature"] == 0
