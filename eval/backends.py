"""Model backends behind one request/completion shape.

Two transports, one harness: the prompt, the page encoding and the scoring are identical
whichever backend answers, so conditions are compared on the model and nothing else.

* `AnthropicBackend`         - the official SDK; the prompted frontier baseline and the
                               Path A small-model pre-check.
* `OpenAICompatibleBackend`  - any `/v1/chat/completions` server, which is how the tuned
                               adapter is served (vLLM with `--enable-lora`). It also
                               carries the optional JSON-schema decoding constraint,
                               which vLLM applies and hosted APIs ignore.

Every completion records the token counts and the wall-clock latency of the call. Those
two numbers are the cost and p95 columns of the ship-gate table, so they are measured
here, at the boundary, and never estimated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from eval.pages import ImagePart

DEFAULT_MAX_TOKENS = 16_000
#: `response_format.json_schema` requires a name. It labels the schema and nothing reads it.
SCHEMA_RESPONSE_NAME = "invoice_record"
#: Batch eval runs hit rate limits; let the SDK's exponential backoff absorb them.
DEFAULT_MAX_RETRIES = 5


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


Part = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]
    #: Everything up to and including this message is a stable prefix worth caching.
    cache_breakpoint: bool = False


@dataclass(frozen=True, slots=True)
class Request:
    system: str
    messages: tuple[Message, ...]
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Anthropic `output_config.effort`; ignored by backends that have no such knob.
    effort: str | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage
    latency_s: float
    model: str
    stop_reason: str | None
    request_id: str | None = None


class BackendError(RuntimeError):
    """A request that failed after the SDK's own retries. Recorded per document, never fatal."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Backend(Protocol):
    name: str
    model: str

    def complete(self, request: Request) -> Completion: ...


def _translate(exc: Exception, sdk: Any) -> BackendError:
    """Map an SDK's typed exceptions onto `BackendError`. Both SDKs share the same three names."""
    if isinstance(exc, sdk.RateLimitError):
        return BackendError(f"rate limited: {exc}", retryable=True)
    if isinstance(exc, sdk.APIStatusError):
        return BackendError(f"HTTP {exc.status_code}: {exc}", retryable=exc.status_code >= 500)
    return BackendError(f"connection error: {exc}", retryable=True)


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str, *, client: Any = None, cache: bool = True) -> None:
        self.model = model
        self.cache = cache
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES)
        self._client = client

    def _parts(self, parts: tuple[Part, ...]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            else:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data,
                        },
                    }
                )
        return blocks

    def build_kwargs(self, request: Request) -> dict[str, Any]:
        """The exact `messages.create` arguments. Public so a test can inspect them."""
        ephemeral = {"type": "ephemeral"}
        system: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        if self.cache:
            system[-1]["cache_control"] = ephemeral

        messages = []
        for message in request.messages:
            blocks = self._parts(message.parts)
            if self.cache and message.cache_breakpoint and blocks:
                blocks[-1]["cache_control"] = ephemeral
            messages.append({"role": message.role, "content": blocks})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        # An empty system prompt is *no* system prompt (the adapter's minimal prompt).
        if request.system:
            kwargs["system"] = system
        if request.effort:
            kwargs["output_config"] = {"effort": request.effort}
        return kwargs

    def complete(self, request: Request) -> Completion:
        import anthropic

        kwargs = self.build_kwargs(request)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise _translate(exc, anthropic) from exc
        latency = time.perf_counter() - started

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            latency_s=latency,
            model=response.model,
            stop_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
        )


# --------------------------------------------------------------------------------------
# OpenAI-compatible (vLLM and friends)
# --------------------------------------------------------------------------------------


class OpenAICompatibleBackend:
    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        guided_json: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        # A JSON schema the server must decode against, masking every token that would
        # break it. Sent as the OpenAI `response_format`, which is a named parameter: an
        # endpoint that cannot honour it rejects the request, where an unknown `extra_body`
        # key is dropped in silence and decodes unconstrained under a config claiming
        # otherwise. `eval/evaluate.py` still treats an invalid output under it as proof
        # the constraint never applied.
        self._guided_json = guided_json
        if client is None:
            import openai

            # vLLM ignores the key but the client insists on one.
            client = openai.OpenAI(
                base_url=base_url, api_key=api_key or "unused", max_retries=DEFAULT_MAX_RETRIES
            )
        self._client = client

    @staticmethod
    def _parts(parts: tuple[Part, ...]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            else:
                url = f"data:{part.media_type};base64,{part.data}"
                blocks.append({"type": "image_url", "image_url": {"url": url}})
        return blocks

    def build_kwargs(self, request: Request) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": message.role, "content": self._parts(message.parts)}
            for message in request.messages
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": 0,
        }
        # Absent unless constrained, so an unconstrained request is exactly what the
        # earlier runs sent.
        if self._guided_json is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_RESPONSE_NAME,
                    "schema": self._guided_json,
                    "strict": True,
                },
            }
        return kwargs

    def complete(self, request: Request) -> Completion:
        import openai

        kwargs = self.build_kwargs(request)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            raise _translate(exc, openai) from exc
        latency = time.perf_counter() - started

        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        # Some providers (measured live: Gemini via this endpoint) spend tokens on hidden
        # reasoning that never shows up in `completion_tokens` - only `total_tokens`
        # reflects it, and it is billed at the completion rate. A provider that omits
        # `total_tokens`, or reports it as exactly prompt + completion, has nothing hidden.
        total_tokens = getattr(usage, "total_tokens", None)
        hidden_reasoning = (
            max(0, total_tokens - prompt_tokens - completion_tokens) if total_tokens else 0
        )
        return Completion(
            text=choice.message.content or "",
            usage=Usage(
                input_tokens=prompt_tokens - cached,
                output_tokens=completion_tokens + hidden_reasoning,
                cache_read_tokens=cached,
            ),
            latency_s=latency,
            model=response.model or self.model,
            stop_reason=choice.finish_reason,
            request_id=getattr(response, "id", None),
        )
