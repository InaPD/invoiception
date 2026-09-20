"""Published API prices. Cost per 1,000 invoices = measured tokens x these rates.

Self-hosted conditions have no entry here on purpose: their cost is GPU $/hour divided by
measured throughput, which is a different measurement and is recorded by the serving
harness, not looked up in a table.

Rates are USD per million tokens, as of 2026-06 unless noted. Anthropic's cache reads are
billed at 0.1x input and cache writes at 1.25x input (5-minute TTL) - other providers price
caching differently and are given their own real rates rather than reused off that formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from eval.backends import Usage

_MTOK = 1_000_000


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


def _anthropic(input_per_mtok: float, output_per_mtok: float) -> Price:
    return Price(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_read_per_mtok=input_per_mtok * 0.1,
        cache_write_per_mtok=input_per_mtok * 1.25,
    )


#: $0.75/$3.75 confirmed independently from both ai.google.dev/gemini-api/docs/pricing and
#: OpenRouter's own per-model endpoint; cache rates from the latter (Google prices a cache
#: write below input cost, the opposite of Anthropic's markup - a real difference, not a
#: typo). Listed under both the bare Google name (direct API) and the OpenRouter-prefixed
#: name, since `--model` is whatever string the run actually used.
_GEMINI_3_8_FLASH = Price(
    input_per_mtok=0.75,
    output_per_mtok=3.75,
    cache_read_per_mtok=0.075,
    cache_write_per_mtok=0.0416667,
)

PRICES = MappingProxyType(
    {
        "claude-opus-5": _anthropic(5.0, 25.0),
        "claude-opus-4-8": _anthropic(5.0, 25.0),
        "claude-sonnet-5": _anthropic(2.0, 10.0),
        "claude-sonnet-4-6": _anthropic(3.0, 15.0),
        "claude-haiku-4-5": _anthropic(1.0, 5.0),
        "gemini-3.8-flash": _GEMINI_3_8_FLASH,
        "google/gemini-3.8-flash": _GEMINI_3_8_FLASH,
    }
)


def price_for(model: str) -> Price | None:
    return PRICES.get(model)


def cost_usd(model: str, usage: Usage) -> float | None:
    """Dollar cost of one request, or None when the model has no published per-token price."""
    price = price_for(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_tokens * price.cache_read_per_mtok
        + usage.cache_write_tokens * price.cache_write_per_mtok
    ) / _MTOK
