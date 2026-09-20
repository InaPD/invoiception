"""Cost per 1,000 invoices is a headline number; the arithmetic behind it gets a test."""

import pytest

from eval.backends import Usage
from eval.pricing import PRICES, cost_usd, price_for


def test_known_model_has_all_four_rates():
    price = price_for("claude-opus-5")
    assert price is not None
    assert price.input_per_mtok == 5.0
    assert price.output_per_mtok == 25.0
    assert price.cache_read_per_mtok == pytest.approx(0.5)
    assert price.cache_write_per_mtok == pytest.approx(6.25)


def test_unknown_model_has_no_price():
    assert price_for("qwen2.5-vl-3b-lora") is None
    assert cost_usd("qwen2.5-vl-3b-lora", Usage(1000, 100)) is None


def test_cost_counts_every_token_class():
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=500_000,
    )
    assert cost_usd("claude-opus-5", usage) == pytest.approx(5.0 + 2.5 + 1.0 + 3.125)


def test_every_listed_price_is_positive():
    for model, price in PRICES.items():
        assert price.input_per_mtok > 0, model
        assert price.output_per_mtok > 0, model


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "google/gemini-3.8-flash"])
def test_gemini_3_8_flash_is_priced_under_both_names_it_actually_arrives_as(model):
    price = price_for(model)
    assert price is not None
    assert price.input_per_mtok == 0.75
    assert price.output_per_mtok == 3.75


def test_gemini_prices_a_cache_write_below_input_cost_unlike_anthropic():
    """Real, verified behaviour, not a copy-paste of the Anthropic formula: Google's cache
    write is *cheaper* than a fresh input token, the opposite of Anthropic's 1.25x markup."""
    price = price_for("gemini-3.8-flash")
    assert price.cache_write_per_mtok < price.input_per_mtok
    assert price.cache_read_per_mtok == pytest.approx(price.input_per_mtok * 0.1)
