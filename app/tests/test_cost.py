import pytest

from utils.cost import _normalize_model_name, compute_cost, price_key, price_of, set_web_prices


@pytest.fixture
def web_prices():
    yield set_web_prices
    set_web_prices({})


@pytest.mark.parametrize("model, expected", [
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("gpt-4o-2024-08-06", "gpt-4o"),
    ("openai/gpt-4.1-mini", "gpt-4.1-mini"),
    ("gpt-4.1-nano", "gpt-4.1-nano"),
    ("gpt-5.4-mini", "gpt-5.4-mini"),
    ("gpt-5.4-nano", "gpt-5.4-nano"),
    ("o1-mini", "o1-mini"),
    ("o3-mini", "o3-mini"),
    ("o3", "o3"),
])
def test_longest_matching_name_wins(model, expected):
    assert _normalize_model_name(model) == expected


@pytest.mark.parametrize("model, expected", [
    ("claude-opus-4-5-20251101", "claude-opus-4.5"),
    ("claude-opus-4-1-20250805", "claude-opus-4.1"),
    ("claude-opus-4-20250514", "claude-opus-4"),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4.5"),
    ("claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("claude-haiku-4-5-20251001", "claude-haiku-4.5"),
    ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
    ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", "claude-3-7-sonnet"),
    ("claude-3-haiku-20240307", "claude-3-haiku"),
])
def test_claude_ids_written_with_dashes(model, expected):
    assert _normalize_model_name(model) == expected


def test_dates_and_model_sizes_are_not_rewritten():
    assert _normalize_model_name("llama-3.3-70b-versatile") == "llama-3.3-70b"
    assert _normalize_model_name("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b"


def test_compute_cost_uses_the_matching_price():
    assert compute_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert compute_cost("claude-opus-4-5-20251101", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert compute_cost("unknown-model", 1_000_000, 0) == pytest.approx(2.0)


@pytest.mark.parametrize("model", [
    "claude-opus-4-8-thinking-high",
    "gpt-4.5-preview",
    "grok-4.7-high-fast",
    "chatgpt-4o-latest",
])
def test_another_version_is_not_billed_as_a_known_one(model):
    assert price_key(model) is None


def test_web_prices_add_models_and_override_builtin_ones(web_prices):
    web_prices({
        "claude-opus-5.5": {"in": 5.0, "out": 25.0, "cache_read": 0.5},
        "gpt-4o": {"in": 3.0, "out": 12.0},
        "claude-3.7-sonnet": {"in": 4.0, "out": 16.0},
    })
    assert price_key("claude-opus-5-5-high") == "claude-opus-5.5"
    assert compute_cost("claude-opus-5-5-high", 1_000_000, 0, cache_read_tokens=1_000_000) == pytest.approx(0.5)
    assert price_key("gpt-4o-2024-08-06") == "gpt-4o"
    assert compute_cost("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(3.0)
    assert price_key("claude-3-7-sonnet-20250219") == "claude-3-7-sonnet"
    assert price_of("claude-3-7-sonnet")["in"] == 4.0
    assert price_key("gpt-4o-mini") == "gpt-4o-mini"


def test_cache_prices_default_to_the_input_price_factors():
    price = price_of("claude-sonnet-4.5")
    assert price["cache_read"] == pytest.approx(0.3)
    assert price["cache_write"] == pytest.approx(3.75)
    assert price_of("my-local-model") is None
