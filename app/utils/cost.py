"""
Cost calculation engine.
Prices are per 1M tokens (input / output) in USD.
Source: official pricing pages (OpenAI, Anthropic, Google, Mistral, etc.) — April 2026.
This built-in table is the fallback: services/pricing.py adds the current prices fetched
from the web (set_web_prices), which override it.
"""

import re
from functools import lru_cache

PRICES: dict[str, dict[str, float]] = {
    # ── OpenAI ────────────────────────────────────────────────────────
    # GPT-5.x
    "gpt-5.4":            {"in": 2.50,  "out": 15.00},
    "gpt-5.4-mini":       {"in": 0.75,  "out": 4.50},
    "gpt-5.4-nano":       {"in": 0.20,  "out": 1.25},
    # GPT-4.1
    "gpt-4.1":            {"in": 2.00,  "out": 8.00},
    "gpt-4.1-mini":       {"in": 0.40,  "out": 1.60},
    "gpt-4.1-nano":       {"in": 0.10,  "out": 0.40},
    # GPT-4o
    "gpt-4o":             {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":        {"in": 0.15,  "out": 0.60},
    # GPT-4 legacy
    "gpt-4-turbo":        {"in": 10.00, "out": 30.00},
    "gpt-4":              {"in": 30.00, "out": 60.00},
    "gpt-3.5-turbo":      {"in": 0.50,  "out": 1.50},
    # Reasoning
    "o3":                 {"in": 2.00,  "out": 8.00},
    "o4-mini":            {"in": 1.10,  "out": 4.40},
    "o1":                 {"in": 15.00, "out": 60.00},
    "o1-mini":            {"in": 3.00,  "out": 12.00},
    "o3-mini":            {"in": 1.10,  "out": 4.40},

    # ── Anthropic ─────────────────────────────────────────────────────
    "claude-opus-4.7":    {"in": 5.00,  "out": 25.00},
    "claude-opus-4.6":    {"in": 5.00,  "out": 25.00},
    "claude-opus-4.5":    {"in": 5.00,  "out": 25.00},
    "claude-opus-4.1":    {"in": 15.00, "out": 75.00},
    "claude-opus-4":      {"in": 15.00, "out": 75.00},
    "claude-sonnet-4.6":  {"in": 3.00,  "out": 15.00},
    "claude-sonnet-4.5":  {"in": 3.00,  "out": 15.00},
    "claude-sonnet-4":    {"in": 3.00,  "out": 15.00},
    "claude-haiku-4.5":   {"in": 1.00,  "out": 5.00},
    "claude-haiku-3.5":   {"in": 0.80,  "out": 4.00},
    "claude-haiku-3":     {"in": 0.25,  "out": 1.25},
    # Aliases
    "claude-3-7-sonnet":  {"in": 3.00,  "out": 15.00},
    "claude-3-5-sonnet":  {"in": 3.00,  "out": 15.00},
    "claude-3-5-haiku":   {"in": 0.80,  "out": 4.00},
    "claude-3-opus":      {"in": 15.00, "out": 75.00},
    "claude-3-sonnet":    {"in": 3.00,  "out": 15.00},
    "claude-3-haiku":     {"in": 0.25,  "out": 1.25},

    # ── Google ────────────────────────────────────────────────────────
    "gemini-2.5-pro":     {"in": 1.25,  "out": 10.00},
    "gemini-2.5-flash":   {"in": 0.15,  "out": 0.60},
    "gemini-2.0-flash":   {"in": 0.10,  "out": 0.40},
    "gemini-1.5-pro":     {"in": 1.25,  "out": 5.00},
    "gemini-1.5-flash":   {"in": 0.075, "out": 0.30},

    # ── Mistral ───────────────────────────────────────────────────────
    "mistral-large":      {"in": 2.00,  "out": 6.00},
    "mistral-small":      {"in": 0.20,  "out": 0.60},
    "mistral-medium":     {"in": 2.70,  "out": 8.10},
    "codestral":          {"in": 0.30,  "out": 0.90},

    # ── DeepSeek ──────────────────────────────────────────────────────
    "deepseek-chat":      {"in": 0.14,  "out": 0.28},
    "deepseek-reasoner":  {"in": 0.55,  "out": 2.19},

    # ── Cohere ────────────────────────────────────────────────────────
    "command-r-plus":     {"in": 2.50,  "out": 10.00},
    "command-r":          {"in": 0.50,  "out": 1.50},
    "command-a":          {"in": 2.50,  "out": 10.00},

    # ── Meta (via providers) ──────────────────────────────────────────
    "llama-4-maverick":   {"in": 0.20,  "out": 0.60},
    "llama-4-scout":      {"in": 0.15,  "out": 0.40},
    "llama-3.3-70b":      {"in": 0.60,  "out": 0.60},
    "llama-3.1-70b":      {"in": 0.90,  "out": 0.90},
    "llama-3.1-8b":       {"in": 0.20,  "out": 0.20},
}

DEFAULT_PRICE = {"in": 2.00, "out": 6.00}


# Anthropic API ids write versions with dashes ("claude-opus-4-5-20251101"), this table
# with dots ("claude-opus-4.5"). Only single-digit pairs are rewritten, so dates
# ("2024-08-06") and sizes ("llama-3.3-70b") stay intact.
_VERSION_DASH = re.compile(r"\b(\d)-(\d)\b")


def canonical_model_name(name: str) -> str:
    return _VERSION_DASH.sub(r"\1.\2", name.lower().strip())


_canonical = canonical_model_name

# Prices fetched from the web (services/pricing.py), keyed by canonical name. They override
# the built-in PRICES of the same model and add the models PRICES doesn't list.
_web_prices: dict[str, dict[str, float]] = {}
_table: dict[str, dict[str, float]] = dict(PRICES)
_match_order: list[tuple[str, str]] = []

_SEPARATORS = frozenset("/.:_-@ ")


def _rebuild() -> None:
    global _table, _match_order
    table = dict(PRICES)
    key_of = {_canonical(key): key for key in PRICES}
    for name, price in _web_prices.items():
        key = key_of.setdefault(_canonical(name), name)
        table[key] = price
    # Longest names first, so "gpt-4o-mini" is matched before its prefix "gpt-4o".
    _match_order = sorted(((_canonical(key), key) for key in table), key=lambda pair: len(pair[0]), reverse=True)
    _table = table
    _normalize_model_name.cache_clear()


def set_web_prices(prices: dict[str, dict[str, float]]) -> None:
    """Replace the web prices (per 1M tokens: in, out, and optionally cache_read, cache_write)."""
    global _web_prices
    _web_prices = dict(prices)
    _rebuild()


def web_price_count() -> int:
    return len(_web_prices)


def price_keys():
    """Keys of every priced model (built-in and web)."""
    return _table.keys()


def _contains(name: str, key: str) -> bool:
    """`key` appears in `name` as a whole name: "claude-opus-4" matches "claude-opus-4-20250514"
    but not "claude-opus-4.8-thinking" (another version) nor "gpt-4" in "gpt-4o"."""
    start = name.find(key)
    while start != -1:
        end = start + len(key)
        before_ok = start == 0 or name[start - 1] in _SEPARATORS
        nxt = name[end:end + 1]
        after_ok = not nxt or not (nxt.isalnum() or (nxt == "." and name[end + 1:end + 2].isdigit()))
        if before_ok and after_ok:
            return True
        start = name.find(key, start + 1)
    return False


@lru_cache(maxsize=4096)
def _normalize_model_name(raw: str) -> str:
    """Best-effort normalization of model identifiers to match the price table keys."""
    name = _canonical(raw)
    for canonical_key, key in _match_order:
        if _contains(name, canonical_key):
            return key
    return name


_rebuild()


# Prompt caching, relative to the input price (Anthropic: reads 0.1×, writes 1.25×), for
# models whose cache prices are not known.
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


def has_price(model: str) -> bool:
    """Whether the model is in the table (otherwise DEFAULT_PRICE is used)."""
    return _normalize_model_name(model or "") in _table


def price_key(model: str) -> str | None:
    """The price table key a model id resolves to, or None when its price is unknown."""
    key = _normalize_model_name(model or "")
    return key if key in _table else None


def _full(prices: dict[str, float]) -> dict[str, float]:
    return {
        "in": prices["in"],
        "out": prices["out"],
        "cache_read": prices.get("cache_read", prices["in"] * CACHE_READ_FACTOR),
        "cache_write": prices.get("cache_write", prices["in"] * CACHE_WRITE_FACTOR),
    }


def price_of(key: str) -> dict[str, float] | None:
    """Prices per 1M tokens (in, out, cache_read, cache_write) of a price table key."""
    prices = _table.get(key)
    return _full(prices) if prices else None


def compute_cost(model: str, tokens_in: int, tokens_out: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Compute cost in USD. Prices are per 1M tokens. `tokens_in` includes the cached
    tokens, which are billed at their own rate."""
    prices = _full(_table.get(_normalize_model_name(model or ""), DEFAULT_PRICE))
    cache_read = cache_read_tokens or 0
    cache_write = cache_write_tokens or 0
    uncached = max(0, (tokens_in or 0) - cache_read - cache_write)
    input_cost = uncached * prices["in"] + cache_read * prices["cache_read"] + cache_write * prices["cache_write"]
    return round((input_cost + (tokens_out or 0) * prices["out"]) / 1_000_000, 6)
