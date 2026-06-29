"""
Cost calculation engine.
Prices are per 1M tokens (input / output) in USD.
Source: official pricing pages (OpenAI, Anthropic, Google, Mistral, etc.) — April 2026.
"""

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
    "claude-sonnet-4.6":  {"in": 3.00,  "out": 15.00},
    "claude-sonnet-4.5":  {"in": 3.00,  "out": 15.00},
    "claude-sonnet-4":    {"in": 3.00,  "out": 15.00},
    "claude-haiku-4.5":   {"in": 1.00,  "out": 5.00},
    "claude-haiku-3.5":   {"in": 0.80,  "out": 4.00},
    "claude-haiku-3":     {"in": 0.25,  "out": 1.25},
    # Aliases
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


def _normalize_model_name(raw: str) -> str:
    """Best-effort normalization of model identifiers to match PRICES keys."""
    name = raw.lower().strip()
    for key in PRICES:
        if key in name:
            return key
    return name


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute cost in USD. Prices are per 1M tokens."""
    key = _normalize_model_name(model)
    prices = PRICES.get(key, DEFAULT_PRICE)
    return round(
        tokens_in * prices["in"] / 1_000_000 + tokens_out * prices["out"] / 1_000_000,
        6,
    )
