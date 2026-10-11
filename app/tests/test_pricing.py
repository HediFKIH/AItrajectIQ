import asyncio
import json

import httpx
import pytest

from services import pricing
from utils.cost import price_key, set_web_prices


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AITRAJECTIQ_PRICES_CACHE", str(tmp_path / "prices.json"))
    yield
    set_web_prices({})
    pricing._status.update(source="builtin", updated_at=None, models=0, sources={}, error=None)


def openrouter_payload(n=60) -> dict:
    models = [{"id": f"vendor/model-{i}", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
              for i in range(n)]
    models.append({"id": "anthropic/claude-opus-5.5",
                   "pricing": {"prompt": "0.000005", "completion": "0.000025", "input_cache_read": "0.0000005"}})
    return {"data": models}


@pytest.mark.parametrize("model_id, expected", [
    ("anthropic/claude-opus-4-5-20251101", ("claude-opus-4.5", True)),
    ("gpt-4o-2024-08-06", ("gpt-4o", True)),
    ("mistral/mistral-large-latest", ("mistral-large", True)),
    ("xai/grok-4", ("grok-4", False)),
    ("gemini/gemini-2.5-pro", ("gemini-2.5-pro", False)),
])
def test_base_name(model_id, expected):
    assert pricing.base_name(model_id) == expected


def test_parse_litellm_keeps_direct_providers_and_prefers_undated_ids():
    spec = {"litellm_provider": "anthropic", "mode": "chat"}
    data = {
        "sample_spec": {"litellm_provider": "one of https://docs.litellm.ai/docs/providers", "mode": "chat"},
        "claude-opus-4-5-20251101": {**spec, "input_cost_per_token": 6e-6, "output_cost_per_token": 3e-5},
        "claude-opus-4-5": {**spec, "input_cost_per_token": 5e-6, "output_cost_per_token": 2.5e-5,
                            "cache_read_input_token_cost": 5e-7, "cache_creation_input_token_cost": 6.25e-6},
        "bedrock/anthropic.claude-opus-4-5": {"litellm_provider": "bedrock", "mode": "chat",
                                              "input_cost_per_token": 1e-5, "output_cost_per_token": 1e-5},
        "text-embedding-3-small": {"litellm_provider": "openai", "mode": "embedding",
                                   "input_cost_per_token": 2e-8, "output_cost_per_token": 0},
        "ft:gpt-4o": {"litellm_provider": "openai", "mode": "chat",
                      "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6},
        "gemini/gemini-2.5-pro": {"litellm_provider": "gemini", "mode": "chat",
                                  "input_cost_per_token": 1.25e-6, "output_cost_per_token": 1e-5},
    }
    prices = pricing.parse_litellm(data)
    assert set(prices) == {"claude-opus-4.5", "gemini-2.5-pro"}
    assert prices["claude-opus-4.5"] == {"in": 5.0, "out": 25.0, "cache_read": 0.5, "cache_write": 6.25}


def test_parse_openrouter_skips_variants_free_and_invalid_prices():
    data = {"data": [
        {"id": "anthropic/claude-opus-5.5",
         "pricing": {"prompt": "0.000005", "completion": "0.000025", "input_cache_read": "0.0000005"}},
        {"id": "anthropic/claude-opus-5.5:thinking", "pricing": {"prompt": "0.00001", "completion": "0.00005"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "meta/free-model", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "x/broken", "pricing": {"prompt": "abc", "completion": "0.000001"}},
        {"id": "x/absurd", "pricing": {"prompt": "1", "completion": "1"}},
    ]}
    assert pricing.parse_openrouter(data) == {"claude-opus-5.5": {"in": 5.0, "out": 25.0, "cache_read": 0.5}}


def test_merge_later_tables_win():
    assert pricing.merge({"a": {"in": 1}, "b": {"in": 1}}, {"b": {"in": 2}}) == {"a": {"in": 1}, "b": {"in": 2}}


def test_refresh_uses_and_caches_web_prices(monkeypatch, tmp_path):
    async def fake_fetch(client, url):
        if url == pricing.OPENROUTER_URL:
            return openrouter_payload()
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(pricing, "_fetch", fake_fetch)
    status = asyncio.run(pricing.refresh(reprice=False))

    assert status["source"] == "web"
    assert status["sources"] == {"openrouter": 61}
    assert "litellm" in status["error"]
    assert price_key("claude-opus-5-5-medium") == "claude-opus-5.5"
    cached = json.loads((tmp_path / "prices.json").read_text(encoding="utf-8"))
    assert cached["prices"]["claude-opus-5.5"]["in"] == 5.0

    set_web_prices({})
    assert price_key("claude-opus-5-5-medium") is None
    assert pricing.load_cache() is True
    assert price_key("claude-opus-5-5-medium") == "claude-opus-5.5"


def test_refresh_keeps_current_prices_when_sources_fail(monkeypatch):
    async def offline(client, url):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(pricing, "_fetch", offline)
    status = asyncio.run(pricing.refresh(reprice=False))
    assert status["source"] == "builtin"
    assert "offline" in status["error"]
    assert price_key("gpt-4o") == "gpt-4o"


def test_cached_entries_are_validated():
    prices = {
        "good": {"in": 1.0, "out": 2.0, "cache_read": 0.1, "cache_write": "x"},
        "missing-out": {"in": 1.0},
        "text": {"in": "1", "out": "2"},
        "negative": {"in": -1, "out": 2},
        "absurd": {"in": 1.0, "out": pricing.MAX_PRICE + 1},
        "flag": {"in": True, "out": 2},
        "not-a-dict": [1, 2],
    }
    assert pricing.clean_cached(prices) == {"good": {"in": 1.0, "out": 2.0, "cache_read": 0.1}}
    assert pricing.clean_cached(["not", "a", "dict"]) == {}


def test_load_cache_drops_bad_entries_and_rejects_a_mostly_broken_file(tmp_path):
    good = {f"model-{i}": {"in": 1.0, "out": 2.0} for i in range(pricing.MIN_ENTRIES)}
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"prices": {**good, "broken": {"in": 1.0}}}), encoding="utf-8")
    assert pricing.load_cache() is True
    assert price_key("broken") is None

    set_web_prices({})
    broken = {f"model-{i}": {"in": "cheap"} for i in range(pricing.MIN_ENTRIES)}
    path.write_text(json.dumps({"prices": broken}), encoding="utf-8")
    assert pricing.load_cache() is False


def test_too_small_table_is_rejected(monkeypatch):
    async def tiny(client, url):
        return openrouter_payload(n=3) if url == pricing.OPENROUTER_URL else {}

    monkeypatch.setattr(pricing, "_fetch", tiny)
    status = asyncio.run(pricing.refresh(reprice=False))
    assert status["source"] == "builtin"
    assert price_key("claude-opus-5-5-medium") is None
