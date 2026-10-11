"""
Model prices fetched from the web, so costs follow price changes without a release.

Sources (public, no API key):
    LiteLLM     model_prices_and_context_window.json: list prices of the providers' own APIs
    OpenRouter  /api/v1/models: models LiteLLM doesn't list yet
LiteLLM wins when both list a model. The built-in table of utils/cost.py stays the fallback
for models neither lists.

Prices are cached in AITRAJECTIQ_PRICES_CACHE (default app/.cache/model_prices.json), loaded at
startup and refreshed every AITRAJECTIQ_PRICE_SYNC_HOURS hours (default 24, 0 = never fetch).
After a refresh, LLM spans billed at the default price because their model was unknown
(metadata.price = "default") are re-priced once their model has a price. Costs computed
with a known price are kept: they are the price at the time of the call.

    cd app
    python -m scripts.sync_prices
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from utils.cost import canonical_model_name, compute_cost, has_price, set_web_prices, web_price_count

log = logging.getLogger("aitrajectiq.pricing")

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

# LiteLLM providers billed at the provider's own list price (not a cloud reseller).
LITELLM_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "mistral", "deepseek", "cohere_chat", "xai"})
LITELLM_MODES = frozenset({"chat", "responses"})

MIN_ENTRIES = 50            # a smaller result is treated as a broken source, not as the new table
MAX_PRICE = 1_000.0         # USD per 1M tokens; above that the entry is an error
RETRY_AFTER = timedelta(hours=1)

_SUFFIX = re.compile(r"(-\d{8}|-\d{4}-\d{2}-\d{2}|-\d{4}|-latest|-preview)$")

_status: dict = {"source": "builtin", "updated_at": None, "models": 0, "sources": {}, "error": None}


def _cache_path() -> Path:
    default = Path(__file__).resolve().parents[1] / ".cache" / "model_prices.json"
    return Path(os.getenv("AITRAJECTIQ_PRICES_CACHE") or default)


def _interval() -> Optional[timedelta]:
    try:
        hours = float(os.getenv("AITRAJECTIQ_PRICE_SYNC_HOURS", "24"))
    except ValueError:
        hours = 24.0
    return timedelta(hours=hours) if hours > 0 else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _per_million(value) -> Optional[float]:
    try:
        price = float(value) * 1_000_000
    except (TypeError, ValueError):
        return None
    return round(price, 6) if 0 <= price <= MAX_PRICE else None


def _entry(inp, out, cache_read=None, cache_write=None) -> Optional[dict]:
    price_in, price_out = _per_million(inp), _per_million(out)
    if price_in is None or price_out is None or (price_in == 0 and price_out == 0):
        return None
    entry = {"in": price_in, "out": price_out}
    for name, value in (("cache_read", cache_read), ("cache_write", cache_write)):
        price = _per_million(value) if value is not None else None
        if price is not None:
            entry[name] = price
    return entry


def base_name(model_id: str) -> tuple[str, bool]:
    """Canonical name without provider prefix and date / "latest" suffix, and whether a
    suffix was removed ("anthropic/claude-opus-4-5-20251101" -> "claude-opus-4.5", True)."""
    name = model_id.lower().strip().rsplit("/", 1)[-1]
    stripped = False
    while (match := _SUFFIX.search(name)) is not None:
        name = name[:match.start()]
        stripped = True
    return canonical_model_name(name), stripped


def _put(prices: dict, exact: dict, model_id: str, entry: Optional[dict]) -> None:
    """Keep one entry per name, preferring the undated id over a dated snapshot."""
    if entry is None:
        return
    name, dated = base_name(model_id)
    if not name or " " in name:
        return
    if name not in prices or (exact.get(name) is False and not dated):
        prices[name] = entry
        exact[name] = not dated


def parse_litellm(data: dict) -> dict[str, dict]:
    prices: dict = {}
    exact: dict = {}
    for model_id, spec in (data or {}).items():
        if not isinstance(spec, dict) or model_id.startswith("ft:"):
            continue
        if spec.get("litellm_provider") not in LITELLM_PROVIDERS or spec.get("mode", "chat") not in LITELLM_MODES:
            continue
        _put(prices, exact, model_id, _entry(
            spec.get("input_cost_per_token"),
            spec.get("output_cost_per_token"),
            spec.get("cache_read_input_token_cost"),
            spec.get("cache_creation_input_token_cost"),
        ))
    return prices


def parse_openrouter(data: dict) -> dict[str, dict]:
    prices: dict = {}
    exact: dict = {}
    for model in (data or {}).get("data") or []:
        model_id = model.get("id") or ""
        # ":free", ":thinking", ... are variants of a model listed on its own.
        if ":" in model_id or model_id.startswith("openrouter/"):
            continue
        pricing = model.get("pricing") or {}
        _put(prices, exact, model_id, _entry(
            pricing.get("prompt"),
            pricing.get("completion"),
            pricing.get("input_cache_read"),
            pricing.get("input_cache_write"),
        ))
    return prices


def merge(*tables: dict[str, dict]) -> dict[str, dict]:
    """Later tables win."""
    merged: dict = {}
    for table in tables:
        merged.update(table)
    return merged


# ---------------------------------------------------------------------------
# Cache and refresh
# ---------------------------------------------------------------------------

def _apply(prices: dict, updated_at: Optional[str], sources: dict) -> None:
    set_web_prices(prices)
    _status.update(source="web", updated_at=updated_at, models=web_price_count(), sources=sources, error=None)


def _valid_price(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= MAX_PRICE


def clean_cached(prices) -> dict[str, dict]:
    """Entries of a cached table that compute_cost can use: numeric "in" and "out" (USD per
    1M tokens), cache prices kept only when valid. A hand-edited file must not crash costs."""
    if not isinstance(prices, dict):
        return {}
    clean = {}
    for name, entry in prices.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        if not (_valid_price(entry.get("in")) and _valid_price(entry.get("out"))):
            continue
        clean[name] = {key: float(value) for key, value in entry.items()
                       if key in ("in", "out", "cache_read", "cache_write") and _valid_price(value)}
    return clean


def load_cache() -> bool:
    """Use the prices of the last refresh, if any. Returns whether a cache was loaded."""
    path = _cache_path()
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        prices = clean_cached(cached["prices"])
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("Ignoring unreadable price cache %s", path)
        return False
    if len(prices) < MIN_ENTRIES:
        log.warning("Ignoring price cache %s: only %d valid entries", path, len(prices))
        return False
    _apply(prices, cached.get("updated_at"), cached.get("sources") or {})
    return True


def _cache_age() -> Optional[timedelta]:
    updated_at = _status.get("updated_at")
    if not updated_at:
        return None
    try:
        return datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)
    except ValueError:
        return None


async def _fetch(client: httpx.AsyncClient, url: str) -> dict:
    response = await client.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


async def refresh(reprice: bool = True) -> dict:
    """Fetch both sources, then use and cache their prices. The current prices are kept when
    no source answers with a plausible table."""
    tables, sources, errors = {}, {}, []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for name, url, parse in (("openrouter", OPENROUTER_URL, parse_openrouter),
                                 ("litellm", LITELLM_URL, parse_litellm)):
            try:
                tables[name] = parse(await _fetch(client, url))
                sources[name] = len(tables[name])
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                errors.append(f"{name}: {exc}")
                log.warning("Price source %s failed: %s", name, exc)

    prices = merge(tables.get("openrouter", {}), tables.get("litellm", {}))
    if len(prices) < MIN_ENTRIES:
        _status["error"] = "; ".join(errors) or f"only {len(prices)} prices received"
        return status()

    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _apply(prices, updated_at, sources)
    _status["error"] = "; ".join(errors) or None
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"updated_at": updated_at, "sources": sources, "prices": prices},
                                   indent=0, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write the price cache %s: %s", path, exc)
    log.info("Model prices updated: %d models (%s)", len(prices), sources)

    if reprice:
        repriced = await reprice_default_spans()
        if repriced:
            log.info("Re-priced %d span(s) billed at the default price", repriced)
    return status()


def status() -> dict:
    """Where the prices come from: "web" (with the last refresh time) or "builtin"."""
    return dict(_status)


async def run_price_sync() -> None:
    """Background task: refresh when the cache is missing or older than the interval."""
    interval = _interval()
    if interval is None:
        return
    while True:
        age = _cache_age()
        if age is None or age >= interval:
            try:
                await refresh()
            except Exception:
                log.exception("Price refresh failed")
            age = _cache_age()
        wait = interval - age if age is not None and age < interval else RETRY_AFTER
        await asyncio.sleep(max(wait.total_seconds(), 60))


# ---------------------------------------------------------------------------
# Re-pricing
# ---------------------------------------------------------------------------

async def refresh_trace_cost(trace_id) -> None:
    """Recompute a trace's total cost from its LLM spans, and its cost-based metrics."""
    from evaluation.runner import TRACE_EVALUATORS
    from models.metric import MetricName
    from models.span import Span, SpanType
    from models.trace import Trace

    trace = await Trace.get(trace_id)
    if trace is None:
        return
    spans = await Span.find(Span.trace_id == trace_id).sort("+step_index").to_list()
    trace.total_cost_usd = round(sum(s.cost_usd or 0.0 for s in spans if s.span_type == SpanType.llm), 6)
    metrics = list(trace.metrics)
    for index, metric in enumerate(metrics):
        if metric.name in (MetricName.cost_efficiency, MetricName.total_cost_summary):
            metrics[index] = await TRACE_EVALUATORS[metric.name](trace, spans)
    await Trace.get_motor_collection().update_one(
        {"_id": trace.id},
        {"$set": {"total_cost_usd": trace.total_cost_usd, "metrics": [m.model_dump(mode="json") for m in metrics]}},
    )


async def reprice_default_spans() -> int:
    """Re-price the LLM spans billed at the default price whose model now has a price."""
    from pymongo import UpdateOne

    from models.span import Span, SpanType

    collection = Span.get_motor_collection()
    query = {"span_type": SpanType.llm.value, "metadata.price": "default"}
    models = [m for m in await collection.distinct("model", query) if m and has_price(m)]
    if not models:
        return 0

    updates, traces = [], set()
    projection = {"model": 1, "tokens_in": 1, "tokens_out": 1, "cache_read_tokens": 1,
                  "cache_write_tokens": 1, "trace_id": 1}
    async for doc in collection.find({**query, "model": {"$in": models}}, projection):
        cost = compute_cost(doc["model"], doc.get("tokens_in") or 0, doc.get("tokens_out") or 0,
                            doc.get("cache_read_tokens") or 0, doc.get("cache_write_tokens") or 0)
        updates.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"cost_usd": cost}, "$unset": {"metadata.price": ""}}))
        traces.add(doc["trace_id"])
    if updates:
        await collection.bulk_write(updates, ordered=False)
    for trace_id in traces:
        await refresh_trace_cost(trace_id)
    return len(updates)
