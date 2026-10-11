"""
Queries behind the dashboard API (routes/dashboard_routes.py): period KPIs, time
series, cost splits, metric trends and the trace tree with waterfall offsets.

Time buckets are computed with date arithmetic ($toLong / $mod), which works on every
MongoDB version the backend supports. Empty buckets are filled so charts stay continuous.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from beanie import PydanticObjectId

from models.span import Span, SpanType
from models.trace import Trace
from services import insights


@dataclass(frozen=True)
class Range:
    key: str
    span: timedelta
    bucket: timedelta

    @property
    def bucket_ms(self) -> int:
        return int(self.bucket.total_seconds() * 1000)

    @property
    def days(self) -> float:
        return self.span.total_seconds() / 86400


RANGES = {
    "1h": Range("1h", timedelta(hours=1), timedelta(minutes=1)),
    "24h": Range("24h", timedelta(hours=24), timedelta(hours=1)),
    "7d": Range("7d", timedelta(days=7), timedelta(hours=6)),
    "30d": Range("30d", timedelta(days=30), timedelta(days=1)),
}

MAX_COST_SERIES = 6
OTHER = "other"


def get_range(key: str) -> Range:
    try:
        return RANGES[key]
    except KeyError:
        raise ValueError(f"range must be one of {list(RANGES)}") from None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def epoch_ms(dt: datetime) -> int:
    """Milliseconds since the epoch; naive datetimes are UTC (as stored by the backend)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_epoch_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="milliseconds") + "Z"


def bucket_expr(field: str, bucket_ms: int) -> dict:
    """Aggregation expression flooring a date field to the start of its bucket."""
    as_long = {"$toLong": f"${field}"}
    return {"$toDate": {"$subtract": [as_long, {"$mod": [as_long, bucket_ms]}]}}


def bucket_starts(since: datetime, until: datetime, bucket_ms: int) -> list:
    """Start of every bucket overlapping [since, until], aligned like bucket_expr."""
    first = epoch_ms(since) // bucket_ms * bucket_ms
    last = epoch_ms(until) // bucket_ms * bucket_ms
    return [from_epoch_ms(ms) for ms in range(first, last + 1, bucket_ms)]


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (q in 0..100); None for no values."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    rank = max(1, math.ceil(q / 100 * len(data)))
    return data[min(rank, len(data)) - 1]


def change_ratio(current: float, previous: float) -> Optional[float]:
    """Relative change from the previous period; None when there is nothing to compare."""
    if not previous:
        return None
    return round((current - previous) / previous, 4)


def span_start(span: dict) -> Optional[datetime]:
    """When a span started. OTel spans store their start; SDK spans are stored when they end."""
    ts = span.get("timestamp")
    if ts is None:
        return None
    if span.get("source") == "otel":
        return ts
    return ts - timedelta(milliseconds=span.get("latency_ms") or 0)


def attribute_agents(llm_spans: list, agent_spans: list) -> dict:
    """Agent name of each LLM span: nearest agent ancestor, else the span's own agent name,
    else the innermost application function recorded by the SDK. Keyed by the span `_id`."""
    agent_names = {
        (a["trace_id"], a["sdk_span_id"]): a.get("agent_name") or a.get("name")
        for a in agent_spans
        if a.get("sdk_span_id")
    }
    parents = {
        (s["trace_id"], s["sdk_span_id"]): s.get("parent_span_id")
        for s in list(llm_spans) + list(agent_spans)
        if s.get("sdk_span_id")
    }
    result = {}
    for span in llm_spans:
        name, seen = None, set()
        current = span.get("parent_span_id")
        while current and (span["trace_id"], current) not in seen:
            key = (span["trace_id"], current)
            seen.add(key)
            if key in agent_names:
                name = agent_names[key]
                break
            current = parents.get(key)
        if not name:
            chain = span.get("caller_chain") or []
            name = span.get("agent_name") or (chain[-1] if chain else None)
        result[span["_id"]] = name or "unattributed"
    return result


def top_series(totals: dict, limit: int = MAX_COST_SERIES) -> list:
    """The `limit` largest keys by total (callers fold the rest into OTHER)."""
    ranked = sorted(totals, key=lambda k: totals[k], reverse=True)
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

async def totals(pid: PydanticObjectId, since: datetime, until: datetime) -> dict:
    match = {"project_id": pid, "started_at": {"$gte": since, "$lt": until}}
    rows = await Trace.aggregate([
        {"$match": match},
        {"$group": {
            "_id": None,
            "traces": {"$sum": 1},
            "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
            "running": {"$sum": {"$cond": [{"$eq": ["$status", "running"]}, 1, 0]}},
            "llm_calls": {"$sum": "$total_llm_spans"},
            "tool_calls": {"$sum": "$total_tool_spans"},
            "tokens_in": {"$sum": "$total_tokens_in"},
            "tokens_out": {"$sum": "$total_tokens_out"},
            "cost_usd": {"$sum": "$total_cost_usd"},
            "durations": {"$push": "$duration_ms"},
        }},
    ]).to_list()
    row = rows[0] if rows else {}
    durations = [d for d in row.get("durations", []) if d is not None]
    error_spans = await Span.find({
        "project_id": pid, "timestamp": {"$gte": since, "$lt": until}, **insights.ERROR_SPANS,
    }).count()
    traces = row.get("traces", 0)
    failed = row.get("failed", 0)
    return {
        "traces": traces,
        "failed_traces": failed,
        "running_traces": row.get("running", 0),
        "error_rate": round(failed / traces, 4) if traces else 0.0,
        "error_spans": error_spans,
        "llm_calls": row.get("llm_calls", 0),
        "tool_calls": row.get("tool_calls", 0),
        "tokens_in": row.get("tokens_in", 0),
        "tokens_out": row.get("tokens_out", 0),
        "cost_usd": round(row.get("cost_usd") or 0.0, 6),
        "latency_p50_ms": percentile(durations, 50),
        "latency_p95_ms": percentile(durations, 95),
        "avg_duration_ms": round(sum(durations) / len(durations)) if durations else None,
    }


async def overview(pid: PydanticObjectId, rng: Range, now: Optional[datetime] = None) -> dict:
    now = now or datetime.utcnow()
    current = await totals(pid, now - rng.span, now)
    previous = await totals(pid, now - 2 * rng.span, now - rng.span)
    compared = ("traces", "cost_usd", "llm_calls", "error_rate", "latency_p95_ms")
    changes = {
        key: change_ratio(current[key] or 0, previous[key] or 0) for key in compared
    }
    changes["tokens"] = change_ratio(
        current["tokens_in"] + current["tokens_out"], previous["tokens_in"] + previous["tokens_out"],
    )
    return {"range": rng.key, "current": current, "previous": previous, "change": changes}


async def timeseries(pid: PydanticObjectId, rng: Range, now: Optional[datetime] = None) -> dict:
    now = now or datetime.utcnow()
    since = now - rng.span
    trace_rows = await Trace.aggregate([
        {"$match": {"project_id": pid, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": bucket_expr("started_at", rng.bucket_ms),
            "traces": {"$sum": 1},
            "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
            "cost_usd": {"$sum": "$total_cost_usd"},
            "tokens_in": {"$sum": "$total_tokens_in"},
            "tokens_out": {"$sum": "$total_tokens_out"},
            "llm_calls": {"$sum": "$total_llm_spans"},
            "durations": {"$push": "$duration_ms"},
        }},
    ]).to_list()
    error_rows = await Span.aggregate([
        {"$match": {"project_id": pid, "timestamp": {"$gte": since}, **insights.ERROR_SPANS}},
        {"$group": {"_id": bucket_expr("timestamp", rng.bucket_ms), "errors": {"$sum": 1}}},
    ]).to_list()

    by_bucket = {epoch_ms(r["_id"]): r for r in trace_rows}
    errors = {epoch_ms(r["_id"]): r["errors"] for r in error_rows}
    points = []
    for start in bucket_starts(since, now, rng.bucket_ms):
        key = epoch_ms(start)
        row = by_bucket.get(key, {})
        durations = [d for d in row.get("durations", []) if d is not None]
        points.append({
            "t": iso_utc(start),
            "traces": row.get("traces", 0),
            "failed": row.get("failed", 0),
            "errors": errors.get(key, 0),
            "cost_usd": round(row.get("cost_usd") or 0.0, 6),
            "tokens_in": row.get("tokens_in", 0),
            "tokens_out": row.get("tokens_out", 0),
            "llm_calls": row.get("llm_calls", 0),
            "latency_p50_ms": percentile(durations, 50),
            "latency_p95_ms": percentile(durations, 95),
        })
    return {"range": rng.key, "bucket_ms": rng.bucket_ms, "points": points}


async def list_traces(
    pid: PydanticObjectId,
    rng: Optional[Range] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    query: dict = {"project_id": pid}
    if rng is not None:
        query["started_at"] = {"$gte": datetime.utcnow() - rng.span}
    if status:
        query["status"] = status
    if source:
        query.update(insights.origin_query(source))
    if q:
        q = q.strip()
        ors: list = [{"name": {"$regex": re.escape(q), "$options": "i"}}, {"external_id": q.lower()}]
        try:
            ors.append({"_id": PydanticObjectId(q)})
        except Exception:
            pass
        query["$or"] = ors
    total = await Trace.find(query).count()
    docs = (
        await Trace.find(query).sort("-started_at")
        .skip((page - 1) * page_size).limit(page_size).to_list()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "traces": [_trace_row(t) for t in docs],
    }


def _trace_row(t: Trace) -> dict:
    row = insights.trace_summary(t, include_metrics=True)
    row["started_at"] = iso_utc(t.started_at)
    row["ended_at"] = iso_utc(t.ended_at)
    return row


async def trace_tree(trace: Trace) -> dict:
    """Trace detail with full content, plus each span's offset/duration for the waterfall."""
    detail = await insights.trace_detail(trace, include_content=True)
    raw = {
        str(s["_id"]): s
        for s in await Span.get_motor_collection().find(
            {"trace_id": trace.id},
            {"timestamp": 1, "latency_ms": 1, "source": 1, "agent_name": 1, "metadata": 1},
        ).to_list(None)
    }
    starts = {sid: span_start(s) for sid, s in raw.items()}
    known = [s for s in starts.values() if s is not None]
    origin = min(known + [trace.started_at]) if known else trace.started_at
    for span in detail["spans"]:
        start = starts.get(span["id"])
        doc = raw.get(span["id"], {})
        span["start_offset_ms"] = max(0, epoch_ms(start) - epoch_ms(origin)) if start else 0
        span["duration_ms"] = span.get("latency_ms") or 0
        span["timestamp"] = iso_utc(doc.get("timestamp"))
        span.setdefault("agent_name", doc.get("agent_name"))
        span.setdefault("metadata", doc.get("metadata"))
    trace_row = _trace_row(trace)
    trace_row["metadata"] = trace.metadata
    ends = [s["start_offset_ms"] + s["duration_ms"] for s in detail["spans"]]
    return {
        "trace": trace_row,
        "metrics": detail["metrics"],
        "spans": detail["spans"],
        "origin": iso_utc(origin),
        "total_ms": max(ends + [trace.duration_ms or 0]) if ends else (trace.duration_ms or 0),
    }


async def _llm_span_docs(pid: PydanticObjectId, since: datetime) -> list:
    return await Span.get_motor_collection().find(
        {"project_id": pid, "span_type": SpanType.llm.value, "timestamp": {"$gte": since}},
        {
            "trace_id": 1, "sdk_span_id": 1, "parent_span_id": 1, "agent_name": 1,
            "caller_chain": 1, "model": 1, "cost_usd": 1, "tokens_in": 1, "tokens_out": 1,
            "timestamp": 1,
        },
    ).to_list(None)


async def cost_by_agent(pid: PydanticObjectId, since: datetime, llm_spans: Optional[list] = None) -> list:
    llm_spans = llm_spans if llm_spans is not None else await _llm_span_docs(pid, since)
    trace_ids = list({s["trace_id"] for s in llm_spans})
    agent_spans = await Span.get_motor_collection().find(
        {"trace_id": {"$in": trace_ids}, "span_type": {"$in": [SpanType.agent.value, SpanType.chain.value]}},
        {"trace_id": 1, "sdk_span_id": 1, "parent_span_id": 1, "agent_name": 1, "name": 1},
    ).to_list(None) if trace_ids else []
    names = attribute_agents(llm_spans, agent_spans)
    groups: dict = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0})
    for span in llm_spans:
        g = groups[names[span["_id"]]]
        g["cost_usd"] += span.get("cost_usd") or 0.0
        g["calls"] += 1
        g["tokens_in"] += span.get("tokens_in") or 0
        g["tokens_out"] += span.get("tokens_out") or 0
    items = [{"agent": name, **values, "cost_usd": round(values["cost_usd"], 6)} for name, values in groups.items()]
    return sorted(items, key=lambda i: i["cost_usd"], reverse=True)


async def costs(pid: PydanticObjectId, rng: Range, now: Optional[datetime] = None) -> dict:
    now = now or datetime.utcnow()
    since = now - rng.span
    by_model = await insights.cost_breakdown(pid, since, group_by="model")
    llm_spans = await _llm_span_docs(pid, since)
    top = await Trace.find(
        {"project_id": pid, "started_at": {"$gte": since}, "total_cost_usd": {"$gt": 0}}
    ).sort("-total_cost_usd").limit(10).to_list()
    current = await totals(pid, since, now)
    previous = await totals(pid, since - rng.span, since)
    return {
        "range": rng.key,
        "total_cost_usd": current["cost_usd"],
        "previous_cost_usd": previous["cost_usd"],
        "change": change_ratio(current["cost_usd"], previous["cost_usd"]),
        "avg_cost_per_trace": round(current["cost_usd"] / current["traces"], 6) if current["traces"] else 0.0,
        "by_model": by_model["items"],
        "by_agent": await cost_by_agent(pid, since, llm_spans),
        "top_traces": [_trace_row(t) for t in top],
    }


async def cost_timeseries(pid: PydanticObjectId, rng: Range, now: Optional[datetime] = None) -> dict:
    """Cost per bucket split by model (largest models first, the rest folded into 'other')."""
    now = now or datetime.utcnow()
    since = now - rng.span
    rows = await Span.aggregate([
        {"$match": {"project_id": pid, "span_type": SpanType.llm.value, "timestamp": {"$gte": since}}},
        {"$group": {
            "_id": {"t": bucket_expr("timestamp", rng.bucket_ms), "model": {"$ifNull": ["$model", "unknown"]}},
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
        }},
    ]).to_list()
    model_totals: dict = defaultdict(float)
    for r in rows:
        model_totals[r["_id"]["model"]] += r["cost_usd"]
    series = top_series(model_totals)
    has_other = len(model_totals) > len(series)
    keys = series + ([OTHER] if has_other else [])
    by_bucket: dict = defaultdict(lambda: defaultdict(float))
    for r in rows:
        model = r["_id"]["model"]
        by_bucket[epoch_ms(r["_id"]["t"])][model if model in series else OTHER] += r["cost_usd"]
    points = []
    for start in bucket_starts(since, now, rng.bucket_ms):
        values = by_bucket.get(epoch_ms(start), {})
        points.append({"t": iso_utc(start), **{k: round(values.get(k, 0.0), 6) for k in keys}})
    return {"range": rng.key, "series": keys, "points": points}


async def metric_timeseries(pid: PydanticObjectId, name: str, rng: Range, now: Optional[datetime] = None) -> dict:
    """Average of one metric per bucket, from span and trace evaluations (see NUMERIC_METRIC_VALUE)."""
    now = now or datetime.utcnow()
    since = now - rng.span
    numeric = insights.NUMERIC_METRIC_VALUE

    def pipeline(time_field: str) -> list:
        return [
            {"$match": {"project_id": pid, time_field: {"$gte": since}, "metrics.name": name}},
            {"$unwind": "$metrics"},
            {"$match": {"metrics.name": name}},
            *insights.current_verdicts_stage(),
            {"$group": {
                "_id": bucket_expr(time_field, rng.bucket_ms),
                "sum": {"$sum": {"$ifNull": [numeric, 0]}},
                "n": {"$sum": {"$cond": [{"$eq": [numeric, None]}, 0, 1]}},
                "count": {"$sum": 1},
            }},
        ]

    merged: dict = defaultdict(lambda: {"sum": 0.0, "n": 0, "count": 0})
    for rows in (await Span.aggregate(pipeline("timestamp")).to_list(),
                 await Trace.aggregate(pipeline("started_at")).to_list()):
        for r in rows:
            m = merged[epoch_ms(r["_id"])]
            m["sum"] += r["sum"]
            m["n"] += r["n"]
            m["count"] += r["count"]
    points = []
    for start in bucket_starts(since, now, rng.bucket_ms):
        m = merged.get(epoch_ms(start))
        avg = round(m["sum"] / m["n"], 4) if m and m["n"] else None
        points.append({"t": iso_utc(start), "avg": avg, "count": m["count"] if m else 0})
    return {"range": rng.key, "metric": name, "points": points}
