"""
Trace, span, cost and metric queries shared by the REST routes (`routes/traces.py`,
`routes/metrics.py`) and the MCP server (`mcp_server/server.py`).

Every function returns JSON-native values (ids as strings, datetimes as ISO strings).
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from beanie import PydanticObjectId

from evaluation.runner import JUDGED_METRICS, current_verdict_condition, judge_tag
from models.log import Log
from models.metric import MetricName
from models.span import Span, SpanType
from models.trace import Trace

_OTEL_TRACE_ID = re.compile(r"^[0-9a-fA-F]{32}$")

# Metrics where a higher value or a True/False flag signals the problem.
_HIGHER_IS_WORSE = {MetricName.moderation.value}
_BOOLEAN_METRICS = {
    MetricName.is_json.value,
    MetricName.output_format_compliance.value,
    MetricName.equals.value,
    MetricName.contains.value,
    MetricName.matches_regex.value,
    MetricName.compilable.value,
}

# Metric value as a number for averaging: booleans count 0/1, a list (hallucinated claims)
# counts 1 when non-empty, and labels ("fast") or objects (total_cost_summary) become null,
# which $avg skips.
NUMERIC_METRIC_VALUE = {
    "$cond": [
        {"$isArray": "$metrics.value"},
        {"$cond": [{"$gt": [{"$size": "$metrics.value"}, 0]}, 1, 0]},
        {"$convert": {"input": "$metrics.value", "to": "double", "onError": None, "onNull": None}},
    ]
}

ERROR_SPANS = {
    "$or": [
        {"error": {"$nin": [None, ""]}},
        {"tool_error": {"$nin": [None, ""]}},
        {"status": {"$in": ["failed", "error"]}},
    ]
}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _clip(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "..."


def metric_values(metrics: list) -> dict:
    return {getattr(m.name, "value", m.name): m.value for m in metrics or []}


def metric_dict(m) -> dict:
    """A metric as JSON; `stale` flags an LLM verdict from another judge than the current one."""
    tag = judge_tag()
    data = m.model_dump(mode="json")
    data["stale"] = bool(tag) and data["name"] in JUDGED_METRICS and data.get("judge") != tag
    return data


def current_verdicts_stage() -> list:
    """Pipeline stages, after an $unwind of `metrics`, dropping verdicts of an older judge."""
    condition = current_verdict_condition("metrics.")
    return [{"$match": condition}] if condition else []


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

TOOL_ORIGINS = ("cursor", "claude-code")
ORIGINS = (*TOOL_ORIGINS, "sdk", "otel")


def trace_origin(t: Trace) -> str:
    """The tool that produced a trace (recorded in metadata.source), else its ingestion path."""
    tool = (t.metadata or {}).get("source")
    return tool if tool in TOOL_ORIGINS else t.source


def origin_query(origin: str) -> dict:
    """Trace filter matching `trace_origin(trace) == origin`."""
    if origin in TOOL_ORIGINS:
        return {"metadata.source": origin}
    return {"source": origin, "metadata.source": {"$nin": list(TOOL_ORIGINS)}}


def trace_summary(t: Trace, include_metrics: bool = False) -> dict:
    summary = {
        "id": str(t.id),
        "name": t.name,
        "status": t.status.value,
        "source": t.source,
        "origin": trace_origin(t),
        "otel_trace_id": t.external_id,
        "total_spans": t.total_spans,
        "total_llm_spans": t.total_llm_spans,
        "total_tool_spans": t.total_tool_spans,
        "total_tokens_in": t.total_tokens_in,
        "total_tokens_out": t.total_tokens_out,
        "total_cost_usd": t.total_cost_usd,
        "duration_ms": t.duration_ms,
        "started_at": _iso(t.started_at),
        "ended_at": _iso(t.ended_at),
        "has_metrics": len(t.metrics) > 0,
    }
    if include_metrics:
        summary["metrics"] = metric_values(t.metrics)
        if t.metadata:
            summary["metadata"] = t.metadata
    return summary


def span_detail(
    s: Span,
    message_limit: int = 500,
    response_limit: int = 2000,
    preview_limit: int = 300,
    tool_output_limit: int = 300,
) -> dict:
    base = {
        "id": str(s.id),
        "step_index": s.step_index,
        "span_type": s.span_type.value,
        "name": s.name,
        "status": s.status,
        "error": s.error,
        "source": s.source,
        "sdk_span_id": s.sdk_span_id,
        "parent_span_id": s.parent_span_id,
        "caller_chain": s.caller_chain,
        "latency_ms": s.latency_ms,
        "timestamp": _iso(s.timestamp),
        "metrics": [metric_dict(m) for m in s.metrics],
    }
    if s.span_type in (SpanType.agent, SpanType.chain):
        base.update({
            "agent_name": s.agent_name,
            "role": s.role,
            "metadata": s.metadata,
        })
    elif s.span_type in (SpanType.llm, SpanType.embedding):
        msgs_in_truncated = None
        if s.messages_in:
            msgs_in_truncated = [
                {**m, "content": (m.get("content", "") or "")[:message_limit]}
                for m in s.messages_in if isinstance(m, dict)
            ]
        base.update({
            "provider": s.provider,
            "model": s.model,
            "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out,
            "cache_read_tokens": s.cache_read_tokens,
            "cache_write_tokens": s.cache_write_tokens,
            "cost_usd": s.cost_usd,
            "temperature": s.temperature,
            "response_preview": (s.response or "")[:preview_limit],
            "response": (s.response or "")[:response_limit],
            "messages_in": msgs_in_truncated,
            "iteration": s.iteration,
            "action_success": s.action_success,
            "similarity_score": s.similarity_score,
            "feedback": s.feedback,
        })
    else:
        base.update({
            "tool_name": s.tool_name,
            "tool_input": s.tool_input,
            "tool_output": (s.tool_output or "")[:tool_output_limit],
            "tool_error": s.tool_error,
        })
    return base


def span_depths(spans: list) -> dict:
    """Depth of each span in its trace's tree, keyed by sdk_span_id (0 = top level)."""
    parents = {s.sdk_span_id: s.parent_span_id for s in spans if s.sdk_span_id}
    depths = {}
    for span_id in parents:
        depth, current, seen = 0, parents[span_id], set()
        while current and current in parents and current not in seen:
            seen.add(current)
            depth += 1
            current = parents[current]
        depths[span_id] = depth
    return depths


def span_brief(s: Span) -> dict:
    """One-line view of a span, used in problem listings."""
    return {
        "span_id": str(s.id),
        "trace_id": str(s.trace_id),
        "span_type": s.span_type.value,
        "name": s.name or s.tool_name or s.model,
        "model": s.model,
        "tool_name": s.tool_name,
        "latency_ms": s.latency_ms,
        "status": s.status,
        "error": _clip(s.error or s.tool_error, 500),
        "timestamp": _iso(s.timestamp),
        "metrics": metric_values(s.metrics),
        "response_preview": _clip(s.response, 200),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

async def project_stats(pid: PydanticObjectId) -> dict:
    """Aggregate stats for all traces of a project (dashboard)."""
    traces = await Trace.find(Trace.project_id == pid).to_list()
    if not traces:
        return {
            "total_traces": 0,
            "total_llm_calls": 0,
            "total_cost_usd": 0,
            "total_tokens": 0,
            "avg_duration_ms": 0,
            "metrics_aggregate": {},
            "daily_costs": [],
        }

    total_traces  = len(traces)
    total_calls   = sum(t.total_llm_spans for t in traces)
    total_cost    = sum(t.total_cost_usd for t in traces)
    total_tokens  = sum(t.total_tokens_in + t.total_tokens_out for t in traces)
    durations     = [t.duration_ms for t in traces if t.duration_ms]
    avg_duration  = int(sum(durations) / len(durations)) if durations else 0

    # Aggregate trace-level metrics (boolean and float)
    metrics_agg: dict = {}
    for t in traces:
        for m in t.metrics:
            name = m.name if isinstance(m.name, str) else m.name.value
            val  = m.value
            if name not in metrics_agg:
                metrics_agg[name] = {"true": 0, "false": 0, "values": []}
            if isinstance(val, bool):
                metrics_agg[name]["true" if val else "false"] += 1
            elif isinstance(val, (int, float)):
                metrics_agg[name]["values"].append(float(val))

    # Daily cost for trend chart
    daily: dict = {}
    for t in traces:
        if t.started_at:
            day = t.started_at.strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + t.total_cost_usd
    daily_costs = [{"date": d, "cost": round(c, 6)} for d, c in sorted(daily.items())]

    return {
        "total_traces":    total_traces,
        "total_llm_calls": total_calls,
        "total_cost_usd":  round(total_cost, 6),
        "total_tokens":    total_tokens,
        "avg_duration_ms": avg_duration,
        "metrics_aggregate": metrics_agg,
        "daily_costs":     daily_costs,
    }


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


async def metrics_summary(pid: PydanticObjectId, days: int, since: Optional[datetime] = None) -> dict:
    """Average of each span- and trace-level metric, plus cost totals, over the last N days
    (or since `since` when given)."""
    since = since or datetime.utcnow() - timedelta(days=days)

    numeric_value = NUMERIC_METRIC_VALUE

    span_pipeline = [
        {"$match": {"project_id": pid, "timestamp": {"$gte": since}, "metrics.0": {"$exists": True}}},
        {"$unwind": "$metrics"},
        *current_verdicts_stage(),
        {"$group": {
            "_id": "$metrics.name",
            "avg_value": {"$avg": numeric_value},
            "count": {"$sum": 1},
        }},
    ]

    trace_pipeline = [
        {"$match": {"project_id": pid, "started_at": {"$gte": since}, "metrics.0": {"$exists": True}}},
        {"$unwind": "$metrics"},
        *current_verdicts_stage(),
        {"$group": {
            "_id": "$metrics.name",
            "avg_value": {"$avg": numeric_value},
            "count": {"$sum": 1},
        }},
    ]

    span_results = await Span.aggregate(span_pipeline).to_list()
    trace_results = await Trace.aggregate(trace_pipeline).to_list()

    cost_pipeline = [
        {"$match": {"project_id": pid, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": None,
            "total_cost": {"$sum": "$total_cost_usd"},
            "total_traces": {"$sum": 1},
            "total_tokens": {"$sum": {"$add": ["$total_tokens_in", "$total_tokens_out"]}},
            "avg_duration_ms": {"$avg": "$duration_ms"},
        }},
    ]
    cost_results = await Trace.aggregate(cost_pipeline).to_list()
    cost_data = cost_results[0] if cost_results else {}

    return {
        "period_days": days,
        "span_metrics": {
            r["_id"]: {"avg": _round_or_none(r["avg_value"]), "evaluated_count": r["count"]}
            for r in span_results
        },
        "trace_metrics": {
            r["_id"]: {"avg": _round_or_none(r["avg_value"]), "evaluated_count": r["count"]}
            for r in trace_results
        },
        "cost": {
            "total_usd": round(cost_data.get("total_cost", 0), 4),
            "total_traces": cost_data.get("total_traces", 0),
            "total_tokens": cost_data.get("total_tokens", 0),
            "avg_duration_ms": round(cost_data.get("avg_duration_ms") or 0),
        },
    }


async def period_totals(pid: PydanticObjectId, since: datetime) -> dict:
    pipeline = [
        {"$match": {"project_id": pid, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": None,
            "traces": {"$sum": 1},
            "failed_traces": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
            "running_traces": {"$sum": {"$cond": [{"$eq": ["$status", "running"]}, 1, 0]}},
            "llm_calls": {"$sum": "$total_llm_spans"},
            "tool_calls": {"$sum": "$total_tool_spans"},
            "tokens_in": {"$sum": "$total_tokens_in"},
            "tokens_out": {"$sum": "$total_tokens_out"},
            "cost_usd": {"$sum": "$total_cost_usd"},
            "avg_duration_ms": {"$avg": "$duration_ms"},
        }},
    ]
    rows = await Trace.aggregate(pipeline).to_list()
    row = rows[0] if rows else {}
    error_spans = await Span.find({"project_id": pid, "timestamp": {"$gte": since}, **ERROR_SPANS}).count()
    return {
        "traces": row.get("traces", 0),
        "failed_traces": row.get("failed_traces", 0),
        "running_traces": row.get("running_traces", 0),
        "llm_calls": row.get("llm_calls", 0),
        "tool_calls": row.get("tool_calls", 0),
        "tokens_in": row.get("tokens_in", 0),
        "tokens_out": row.get("tokens_out", 0),
        "cost_usd": round(row.get("cost_usd") or 0.0, 6),
        "avg_duration_ms": round(row.get("avg_duration_ms") or 0),
        "error_spans": error_spans,
    }


async def recent_traces(
    pid: PydanticObjectId,
    since: datetime,
    status: Optional[str] = None,
    limit: int = 20,
    origin: Optional[str] = None,
) -> dict:
    query = {"project_id": pid, "started_at": {"$gte": since}}
    if status:
        query["status"] = status
    if origin:
        query.update(origin_query(origin))
    total = await Trace.find(query).count()
    traces = await Trace.find(query).sort("-started_at").limit(limit).to_list()
    return {"total": total, "traces": [trace_summary(t, include_metrics=True) for t in traces]}


async def find_project_trace(pid: PydanticObjectId, trace_id: str) -> Optional[Trace]:
    """Trace of this project by AItrajectIQ id or by OpenTelemetry trace id (32 hex chars)."""
    trace_id = (trace_id or "").strip()
    if _OTEL_TRACE_ID.match(trace_id):
        return await Trace.find_one({"project_id": pid, "external_id": trace_id.lower()})
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        return None
    trace = await Trace.get(tid)
    return trace if trace is not None and trace.project_id == pid else None


async def trace_detail(trace: Trace, include_content: bool = False) -> dict:
    spans = await Span.find(Span.trace_id == trace.id).sort("+step_index").to_list()
    if include_content:
        limits = dict(message_limit=4000, response_limit=8000, preview_limit=300, tool_output_limit=4000)
    else:
        limits = dict(message_limit=300, response_limit=600, preview_limit=200, tool_output_limit=300)
    depths = span_depths(spans)
    return {
        "trace": trace_summary(trace, include_metrics=True),
        "metrics": [metric_dict(m) for m in trace.metrics],
        "spans": [{**span_detail(s, **limits), "depth": depths.get(s.sdk_span_id, 0)} for s in spans],
    }


async def cost_breakdown(pid: PydanticObjectId, since: datetime, group_by: str = "model") -> dict:
    keys = {
        "model": "$model",
        "provider": "$provider",
        "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
    }
    if group_by not in keys:
        raise ValueError(f"group_by must be one of {sorted(keys)}")
    pipeline = [
        {"$match": {"project_id": pid, "span_type": SpanType.llm.value, "timestamp": {"$gte": since}}},
        {"$group": {
            "_id": keys[group_by],
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "calls": {"$sum": 1},
            "tokens_in": {"$sum": {"$ifNull": ["$tokens_in", 0]}},
            "tokens_out": {"$sum": {"$ifNull": ["$tokens_out", 0]}},
        }},
        {"$sort": {"_id": 1} if group_by == "day" else {"cost_usd": -1}},
    ]
    rows = await Span.aggregate(pipeline).to_list()
    items = [
        {
            group_by: row["_id"] or "unknown",
            "cost_usd": round(row["cost_usd"], 6),
            "calls": row["calls"],
            "tokens_in": row["tokens_in"],
            "tokens_out": row["tokens_out"],
        }
        for row in rows
    ]
    return {
        "group_by": group_by,
        "total_cost_usd": round(sum(i["cost_usd"] for i in items), 6),
        "items": items,
    }


def _current_judge(metric: str) -> dict:
    """$elemMatch clause restricting an LLM-judged metric to the current judge's verdicts."""
    tag = judge_tag()
    return {"judge": tag} if tag and metric in JUDGED_METRICS else {}


async def evaluated_traces(pid: PydanticObjectId, metric: str, since: datetime, limit: int = 50) -> dict:
    """Latest traces carrying a trace-level metric, with its verdict and the judge's reason."""
    if metric not in {m.value for m in MetricName}:
        raise ValueError(f"Unknown metric {metric!r}")
    query = {
        "project_id": pid,
        "started_at": {"$gte": since},
        "metrics": {"$elemMatch": {"name": metric, **_current_judge(metric)}},
    }
    total = await Trace.find(query).count()
    traces = await Trace.find(query).sort("-started_at").limit(limit).to_list()
    rows = []
    for t in traces:
        m = next(m for m in t.metrics if getattr(m.name, "value", m.name) == metric)
        rows.append({
            "trace_id": str(t.id),
            "name": t.name,
            "status": getattr(t.status, "value", t.status),
            "origin": trace_origin(t),
            "started_at": _iso(t.started_at),
            "value": m.value,
            "reason": m.reason,
        })
    return {"metric": metric, "count": total, "traces": rows}


def _metric_condition(metric: str, threshold: Optional[float]) -> dict:
    if metric == MetricName.hallucination.value:
        return {"name": metric, "value.0": {"$exists": True}}
    if metric == MetricName.latency_classification.value:
        return {"name": metric, "value": {"$in": ["slow", "very_slow"]}}
    if metric in _BOOLEAN_METRICS:
        return {"name": metric, "value": False}
    if metric in _HIGHER_IS_WORSE:
        return {"name": metric, "value": {"$gt": 0.5 if threshold is None else threshold}}
    return {"name": metric, "value": {"$lt": 0.5 if threshold is None else threshold}}


async def problem_spans(
    pid: PydanticObjectId,
    kind: str,
    since: datetime,
    metric: Optional[str] = None,
    threshold: Optional[float] = None,
    limit: int = 20,
) -> dict:
    """Spans with errors, slow spans, or spans with a poor evaluation score."""
    query: dict = {"project_id": pid, "timestamp": {"$gte": since}}
    sort = "-timestamp"
    criteria: dict = {"kind": kind}
    if kind == "errors":
        query.update(ERROR_SPANS)
    elif kind == "slow":
        min_ms = int(threshold) if threshold is not None else 10_000
        query["latency_ms"] = {"$gte": min_ms}
        query["span_type"] = {"$in": ["llm", "tool", "retriever", "embedding"]}
        sort = "-latency_ms"
        criteria["min_latency_ms"] = min_ms
    elif kind == "low_score":
        metric = metric or MetricName.general_faithfulness.value
        if metric not in {m.value for m in MetricName}:
            raise ValueError(f"Unknown metric {metric!r}")
        query["metrics"] = {"$elemMatch": {**_metric_condition(metric, threshold), **_current_judge(metric)}}
        criteria.update(metric=metric, threshold=threshold)
    else:
        raise ValueError("kind must be 'errors', 'slow' or 'low_score'")
    spans = await Span.find(query).sort(sort).limit(limit).to_list()
    return {"criteria": criteria, "count": len(spans), "spans": [span_brief(s) for s in spans]}


async def recent_logs(pid: PydanticObjectId, limit: int = 20, blocked_only: bool = False) -> dict:
    query: dict = {"project_id": pid}
    if blocked_only:
        query["blocked"] = True
    logs = await Log.find(query).sort("-timestamp").limit(limit).to_list()
    return {
        "count": len(logs),
        "logs": [
            {
                "log_id": str(log.id),
                "timestamp": _iso(log.timestamp),
                "model": log.Model,
                "prompt": _clip(log.prompt, 300),
                "output": _clip(log.Output, 300),
                "latency_ms": log.latency,
                "tokens": log.n_tokens,
                "blocked": log.blocked,
                "blocked_reason": log.blocked_reason,
                "contains_pii": log.contains_pii,
                "metrics": metric_values(log.metrics or []),
                "final_score": log.final_score,
            }
            for log in logs
        ],
    }
