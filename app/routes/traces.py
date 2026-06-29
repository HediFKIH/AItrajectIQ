"""
Traces & spans read routes.
"""

from fastapi import APIRouter, HTTPException, Query
from beanie import PydanticObjectId
from typing import Optional
from datetime import datetime

from models.trace import Trace, TraceStatus
from models.span import Span

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/stats")
async def get_project_stats(project_id: str = Query(...)):
    """Aggregate stats for ALL traces of a project — used by the dashboard."""
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")

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

    # Daily cost for trend chart (last 30 days)
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


@router.get("/")
async def list_traces(
    project_id: str = Query(...),
    status: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 50,
):
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")

    query = Trace.find(Trace.project_id == pid)

    if status:
        try:
            query = query.find(Trace.status == TraceStatus(status))
        except ValueError:
            pass

    if from_date:
        query = query.find(Trace.started_at >= from_date)
    if to_date:
        query = query.find(Trace.started_at <= to_date)

    traces = await query.sort("-started_at").skip(skip).limit(limit).to_list()
    total = await Trace.find(Trace.project_id == pid).count()

    return {
        "total": total,
        "traces": [_trace_summary(t) for t in traces],
    }


@router.get("/{trace_id}")
async def get_trace_detail(trace_id: str):
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")

    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")

    spans = await Span.find(Span.trace_id == tid).sort("+step_index").to_list()

    return {
        "trace": _trace_summary(trace),
        "metrics": [m.model_dump() for m in trace.metrics],
        "spans": [_span_detail(s) for s in spans],
    }


@router.get("/{trace_id}/cost")
async def get_trace_cost(trace_id: str):
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")

    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")

    spans = await Span.find(Span.trace_id == tid, Span.cost_usd != None).to_list()

    by_model: dict[str, float] = {}
    by_provider: dict[str, float] = {}
    for s in spans:
        model_key = s.model or "unknown"
        provider_key = s.provider or "unknown"
        by_model[model_key] = by_model.get(model_key, 0) + (s.cost_usd or 0)
        by_provider[provider_key] = by_provider.get(provider_key, 0) + (s.cost_usd or 0)

    return {
        "trace_id": str(trace.id),
        "total_cost_usd": trace.total_cost_usd,
        "by_model": by_model,
        "by_provider": by_provider,
        "spans": [
            {"span_id": str(s.id), "model": s.model, "cost_usd": s.cost_usd, "tokens_in": s.tokens_in, "tokens_out": s.tokens_out}
            for s in spans
        ],
    }


@router.get("/{trace_id}/summary")
async def get_trace_summary(trace_id: str):
    """Human-readable summary of a trace — designed for dashboards and clients."""
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")

    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")

    spans = await Span.find(Span.trace_id == tid).sort("+step_index").to_list()

    llm_spans = [s for s in spans if s.span_type.value == "llm"]
    tool_spans = [s for s in spans if s.span_type.value == "tool"]

    steps = []
    for i, s in enumerate(llm_spans):
        step = {
            "step": i + 1,
            "provider": s.provider or "unknown",
            "model": s.model or "unknown",
            "latency": f"{s.latency_ms}ms" if s.latency_ms else "N/A",
            "tokens": f"{s.tokens_in or 0} in / {s.tokens_out or 0} out",
            "cost": f"${s.cost_usd:.6f}" if s.cost_usd else "N/A",
            "response_preview": (s.response or "")[:150],
        }
        if s.metrics:
            step["metrics"] = {
                m.name if isinstance(m.name, str) else m.name.value: m.value
                for m in s.metrics
            }
        steps.append(step)

    duration_str = f"{trace.duration_ms / 1000:.1f}s" if trace.duration_ms else "en cours"
    total_tokens = trace.total_tokens_in + trace.total_tokens_out

    summary = {
        "trace_id": str(trace.id),
        "name": trace.name,
        "status": trace.status.value,
        "duration": duration_str,
        "llm_calls": len(llm_spans),
        "tool_calls": len(tool_spans),
        "total_tokens": total_tokens,
        "total_cost": f"${trace.total_cost_usd:.4f}",
        "steps": steps,
    }

    if trace.metrics:
        summary["trace_metrics"] = {
            m.name if isinstance(m.name, str) else m.name.value: {
                "value": m.value,
                "detail": m.reason,
            }
            for m in trace.metrics
        }

    return summary


# ---------------------------------------------------------------------------
def _trace_summary(t: Trace) -> dict:
    return {
        "id": str(t.id),
        "name": t.name,
        "status": t.status.value,
        "total_spans": t.total_spans,
        "total_llm_spans": t.total_llm_spans,
        "total_tool_spans": t.total_tool_spans,
        "total_tokens_in": t.total_tokens_in,
        "total_tokens_out": t.total_tokens_out,
        "total_cost_usd": t.total_cost_usd,
        "duration_ms": t.duration_ms,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "ended_at": t.ended_at.isoformat() if t.ended_at else None,
        "has_metrics": len(t.metrics) > 0,
    }


def _span_detail(s: Span) -> dict:
    base = {
        "id": str(s.id),
        "step_index": s.step_index,
        "span_type": s.span_type.value,
        "sdk_span_id": s.sdk_span_id,
        "parent_span_id": s.parent_span_id,
        "caller_chain": s.caller_chain,
        "latency_ms": s.latency_ms,
        "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        "metrics": [m.model_dump() for m in s.metrics],
    }
    if s.span_type.value == "llm":
        msgs_in_truncated = None
        if s.messages_in:
            msgs_in_truncated = [
                {**m, "content": (m.get("content", "") or "")[:500]}
                for m in s.messages_in if isinstance(m, dict)
            ]
        base.update({
            "provider": s.provider,
            "model": s.model,
            "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out,
            "cost_usd": s.cost_usd,
            "temperature": s.temperature,
            "response_preview": (s.response or "")[:300],
            "response": (s.response or "")[:2000],
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
            "tool_output": (s.tool_output or "")[:300],
            "tool_error": s.tool_error,
        })
    return base
