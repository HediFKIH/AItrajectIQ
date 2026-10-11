"""
Traces & spans read routes. Every route needs a `read` API key of the trace's project.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from beanie import PydanticObjectId
from typing import Optional
from datetime import datetime

from models.trace import Trace, TraceStatus
from models.span import Span
from security.api_keys import READ, authenticate
from services.insights import project_stats, span_detail, trace_summary

router = APIRouter(prefix="/traces", tags=["traces"])


async def _readable_trace(request: Request, trace_id: str) -> Trace:
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")
    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")
    await authenticate(request, READ, project_id=str(trace.project_id), allow_legacy=True)
    return trace


@router.get("/stats")
async def get_project_stats(request: Request, project_id: str = Query(...)):
    """Aggregate stats for ALL traces of a project — used by the dashboard."""
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")
    await authenticate(request, READ, project_id=project_id, allow_legacy=True)

    return await project_stats(pid)


@router.get("/")
async def list_traces(
    request: Request,
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
    await authenticate(request, READ, project_id=project_id, allow_legacy=True)

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
        "traces": [trace_summary(t) for t in traces],
    }


@router.get("/{trace_id}")
async def get_trace_detail(trace_id: str, request: Request):
    trace = await _readable_trace(request, trace_id)
    spans = await Span.find(Span.trace_id == trace.id).sort("+step_index").to_list()

    return {
        "trace": trace_summary(trace),
        "metrics": [m.model_dump(mode="json") for m in trace.metrics],
        "spans": [span_detail(s) for s in spans],
    }


@router.get("/{trace_id}/cost")
async def get_trace_cost(trace_id: str, request: Request):
    trace = await _readable_trace(request, trace_id)
    spans = await Span.find(Span.trace_id == trace.id, Span.cost_usd != None).to_list()

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
async def get_trace_summary(trace_id: str, request: Request):
    """Human-readable summary of a trace — designed for dashboards and clients."""
    trace = await _readable_trace(request, trace_id)
    spans = await Span.find(Span.trace_id == trace.id).sort("+step_index").to_list()

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
