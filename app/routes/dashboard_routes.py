"""
Dashboard API. Every route requires a project API key with the `read` scope; the project
always comes from the key. Everything is read-only except the two POST routes, which call
the evaluation LLM.

GET /api/v1/project                    project, key and enabled metrics
GET /api/v1/overview?range=24h         KPIs of the period vs the previous one
GET /api/v1/timeseries?range=24h       traces, errors, cost, tokens, latency per bucket
GET /api/v1/traces                     paginated, filterable trace list
GET /api/v1/traces/{id}                span tree with content and waterfall offsets
GET /api/v1/costs?range=24h            cost by model and agent, most expensive traces
GET /api/v1/costs/timeseries           cost per bucket split by model
GET /api/v1/metrics/summary            average of every evaluation metric
GET /api/v1/metrics/timeseries?name=   one metric's average per bucket
GET /api/v1/metrics/worst?name=        spans with the poorest score for a metric
GET /api/v1/metrics/traces?name=       latest traces evaluated on a trace-level metric
GET /api/v1/recommendations?range=7d   model, caching, prompt and agent recommendations
GET /api/v1/traces/{id}/recommendations   the same, step by step, for one trace
POST /api/v1/recommendations/prompt   AI rewrite of a recommended prompt (on demand)
POST /api/v1/evaluations/run          evaluate one trace, or the period's traces the current judge hasn't seen
GET /api/v1/stream                     Server-Sent Events of the project's activity
"""

import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from evaluation.runner import judge_status, judge_tag
from models.metric import MetricName
from models.trace import TraceStatus
from realtime import bus
from security.api_keys import READ, AuthContext, extract_api_key, has_scope, require_api_key, resolve_api_key
from services import dashboard, evaluations, insights, prompt_advisor, recommendations

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

RangeKey = Literal["1h", "24h", "7d", "30d"]
HEARTBEAT_S = 15.0

reader = require_api_key(READ)


def _rng(key: str) -> dashboard.Range:
    try:
        return dashboard.get_range(key)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def _metric(name: str) -> str:
    if name not in {m.value for m in MetricName}:
        raise HTTPException(422, f"Unknown metric {name!r}")
    return name


@router.get("/project")
async def project_info(auth: AuthContext = Depends(reader)):
    p = auth.project
    return {
        "id": str(p.id),
        "name": p.name,
        "type": getattr(p.type, "value", p.type),
        "task": getattr(p.task, "value", p.task),
        "enabled_metrics": [getattr(m, "value", m) for m in p.enabled_metrics or []],
        "key": {"name": auth.key.name, "prefix": auth.key.prefix, "scopes": sorted(auth.key.scopes)},
        "judge": judge_status(),
    }


@router.get("/overview")
async def overview(range: RangeKey = "24h", auth: AuthContext = Depends(reader)):
    return await dashboard.overview(auth.project.id, _rng(range))


@router.get("/timeseries")
async def timeseries(range: RangeKey = "24h", auth: AuthContext = Depends(reader)):
    return await dashboard.timeseries(auth.project.id, _rng(range))


@router.get("/traces")
async def list_traces(
    range: Optional[RangeKey] = None,
    status: Optional[TraceStatus] = None,
    source: Optional[Literal["sdk", "otel", "cursor", "claude-code"]] = None,
    q: Optional[str] = Query(None, max_length=200),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    auth: AuthContext = Depends(reader),
):
    return await dashboard.list_traces(
        auth.project.id,
        rng=_rng(range) if range else None,
        status=status.value if status else None,
        source=source,
        q=q or None,
        page=page,
        page_size=page_size,
    )


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str, auth: AuthContext = Depends(reader)):
    trace = await insights.find_project_trace(auth.project.id, trace_id)
    if trace is None:
        raise HTTPException(404, "Trace not found")
    return await dashboard.trace_tree(trace)


@router.get("/traces/{trace_id}/recommendations")
async def trace_recommendations(trace_id: str, auth: AuthContext = Depends(reader)):
    trace = await insights.find_project_trace(auth.project.id, trace_id)
    if trace is None:
        raise HTTPException(404, "Trace not found")
    return await recommendations.trace_recommendations(trace)


@router.get("/costs")
async def costs(range: RangeKey = "24h", auth: AuthContext = Depends(reader)):
    return await dashboard.costs(auth.project.id, _rng(range))


@router.get("/costs/timeseries")
async def cost_timeseries(range: RangeKey = "24h", auth: AuthContext = Depends(reader)):
    return await dashboard.cost_timeseries(auth.project.id, _rng(range))


@router.get("/metrics/summary")
async def metrics_summary(range: RangeKey = "7d", auth: AuthContext = Depends(reader)):
    rng = _rng(range)
    summary = await insights.metrics_summary(auth.project.id, rng.days)
    summary["range"] = rng.key
    summary["enabled_metrics"] = [getattr(m, "value", m) for m in auth.project.enabled_metrics or []]
    return summary


@router.get("/metrics/timeseries")
async def metric_timeseries(name: str, range: RangeKey = "7d", auth: AuthContext = Depends(reader)):
    return await dashboard.metric_timeseries(auth.project.id, _metric(name), _rng(range))


@router.get("/metrics/worst")
async def worst_spans(
    name: str,
    range: RangeKey = "7d",
    limit: int = Query(10, ge=1, le=50),
    auth: AuthContext = Depends(reader),
):
    rng = _rng(range)
    since = datetime.utcnow() - rng.span
    return await insights.problem_spans(auth.project.id, "low_score", since, metric=_metric(name), limit=limit)


@router.get("/metrics/traces")
async def evaluated_traces(
    name: str,
    range: RangeKey = "7d",
    limit: int = Query(50, ge=1, le=200),
    auth: AuthContext = Depends(reader),
):
    rng = _rng(range)
    since = datetime.utcnow() - rng.span
    return await insights.evaluated_traces(auth.project.id, _metric(name), since, limit=limit)


@router.get("/recommendations")
async def project_recommendations(range: RangeKey = "7d", auth: AuthContext = Depends(reader)):
    rng = _rng(range)
    result = await recommendations.recommendations(auth.project.id, datetime.utcnow() - rng.span)
    return {"range": rng.key, **result}


class PromptRewrite(BaseModel):
    prompt_id: str
    range: RangeKey = "7d"
    trace_id: Optional[str] = None


@router.post("/recommendations/prompt")
async def rewrite_prompt(body: PromptRewrite, auth: AuthContext = Depends(reader)):
    if body.trace_id:
        trace = await insights.find_project_trace(auth.project.id, body.trace_id)
        if trace is None:
            raise HTTPException(404, "Trace not found")
        result = await recommendations.trace_recommendations(trace)
    else:
        rng = _rng(body.range)
        result = await recommendations.recommendations(auth.project.id, datetime.utcnow() - rng.span)
    item = next((p for p in result["prompts"] if p["id"] == body.prompt_id), None)
    if item is None:
        raise HTTPException(404, "No recommended prompt with this id")
    try:
        return await prompt_advisor.rewrite(item, recommendations.prompt_findings(result))
    except prompt_advisor.RewriteUnavailable as exc:
        raise HTTPException(503, str(exc))


class EvaluationRun(BaseModel):
    range: RangeKey = "7d"
    trace_id: Optional[str] = None


@router.post("/evaluations/run")
async def run_evaluations(body: EvaluationRun, auth: AuthContext = Depends(reader)):
    """Evaluate one trace, or the latest traces of the period without a verdict of the
    current judge (at most evaluations.MAX_BATCH). Results arrive as metrics.updated events."""
    if judge_tag() is None:
        raise HTTPException(503, evaluations.judge_missing_message())
    if body.trace_id:
        trace = await insights.find_project_trace(auth.project.id, body.trace_id)
        if trace is None:
            raise HTTPException(404, "Trace not found")
        if trace.status == TraceStatus.running:
            raise HTTPException(409, "The trace is still running: it will be evaluated when it ends.")
        return {"started": int(evaluations.evaluate_one(trace))}
    rng = _rng(body.range)
    return {"started": await evaluations.evaluate_pending(auth.project.id, datetime.utcnow() - rng.span)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _key_still_valid(raw_key: Optional[str]) -> bool:
    info = await resolve_api_key(raw_key)
    return info is not None and has_scope(info, READ)


@router.get("/stream")
async def stream(request: Request, auth: AuthContext = Depends(reader)):
    """Live project events as Server-Sent Events. A heartbeat every 15 s keeps proxies
    from closing the connection; the stream ends when the key is revoked."""
    project_id = str(auth.project.id)
    raw_key = extract_api_key(request.headers)

    async def events():
        with bus.subscribe(project_id) as sub:
            yield "retry: 3000\n\n"
            yield _sse("ready", {"project_id": project_id})
            while True:
                event = await sub.next(timeout=HEARTBEAT_S)
                if await request.is_disconnected():
                    break
                dropped = sub.take_dropped()
                if dropped:
                    yield _sse("resync", {"dropped": dropped})
                if event is None:
                    if not await _key_still_valid(raw_key):
                        yield _sse("revoked", {})
                        break
                    yield ": heartbeat\n\n"
                    continue
                yield _sse(event["type"], {**event["data"], "ts": event["ts"]})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
