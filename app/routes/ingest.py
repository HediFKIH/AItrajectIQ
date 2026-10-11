"""
Ingest routes — receives data from the SDK.
POST /ingest/trace/start   → creates a Trace, returns trace_id
POST /ingest/trace/end     → closes a Trace, triggers trace-level eval
POST /ingest/span          → stores a Span, triggers span-level eval

Every route requires a project API key (see security/api_keys.py).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from beanie import PydanticObjectId
from pymongo.errors import DuplicateKeyError

from models.trace import Trace, TraceStatus
from models.span import Span, SpanType
from security.api_keys import INGEST, authenticate
from utils.cost import compute_cost, has_price
from evaluation.runner import evaluate_span
from evaluation.scheduler import bounded, schedule
from realtime import publish
from services.insights import trace_summary
from services.sdk_traces import close_trace

router = APIRouter(prefix="/ingest", tags=["ingest"])


async def _get_trace(trace_id: str) -> Trace:
    try:
        tid = PydanticObjectId(trace_id)
    except Exception:
        raise HTTPException(400, f"Invalid trace_id: {trace_id!r}")
    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")
    return trace


# ===================================================================
# TRACE START
# ===================================================================

class TraceStartRequest(BaseModel):
    project_id: Optional[str] = None   # optional: the API key identifies the project
    name: Optional[str] = None
    metadata: Optional[dict] = None


class TraceStartResponse(BaseModel):
    trace_id: str


@router.post("/trace/start", response_model=TraceStartResponse)
async def start_trace(body: TraceStartRequest, request: Request):
    auth = await authenticate(request, INGEST, project_id=body.project_id, allow_legacy=True)
    project = auth.project
    trace = Trace(
        project_id=project.id,
        name=body.name,
        metadata=body.metadata,
        last_activity_at=datetime.utcnow(),
    )
    print(f"Started trace: {trace}")
    await trace.insert()
    publish(project.id, "trace.started", {"trace": trace_summary(trace)})

    return TraceStartResponse(trace_id=str(trace.id))


# ===================================================================
# TRACE END
# ===================================================================

class TraceEndRequest(BaseModel):
    trace_id: str
    status: Optional[str] = "completed"
    ended_at: Optional[datetime] = None   # SDK auto-traces closed after idling: time of the last span


class TraceEndResponse(BaseModel):
    trace_id: str
    status: str
    total_spans: int
    total_cost_usd: float


@router.post("/trace/end", response_model=TraceEndResponse)
async def end_trace(body: TraceEndRequest, request: Request):
    trace = await _get_trace(body.trace_id)
    await authenticate(request, INGEST, project_id=str(trace.project_id), allow_legacy=True)

    try:
        status = TraceStatus(body.status)
    except ValueError:
        status = TraceStatus.completed
    ended_at = None
    if body.ended_at is not None:
        ended_at = body.ended_at
        if ended_at.tzinfo is not None:
            ended_at = ended_at.astimezone(timezone.utc).replace(tzinfo=None)
        ended_at = min(max(ended_at, trace.started_at), datetime.utcnow())
    await close_trace(trace, status, ended_at=ended_at)

    return TraceEndResponse(
        trace_id=str(trace.id),
        status=trace.status.value,
        total_spans=trace.total_spans,
        total_cost_usd=trace.total_cost_usd,
    )


class TraceReopenRequest(BaseModel):
    trace_id: str


@router.post("/trace/reopen")
async def reopen_trace(body: TraceReopenRequest, request: Request):
    """Back to running: the agent resumed after its trace was ended (e.g. Cursor stops while
    waiting for the user to allow a command, then continues the same run)."""
    trace = await _get_trace(body.trace_id)
    await authenticate(request, INGEST, project_id=str(trace.project_id), allow_legacy=True)
    if trace.status != TraceStatus.running:
        await Trace.get_motor_collection().update_one(
            {"_id": trace.id},
            {"$set": {"status": TraceStatus.running.value, "ended_at": None, "duration_ms": None,
                      "last_activity_at": datetime.utcnow()}},
        )
        publish(trace.project_id, "trace.reopened", {"trace_id": str(trace.id)})
    return {"trace_id": str(trace.id), "status": TraceStatus.running.value}


# ===================================================================
# SPAN
# ===================================================================

class SpanRequest(BaseModel):
    trace_id: str
    span_type: str = "llm"
    step_index: int = 0
    name: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None

    # Hierarchy (from SDK)
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    caller_chain: Optional[list] = None

    # Agent
    agent_name: Optional[str] = None
    role: Optional[str] = None

    # LLM
    provider: Optional[str] = None
    model: Optional[str] = None
    messages_in: Optional[list] = None
    response: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cache_read_tokens: Optional[int] = None    # included in tokens_in, billed at the cache rate
    cache_write_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


    # Tool
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None

    latency_ms: Optional[int] = None

    # Generic agent execution feedback
    iteration: Optional[int] = None
    action_success: Optional[bool] = None
    similarity_score: Optional[float] = None
    feedback: Optional[str] = None


class SpanResponse(BaseModel):
    span_id: str
    cost_usd: Optional[float] = None


@router.post("/span", response_model=SpanResponse)
async def ingest_span(body: SpanRequest, request: Request):
    trace = await _get_trace(body.trace_id)
    auth = await authenticate(request, INGEST, project_id=str(trace.project_id), allow_legacy=True)
    project = auth.project
    tid = trace.id

    try:
        stype = SpanType(body.span_type)
    except ValueError:
        stype = SpanType.llm

    cost = None
    if stype == SpanType.llm and body.model and (body.tokens_in or body.tokens_out):
        cost = compute_cost(body.model, body.tokens_in or 0, body.tokens_out or 0,
                            body.cache_read_tokens or 0, body.cache_write_tokens or 0)
        if not has_price(body.model):
            body.metadata = {**(body.metadata or {}), "price": "default"}

    # tool_span() reports failures as `error`; the tool metrics read `tool_error`.
    tool_error = body.tool_error or (body.error if stype == SpanType.tool else None)

    if body.span_id:
        existing = await Span.find_one(Span.trace_id == tid, Span.sdk_span_id == body.span_id)
        if existing is not None:
            return await _update_span(existing, body, stype, cost, tool_error, trace, project)

    span = Span(
        trace_id=tid,
        project_id=project.id,
        step_index=body.step_index,
        span_type=stype,
        name=body.name,
        status=body.status,
        error=body.error,
        metadata=body.metadata or None,
        sdk_span_id=body.span_id,
        parent_span_id=body.parent_span_id,
        caller_chain=body.caller_chain,
        agent_name=body.agent_name,
        role=body.role,
        provider=body.provider,
        model=body.model,
        messages_in=body.messages_in,
        response=body.response,
        tokens_in=body.tokens_in,
        tokens_out=body.tokens_out,
        cache_read_tokens=body.cache_read_tokens,
        cache_write_tokens=body.cache_write_tokens,
        cost_usd=cost,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tool_name=body.tool_name,
        tool_input=body.tool_input,
        tool_output=body.tool_output,
        tool_error=tool_error,
        latency_ms=body.latency_ms,
        iteration=body.iteration,
        action_success=body.action_success,
        similarity_score=body.similarity_score,
        feedback=body.feedback,
    )
    try:
        await span.insert()
    except DuplicateKeyError:
        # The same span_id was inserted concurrently: this request becomes its update.
        existing = await Span.find_one(Span.trace_id == tid, Span.sdk_span_id == body.span_id)
        if existing is None:
            raise
        return await _update_span(existing, body, stype, cost, tool_error, trace, project)

    # Atomic increments: the SDK sends spans concurrently from several threads.
    inc = {"total_spans": 1}
    if stype == SpanType.llm:
        inc.update(
            total_llm_spans=1,
            total_tokens_in=body.tokens_in or 0,
            total_tokens_out=body.tokens_out or 0,
            total_cost_usd=cost or 0.0,
        )
    elif stype == SpanType.tool:
        inc["total_tool_spans"] = 1
    await Trace.get_motor_collection().update_one(
        {"_id": tid}, {"$inc": inc, "$set": {"last_activity_at": datetime.utcnow()}}
    )
    publish(project.id, "span.created", {
        "trace_id": str(tid),
        "count": 1,
        "span": {
            "id": str(span.id),
            "span_type": stype.value,
            "name": body.name or body.tool_name or body.agent_name or body.model,
            "cost_usd": cost,
        },
    })

    if stype == SpanType.llm:
        schedule(bounded(evaluate_span(span.id)))

    return SpanResponse(span_id=str(span.id), cost_usd=cost)


_UPDATABLE = (
    "name", "status", "error", "metadata", "parent_span_id", "caller_chain", "agent_name", "role",
    "provider", "model", "messages_in", "response", "tokens_in", "tokens_out",
    "cache_read_tokens", "cache_write_tokens", "temperature",
    "max_tokens", "tool_name", "tool_input", "tool_output", "latency_ms", "iteration",
    "action_success", "similarity_score", "feedback",
)


async def _update_span(span: Span, body: SpanRequest, stype: SpanType, cost, tool_error, trace, project):
    """A span sent again with the same span_id (opened, then completed) replaces the stored one;
    the trace counters only take the difference."""
    old_in, old_out, old_cost = span.tokens_in or 0, span.tokens_out or 0, span.cost_usd or 0.0
    old_response = span.response
    for field in _UPDATABLE:
        value = getattr(body, field)
        if value is not None:
            setattr(span, field, value)
    if tool_error is not None:
        span.tool_error = tool_error
    if cost is not None:
        span.cost_usd = cost
    span.timestamp = datetime.utcnow()      # SDK spans are stored when they end
    await span.save()

    inc = {}
    if stype == SpanType.llm:
        inc = {
            "total_tokens_in": (span.tokens_in or 0) - old_in,
            "total_tokens_out": (span.tokens_out or 0) - old_out,
            "total_cost_usd": (span.cost_usd or 0.0) - old_cost,
        }
    await Trace.get_motor_collection().update_one(
        {"_id": trace.id}, {"$inc": inc, "$set": {"last_activity_at": datetime.utcnow()}} if inc
        else {"$set": {"last_activity_at": datetime.utcnow()}}
    )
    publish(project.id, "span.created", {
        "trace_id": str(trace.id),
        "count": 0,
        "span": {"id": str(span.id), "span_type": stype.value, "name": span.name, "cost_usd": span.cost_usd},
    })
    if stype == SpanType.llm and span.response and span.response != old_response:
        schedule(bounded(evaluate_span(span.id)))
    return SpanResponse(span_id=str(span.id), cost_usd=span.cost_usd)
