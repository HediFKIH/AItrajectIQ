"""
Ingest routes — receives data from the SDK.
POST /ingest/trace/start   → creates a Trace, returns trace_id
POST /ingest/trace/end     → closes a Trace, triggers trace-level eval
POST /ingest/span          → stores a Span, triggers span-level eval
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional
from beanie import PydanticObjectId

from models.trace import Trace, TraceStatus
from models.span import Span, SpanType
from models.project import Project
from utils.cost import compute_cost
from evaluation.runner import evaluate_span, evaluate_trace

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

async def _auth_project(project_id: str, api_key: str) -> Project:
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")
    project = await Project.get(pid)
    if project is None:
        raise HTTPException(404, "Project not found")
    if project.api_key.key != api_key or not project.api_key.active:
        raise HTTPException(401, "Invalid or inactive API key")
    return project


# ===================================================================
# TRACE START
# ===================================================================

class TraceStartRequest(BaseModel):
    project_id: str
    name: Optional[str] = None
    metadata: Optional[dict] = None


class TraceStartResponse(BaseModel):
    trace_id: str


@router.post("/trace/start", response_model=TraceStartResponse)
async def start_trace(
    body: TraceStartRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    project = await _auth_project(body.project_id, x_api_key)
    trace = Trace(
        project_id=project.id,
        name=body.name,
        metadata=body.metadata,
    )
    print(f"Started trace: {trace}")
    await trace.insert()
    
    return TraceStartResponse(trace_id=str(trace.id))


# ===================================================================
# TRACE END
# ===================================================================

class TraceEndRequest(BaseModel):
    trace_id: str
    status: Optional[str] = "completed"


class TraceEndResponse(BaseModel):
    trace_id: str
    status: str
    total_spans: int
    total_cost_usd: float


@router.post("/trace/end", response_model=TraceEndResponse)
async def end_trace(
    body: TraceEndRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    try:
        tid = PydanticObjectId(body.trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")

    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")

    project = await Project.get(trace.project_id)
    if project is None or project.api_key.key != x_api_key:
        raise HTTPException(401, "Unauthorized")

    try:
        trace.status = TraceStatus(body.status)
    except ValueError:
        trace.status = TraceStatus.completed

    trace.ended_at = datetime.utcnow()
    if trace.started_at:
        trace.duration_ms = int((trace.ended_at - trace.started_at).total_seconds() * 1000)

    # Guard: ended_at must never exist without duration_ms
    if trace.ended_at is not None and trace.duration_ms is None:
        # This is a backend invariant violation — fail fast.
        raise HTTPException(
            500,
            f"Trace duration missing: trace_id={body.trace_id}, started_at={trace.started_at}, ended_at={trace.ended_at}",
        )

    await trace.save()

    asyncio.create_task(evaluate_trace(trace.id))

    return TraceEndResponse(
        trace_id=str(trace.id),
        status=trace.status.value,
        total_spans=trace.total_spans,
        total_cost_usd=trace.total_cost_usd,
    )


# ===================================================================
# SPAN
# ===================================================================

class SpanRequest(BaseModel):
    trace_id: str
    span_type: str = "llm"
    step_index: int = 0

    # Hierarchy (from SDK)
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    caller_chain: Optional[list] = None

    # LLM
    provider: Optional[str] = None
    model: Optional[str] = None
    messages_in: Optional[list] = None
    response: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
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
async def ingest_span(
    body: SpanRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    try:
        tid = PydanticObjectId(body.trace_id)
    except Exception:
        raise HTTPException(400, "Invalid trace_id")

    trace = await Trace.get(tid)
    if trace is None:
        raise HTTPException(404, "Trace not found")

    project = await Project.get(trace.project_id)
    if project is None or project.api_key.key != x_api_key:
        raise HTTPException(401, "Unauthorized")

    try:
        stype = SpanType(body.span_type)
    except ValueError:
        stype = SpanType.llm

    cost = None
    if stype == SpanType.llm and body.model and body.tokens_in and body.tokens_out:
        cost = compute_cost(body.model, body.tokens_in, body.tokens_out)

    span = Span(
        trace_id=tid,
        project_id=project.id,
        step_index=body.step_index,
        span_type=stype,
        sdk_span_id=body.span_id,
        parent_span_id=body.parent_span_id,
        caller_chain=body.caller_chain,
        provider=body.provider,
        model=body.model,
        messages_in=body.messages_in,
        response=body.response,
        tokens_in=body.tokens_in,
        tokens_out=body.tokens_out,
        cost_usd=cost,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tool_name=body.tool_name,
        tool_input=body.tool_input,
        tool_output=body.tool_output,
        tool_error=body.tool_error,
        latency_ms=body.latency_ms,
        iteration=body.iteration,
        action_success=body.action_success,
        similarity_score=body.similarity_score,
        feedback=body.feedback,
    )
    print(f"Ingesting span: {span}")
    await span.insert()

    # Update trace aggregates
    trace.total_spans += 1
    if stype == SpanType.llm:
        trace.total_llm_spans += 1
        trace.total_tokens_in += body.tokens_in or 0
        trace.total_tokens_out += body.tokens_out or 0
        trace.total_cost_usd += cost or 0
    else:
        trace.total_tool_spans += 1
    await trace.save()

    asyncio.create_task(evaluate_span(span.id))

    return SpanResponse(span_id=str(span.id), cost_usd=cost)
