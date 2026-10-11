"""
OTLP/HTTP receiver: point any OpenTelemetry exporter at AItrajectIQ.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8000/otel
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf          (or http/json)
    OTEL_EXPORTER_OTLP_HEADERS=x-api-key=atq_...

Exporters append /v1/traces, /v1/logs and /v1/metrics to the endpoint.
gRPC (port 4317) is not supported.

POST /otel/v1/claude-code/stop receives the Claude Code Stop hook (final answer and tool
arguments that the telemetry doesn't carry), see otel/claude_code_answer.py.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from pymongo.errors import ConnectionFailure

from otel import claude_code_answer
from otel.decode import (
    OtlpDecodeError,
    UnsupportedMediaType,
    decode_logs,
    decode_traces,
    encode_export_response,
    response_media_type,
)
from otel.ingest import ingest_log_records, ingest_spans
from security.api_keys import INGEST, AuthContext, require_api_key

log = logging.getLogger("aitrajectiq.otel")

router = APIRouter(prefix="/otel/v1", tags=["opentelemetry"])


def _error(status_code: int, message: str) -> Response:
    return Response(
        content=json.dumps({"detail": message}),
        status_code=status_code,
        media_type="application/json",
    )


def _decode_error(exc: OtlpDecodeError) -> Response:
    return _error(415 if isinstance(exc, UnsupportedMediaType) else 400, str(exc))


@router.post("/traces")
async def export_traces(request: Request, auth: AuthContext = Depends(require_api_key(INGEST))):
    try:
        spans, rejected, media_type = decode_traces(
            await request.body(),
            request.headers.get("content-type"),
            request.headers.get("content-encoding"),
        )
    except OtlpDecodeError as exc:
        return _decode_error(exc)

    try:
        result = await ingest_spans(auth.project, spans)
    except ConnectionFailure as exc:
        # 503 is retryable for OTLP exporters; the batch will be re-sent.
        return _error(503, f"Database unavailable: {exc}")

    log.debug(
        "OTLP traces project=%s received=%d stored=%d skipped=%d rejected=%d",
        auth.project.id, len(spans), result.stored, result.skipped, rejected + result.rejected,
    )
    rejected += result.rejected
    message = f"{rejected} span(s) rejected, see the AItrajectIQ backend logs" if rejected else None
    return Response(
        content=encode_export_response("traces", media_type, rejected, message),
        media_type=media_type,
    )


@router.post("/logs")
async def export_logs(request: Request, auth: AuthContext = Depends(require_api_key(INGEST))):
    try:
        records, media_type = decode_logs(
            await request.body(),
            request.headers.get("content-type"),
            request.headers.get("content-encoding"),
        )
    except OtlpDecodeError as exc:
        return _decode_error(exc)

    try:
        accepted, ignored = await ingest_log_records(auth.project, records)
    except ConnectionFailure as exc:
        return _error(503, f"Database unavailable: {exc}")

    log.debug("OTLP logs project=%s content_events=%d ignored=%d", auth.project.id, accepted, ignored)
    return Response(content=encode_export_response("logs", media_type), media_type=media_type)


class ClaudeCodeStop(BaseModel):
    """What the Claude Code Stop hook read in the session transcript for the last prompt."""
    session_id: str = Field(min_length=1, max_length=200)
    prompt: Optional[str] = None
    prompt_at: Optional[datetime] = None
    answer: Optional[str] = None
    tools: List[dict] = Field(default_factory=list)
    stopped_at: Optional[datetime] = None


@router.post("/claude-code/stop")
async def claude_code_stop(body: ClaudeCodeStop, auth: AuthContext = Depends(require_api_key(INGEST))):
    stop = body.model_dump()
    stop["prompt_at"] = claude_code_answer.naive_utc(body.prompt_at)
    stop["stopped_at"] = claude_code_answer.naive_utc(body.stopped_at)
    stop["prompt"] = (body.prompt or "")[:claude_code_answer.MAX_TEXT]
    stop["answer"] = (body.answer or "")[:claude_code_answer.MAX_TEXT]
    stop["tools"] = [{"name": t.get("name"), "input": claude_code_answer.compact_input(t.get("input"))}
                     for t in body.tools[:claude_code_answer.MAX_TOOLS]]
    try:
        return {"status": await claude_code_answer.record_stop(auth.project, stop)}
    except ConnectionFailure as exc:
        return _error(503, f"Database unavailable: {exc}")


@router.post("/metrics", dependencies=[Depends(require_api_key(INGEST))])
async def export_metrics(request: Request):
    """Accepted and ignored, so exporters configured for every signal don't log errors."""
    await request.body()
    media_type = response_media_type(request.headers.get("content-type"))
    return Response(content=encode_export_response("metrics", media_type), media_type=media_type)
