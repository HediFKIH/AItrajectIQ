"""
Persist OTLP spans and GenAI log events, and finalize OpenTelemetry traces.

One OTel trace id maps to one AItrajectIQ Trace (`Trace.external_id`), created on the
first GenAI span. OTel has no "trace ended" signal, so a trace is finalized when its
root span arrives (even when the root itself isn't stored, e.g. an HTTP server span),
or after AITRAJECTIQ_OTEL_IDLE_S seconds without new spans (see `run_sweeper`).
LLM spans are evaluated at finalization, once buffered content events are merged in.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from beanie import PydanticObjectId
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure, DuplicateKeyError

from evaluation.runner import evaluate_span, evaluate_trace
from evaluation.scheduler import bounded, schedule
from models.otel_event import OtelEvent
from models.project import Project
from models.span import Span, SpanType
from models.trace import Trace, TraceStatus
from otel import claude_code_answer
from otel.claude_code import TRACE_NAME as CLAUDE_TRACE_NAME, claude_code_spans, is_claude_code
from otel.decode import OtlpSpan
from otel.finalize import (
    apply_content_events,
    duration_ms,
    is_evaluable,
    nested_llm_span_ids,
    ordered,
    time_bounds,
    trace_totals,
)
from otel.mapping import MappedSpan, is_content_event, map_span
from realtime import publish
from services.insights import trace_summary
from utils.cost import compute_cost, has_price

log = logging.getLogger("aitrajectiq.otel")

IDLE_SECONDS = float(os.getenv("AITRAJECTIQ_OTEL_IDLE_S", "60"))
# Claude Code goes quiet while the user approves a command; its Stop hook finalizes the
# trace when the run is really over, so the sweeper only covers installs without the hook.
CLAUDE_CODE_IDLE_SECONDS = float(os.getenv("AITRAJECTIQ_CLAUDE_CODE_IDLE_S", "300"))

_TRACE_METADATA_KEYS = ("service", "session_id", "user_id", "function_id", "workflow")


@dataclass
class IngestResult:
    stored: int = 0      # spans stored (new or re-sent)
    skipped: int = 0     # spans without any GenAI/agent signal
    rejected: int = 0    # spans that failed to ingest
    traces: int = 0


def _dt(ns: int) -> Optional[datetime]:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).replace(tzinfo=None)


def _span_doc(
    project_id: PydanticObjectId,
    trace_id: PydanticObjectId,
    otel_span: OtlpSpan,
    mapped: MappedSpan,
    step_index: int,
) -> Span:
    cost = mapped.cost_usd
    metadata = dict(mapped.metadata or {})
    if mapped.cache_read_tokens or mapped.cache_write_tokens:
        metadata["usage"] = {
            "input_tokens": mapped.tokens_in,
            "output_tokens": mapped.tokens_out,
            "cache_read_tokens": mapped.cache_read_tokens or 0,
            "cache_write_tokens": mapped.cache_write_tokens or 0,
        }
    if cost is None and mapped.span_type == "llm" and mapped.model and (mapped.tokens_in or mapped.tokens_out):
        cost = compute_cost(mapped.model, mapped.tokens_in or 0, mapped.tokens_out or 0,
                            mapped.cache_read_tokens or 0, mapped.cache_write_tokens or 0)
        if not has_price(mapped.model):
            metadata["price"] = "default"
    latency = None
    if otel_span.start_ns and otel_span.end_ns:
        latency = max(0, (otel_span.end_ns - otel_span.start_ns) // 1_000_000)
    return Span(
        id=PydanticObjectId(),
        trace_id=trace_id,
        project_id=project_id,
        step_index=step_index,
        span_type=SpanType(mapped.span_type),
        name=mapped.name,
        status=mapped.status,
        error=mapped.error,
        source="otel",
        metadata=metadata or None,
        sdk_span_id=otel_span.span_id,
        parent_span_id=otel_span.parent_span_id,
        agent_name=mapped.agent_name,
        provider=mapped.provider,
        model=mapped.model,
        messages_in=mapped.messages_in,
        response=mapped.response,
        tokens_in=mapped.tokens_in,
        tokens_out=mapped.tokens_out,
        cache_read_tokens=mapped.cache_read_tokens,
        cache_write_tokens=mapped.cache_write_tokens,
        cost_usd=cost,
        temperature=mapped.temperature,
        max_tokens=mapped.max_tokens,
        tool_name=mapped.tool_name,
        tool_input=mapped.tool_input,
        tool_output=mapped.tool_output,
        tool_error=mapped.tool_error,
        latency_ms=latency,
        timestamp=_dt(otel_span.start_ns) or datetime.utcnow(),
    )


def _trace_metadata(kept: list) -> dict:
    meta: dict = {}
    for _, mapped in kept:
        for key in _TRACE_METADATA_KEYS:
            value = mapped.metadata.get(key)
            if value not in (None, "") and key not in meta:
                meta[key] = value
    return meta


async def _find_trace(project_id: PydanticObjectId, otel_trace_id: str) -> Optional[Trace]:
    return await Trace.find_one({"project_id": project_id, "external_id": otel_trace_id})


async def _create_trace(
    project_id: PydanticObjectId,
    otel_trace_id: str,
    name: str,
    started_at: Optional[datetime],
    metadata: dict,
) -> Trace:
    now = datetime.utcnow()
    trace = Trace(
        project_id=project_id,
        name=name,
        source="otel",
        external_id=otel_trace_id,
        metadata=metadata or None,
        started_at=started_at or now,
        last_activity_at=now,
    )
    try:
        await trace.insert()
        publish(project_id, "trace.started", {"trace": trace_summary(trace)})
        return trace
    except DuplicateKeyError:
        # A concurrent batch of the same trace created it first.
        existing = await _find_trace(project_id, otel_trace_id)
        if existing is None:
            raise
        return existing


async def _store_spans(project_id: PydanticObjectId, trace: Trace, kept: list) -> tuple:
    """Insert new spans and overwrite re-sent ones (exporter retries). Returns (docs, new_ids)."""
    previous = {
        s.sdk_span_id: s
        for s in await Span.find(
            {"trace_id": trace.id, "sdk_span_id": {"$in": [o.span_id for o, _ in kept]}}
        ).to_list()
    }
    docs, new_docs = [], []
    for index, (otel_span, mapped) in enumerate(sorted(kept, key=lambda pair: pair[0].start_ns)):
        doc = _span_doc(project_id, trace.id, otel_span, mapped, trace.total_spans + index)
        old = previous.get(otel_span.span_id)
        if old is not None:
            doc.id = old.id
            doc.step_index = old.step_index
            doc.metrics = old.metrics
            await doc.save()
        else:
            new_docs.append(doc)
        docs.append(doc)
    if new_docs:
        try:
            await Span.insert_many(new_docs)
        except BulkWriteError:
            # A concurrent batch stored some of these spans first: overwrite those instead.
            new_docs = [doc for doc in new_docs if not await _insert_or_overwrite(doc)]
    return docs, {d.id for d in new_docs}


async def _insert_or_overwrite(doc: Span) -> bool:
    """Store one span; True when it already existed and was overwritten."""
    existing = await Span.find_one({"trace_id": doc.trace_id, "sdk_span_id": doc.sdk_span_id})
    if existing is None:
        try:
            await doc.insert()
            return False
        except DuplicateKeyError:
            existing = await Span.find_one({"trace_id": doc.trace_id, "sdk_span_id": doc.sdk_span_id})
            if existing is None:
                raise
    doc.id, doc.step_index, doc.metrics = existing.id, existing.step_index, existing.metrics
    await doc.save()
    return True


async def _touch_trace(trace: Trace, group: list, root: Optional[OtlpSpan], new_docs: list) -> None:
    update: dict = {"$set": {"last_activity_at": datetime.utcnow()}}
    starts = [s.start_ns for s in group if s.start_ns]
    if starts:
        update["$min"] = {"started_at": _dt(min(starts))}
    if root is not None and root.name:
        update["$set"]["name"] = root.name
    if new_docs:
        # Provisional live counters; finalization recomputes them exactly.
        update["$inc"] = trace_totals(new_docs)
    await Trace.get_motor_collection().update_one({"_id": trace.id}, update)


async def _consolidate(trace: Trace, final: bool, root: Optional[OtlpSpan] = None) -> tuple:
    """Merge buffered content events, collapse nested LLM spans, renumber steps and
    refresh the trace aggregates. Returns (spans, ids of spans whose content changed)."""
    spans = await Span.find(Span.trace_id == trace.id).to_list()
    updates: dict = defaultdict(dict)

    events = await OtelEvent.find(
        {"project_id": trace.project_id, "otel_trace_id": trace.external_id}
    ).to_list()
    events_by_span = defaultdict(list)
    for event in events:
        events_by_span[event.otel_span_id].append(event.as_event())
    content_changed = set()
    for span in spans:
        if apply_content_events(span, events_by_span.get(span.sdk_span_id, [])):
            updates[span.id].update(messages_in=span.messages_in, response=span.response)
            content_changed.add(span.id)

    outer = nested_llm_span_ids(spans)
    for span in spans:
        if span.sdk_span_id in outer:
            span.span_type = SpanType.chain
            span.cost_usd = None
            updates[span.id].update(span_type=SpanType.chain.value, cost_usd=None)

    for index, span in enumerate(ordered(spans)):
        if span.step_index != index:
            span.step_index = index
            updates[span.id]["step_index"] = index

    if updates:
        await Span.get_motor_collection().bulk_write(
            [UpdateOne({"_id": sid}, {"$set": fields}) for sid, fields in updates.items()],
            ordered=False,
        )
    stored_span_ids = {s.sdk_span_id for s in spans}
    merged = [e.id for e in events if e.otel_span_id in stored_span_ids]
    if merged:
        await OtelEvent.find({"_id": {"$in": merged}}).delete()

    fields = trace_totals(spans)
    if final:
        started, ended = time_bounds(spans)
        failed = _trace_failed(spans, root)
        if root is not None:
            started = _dt(root.start_ns) or started
            ended = _dt(root.end_ns) or ended
            if root.name:
                fields["name"] = root.name
        fields["status"] = (TraceStatus.failed if failed else TraceStatus.completed).value
        fields["ended_at"] = ended
        fields["duration_ms"] = duration_ms(started, ended)
        if started is not None:
            fields["started_at"] = started
    await Trace.get_motor_collection().update_one({"_id": trace.id}, {"$set": fields})
    return spans, content_changed


def _trace_failed(spans: list, root: Optional[OtlpSpan]) -> bool:
    """Failed when the root span failed, or when the run ended on a failed model call
    (Claude Code has no real root span: its api_error is the only failure signal).
    Earlier failures the agent recovered from don't fail the trace."""
    if root is not None and root.status_code == 2:
        return True
    llm = [s for s in spans if s.span_type == SpanType.llm]
    if not llm:
        return False
    last = max(llm, key=lambda s: (s.timestamp, s.step_index))
    return last.status == "error" or bool(last.error)


async def _evaluate(span_ids: list, trace_id: Optional[PydanticObjectId]) -> None:
    results = await asyncio.gather(
        *(bounded(evaluate_span(sid)) for sid in span_ids), return_exceptions=True
    )
    for sid, result in zip(span_ids, results):
        if isinstance(result, Exception):
            log.error("Evaluation of span %s failed: %r", sid, result)
    if trace_id is not None:
        await bounded(evaluate_trace(trace_id))


async def finalize_trace(trace: Trace, root: Optional[OtlpSpan] = None) -> None:
    """Close a running OTel trace, refresh its aggregates and schedule its evaluation."""
    claim = await Trace.get_motor_collection().update_one(
        {"_id": trace.id, "status": TraceStatus.running.value},
        {"$set": {"status": TraceStatus.completed.value}},
    )
    if claim.modified_count == 0:
        return   # already finalized by another batch, worker or the sweeper
    spans, _ = await _consolidate(trace, final=True, root=root)
    if (trace.metadata or {}).get("source") == "claude-code":
        try:
            await claude_code_answer.apply_pending(trace, spans)
        except Exception:
            log.exception("Could not apply the Claude Code answer to trace %s", trace.id)
    status = TraceStatus.failed if _trace_failed(spans, root) else TraceStatus.completed
    publish(trace.project_id, "trace.completed", {"trace_id": str(trace.id), "status": status.value})
    schedule(_evaluate([s.id for s in spans if is_evaluable(s)], trace.id))


async def _ingest_trace_group(project: Project, otel_trace_id: str, group: list) -> tuple:
    kept = [(s, m) for s, m in ((s, map_span(s)) for s in group) if m is not None]
    root = next((s for s in group if s.is_root), None)

    trace = await _find_trace(project.id, otel_trace_id)
    if trace is None:
        if not kept:
            return 0, len(group)   # no GenAI activity in this trace (yet): nothing to record
        starts = [s.start_ns for s in group if s.start_ns]
        first_name = min(kept, key=lambda pair: pair[0].start_ns)[0].name
        trace = await _create_trace(
            project.id,
            otel_trace_id,
            name=(root.name if root is not None else first_name) or "otel-trace",
            started_at=_dt(min(starts)) if starts else None,
            metadata=_trace_metadata(kept),
        )

    docs, new_ids = await _store_spans(project.id, trace, kept) if kept else ([], set())
    await _touch_trace(trace, group, root, [d for d in docs if d.id in new_ids])
    if new_ids:
        publish(project.id, "span.created", {"trace_id": str(trace.id), "count": len(new_ids)})

    if trace.status == TraceStatus.running:
        if root is not None:
            await finalize_trace(trace, root)
    elif new_ids:
        # Late spans of a finalized trace: count them and extend its duration, evaluate
        # them, but not the trace again.
        spans, _ = await _consolidate(trace, final=True)
        late = [s.id for s in spans if s.id in new_ids and is_evaluable(s)]
        if late:
            schedule(_evaluate(late, None))
    return len(kept), len(group) - len(kept)


async def ingest_spans(project: Project, spans: list) -> IngestResult:
    """Store the GenAI spans of an OTLP export. Database outages propagate (the caller
    answers 503 so the exporter retries); any other per-trace failure is reported as
    rejected spans so the rest of the batch still lands."""
    result = IngestResult()
    by_trace: dict = defaultdict(list)
    for span in spans:
        by_trace[span.trace_id].append(span)
    for otel_trace_id, group in by_trace.items():
        try:
            stored, skipped = await _ingest_trace_group(project, otel_trace_id, group)
        except ConnectionFailure:
            raise
        except Exception:
            log.exception("Could not ingest OTel trace %s", otel_trace_id)
            result.rejected += len(group)
            continue
        result.stored += stored
        result.skipped += skipped
        result.traces += 1 if stored else 0
    return result


async def _ingest_claude_code(project: Project, records: list) -> int:
    spans = claude_code_spans(records)
    if not spans:
        return 0
    await ingest_spans(project, spans)
    await Trace.get_motor_collection().update_many(
        {"project_id": project.id, "external_id": {"$in": list({s.trace_id for s in spans})}},
        [{"$set": {"metadata": {"$mergeObjects": [{"$ifNull": ["$metadata", {}]}, {"source": "claude-code"}]}}}],
    )
    for span in spans:
        if span.attributes.get(CLAUDE_TRACE_NAME):
            await Trace.get_motor_collection().update_one(
                {"project_id": project.id, "external_id": span.trace_id},
                {"$set": {"name": span.attributes[CLAUDE_TRACE_NAME]}},
            )
    return len(spans)


async def ingest_log_records(project: Project, records: list) -> tuple:
    """Buffer GenAI content events linked to a span, and turn Claude Code events into
    traces. Returns (accepted, ignored)."""
    claude_code = [r for r in records if is_claude_code(r)]
    accepted_cc = await _ingest_claude_code(project, claude_code) if claude_code else 0
    records = [r for r in records if not is_claude_code(r)]

    docs = [
        OtelEvent(
            project_id=project.id,
            otel_trace_id=r.trace_id,
            otel_span_id=r.span_id,
            name=r.event_name,
            time_ns=r.time_ns,
            data=json.dumps({"attributes": r.attributes, "body": r.body}, default=str),
        )
        for r in records
        if r.trace_id and r.span_id and is_content_event(r.event_name)
    ]
    if docs:
        await OtelEvent.insert_many(docs)

    for otel_trace_id in {d.otel_trace_id for d in docs}:
        trace = await _find_trace(project.id, otel_trace_id)
        if trace is not None and trace.status != TraceStatus.running:
            # Already finalized: merge now and evaluate the spans that gained content.
            spans, changed = await _consolidate(trace, final=False)
            ids = [s.id for s in spans if s.id in changed and is_evaluable(s)]
            if ids:
                schedule(_evaluate(ids, None))
    return len(docs) + accepted_cc, len(records) - len(docs) + len(claude_code) - accepted_cc


async def sweep_idle_traces(idle_seconds: float = IDLE_SECONDS,
                            claude_code_idle_seconds: float = CLAUDE_CODE_IDLE_SECONDS) -> int:
    """Finalize running OTel traces that received no span for `idle_seconds`
    (`claude_code_idle_seconds` for Claude Code runs)."""
    now = datetime.utcnow()
    stale = await Trace.find({
        "source": "otel",
        "status": TraceStatus.running.value,
        "$or": [
            {"metadata.source": {"$ne": "claude-code"},
             "last_activity_at": {"$lt": now - timedelta(seconds=idle_seconds)}},
            {"metadata.source": "claude-code",
             "last_activity_at": {"$lt": now - timedelta(seconds=max(idle_seconds, claude_code_idle_seconds))}},
        ],
    }).to_list()
    for trace in stale:
        try:
            await finalize_trace(trace)
        except Exception:
            log.exception("Could not finalize idle OTel trace %s", trace.id)
    return len(stale)


async def run_sweeper(idle_seconds: float = IDLE_SECONDS) -> None:
    """Background loop finalizing OTel traces whose root span never arrived."""
    interval = max(5.0, min(30.0, idle_seconds / 2))
    while True:
        await asyncio.sleep(interval)
        try:
            count = await sweep_idle_traces(idle_seconds)
            if count:
                log.info("Finalized %d idle OpenTelemetry trace(s)", count)
        except Exception:
            log.exception("OpenTelemetry trace sweeper failed")
