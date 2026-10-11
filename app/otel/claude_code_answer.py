"""
Final answer and tool arguments of Claude Code runs, sent by its Stop hook.

Claude Code's telemetry carries tokens, costs and tool names, but neither the agent's answer
nor (by default) the file a tool read: the answer-based metrics (task completion, grounded
claims, instruction adherence) had nothing to judge, and seven reads of seven files looked
like one action repeated seven times. The hook (sdk/aitrajectiq/integrations/claude_code_hook.py)
reads the session transcript when Claude stops and posts the prompt, the final answer and
the tool inputs of the turn.

The payload is kept as a pending event (24 h TTL, see OtelEvent) and applied to the trace of
that prompt: right away when the trace exists, and again when it is finalized, once every
span arrived. The trace of a prompt is the Claude Code trace of the session that started
within MATCH_SLACK of the prompt. The Stop hook also tells when the run is over, so the
trace is finalized FINALIZE_DELAY_S after it (the last telemetry batch may still be on its
way) instead of waiting for the idle sweeper.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from pymongo import UpdateOne

from models.otel_event import OtelEvent
from models.span import Span, SpanType
from models.trace import Trace, TraceStatus

log = logging.getLogger("aitrajectiq.otel")

STOP_EVENT = "aitrajectiq.claude_code.stop"
MATCH_SLACK = timedelta(seconds=15)
FINALIZE_DELAY_S = 15.0
MAX_TEXT = 20_000
MAX_ARG = 300
MAX_TOOLS = 300
ANSWER_FLAG = "claude_code_answer"
_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})


def pending_key(session_id: str) -> str:
    return f"claude-code-stop:{session_id}"


def naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_time(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return naive_utc(value)
    if isinstance(value, str) and value:
        try:
            return naive_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def compact_input(value) -> dict:
    """Tool arguments with long values cut: enough to tell which file or command, without
    storing a whole file written by the agent."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if isinstance(item, str):
            out[key] = item if len(item) <= MAX_ARG else item[:MAX_ARG] + "…"
        elif item is None or isinstance(item, (bool, int, float)):
            out[key] = item
        else:
            out[key] = json.dumps(item, ensure_ascii=False, default=str)[:MAX_ARG]
    return out


def match_tool_inputs(tool_spans: list, tools: list) -> dict:
    """{span id: arguments}, pairing the run's tool spans (in execution order) with the
    transcript's tool calls of the same name, in order. Subagent launches are agent spans."""
    calls = [(t.get("name"), t.get("input")) for t in tools
             if isinstance(t, dict) and t.get("name") not in _SUBAGENT_TOOLS]
    out, start = {}, 0
    for span in tool_spans:
        name = span.tool_name or span.name
        for index in range(start, len(calls)):
            if calls[index][0] == name:
                out[span.id] = compact_input(calls[index][1])
                start = index + 1
                break
    return out


def answer_span(spans: list):
    """The main agent's last model call: the one that wrote the final answer (subagents'
    calls and failed calls excluded when possible)."""
    llm = [s for s in spans if s.span_type == SpanType.llm]
    if not llm:
        return None
    agents = {s.sdk_span_id: s for s in spans if s.span_type == SpanType.agent and s.sdk_span_id}
    nested = {sid for sid, a in agents.items() if a.parent_span_id in agents}
    top = [s for s in llm if s.parent_span_id not in nested] or llm
    ok = [s for s in top if not s.error and s.status != "error"] or top
    return max(ok, key=lambda s: (s.timestamp, s.step_index))


def matches(trace, stop: dict) -> bool:
    prompt_at = _parse_time(stop.get("prompt_at"))
    return prompt_at is None or abs(trace.started_at - prompt_at) <= MATCH_SLACK


def stop_updates(trace, spans: list, stop: dict) -> dict:
    """{span id: fields to $set} that put the answer, the prompt and the tool arguments on
    the trace's spans. Pure: the caller writes them."""
    updates: dict = {}
    target = answer_span(spans)
    answer = (stop.get("answer") or "").strip()[:MAX_TEXT]
    prompt = (stop.get("prompt") or "").strip()[:MAX_TEXT]
    if target is not None and answer:
        for span in spans:
            if span is not target and (span.metadata or {}).get(ANSWER_FLAG):
                updates[span.id] = {"response": None, "messages_in": None,
                                    "metadata": {k: v for k, v in span.metadata.items() if k != ANSWER_FLAG}}
        fields = {"response": answer, "metadata": {**(target.metadata or {}), ANSWER_FLAG: True}}
        if prompt:
            fields["messages_in"] = [{"role": "user", "content": prompt}]
        updates[target.id] = fields
    tools = sorted((s for s in spans if s.span_type == SpanType.tool), key=lambda s: (s.timestamp, s.step_index))
    for span_id, arguments in match_tool_inputs(tools, (stop.get("tools") or [])[:MAX_TOOLS]).items():
        span = next(s for s in tools if s.id == span_id)
        if arguments and not span.tool_input:
            updates.setdefault(span_id, {})["tool_input"] = arguments
    return updates


async def _apply(trace, spans: list, stop: dict) -> bool:
    updates = stop_updates(trace, spans, stop)
    if updates:
        await Span.get_motor_collection().bulk_write(
            [UpdateOne({"_id": sid}, {"$set": fields}) for sid, fields in updates.items()], ordered=False,
        )
    session_id = stop.get("session_id")
    if session_id:
        # Each prompt is its own trace: the session ties them as one conversation (earlier turns).
        await Trace.get_motor_collection().update_one(
            {"_id": trace.id},
            [{"$set": {"metadata": {"$mergeObjects": [{"$ifNull": ["$metadata", {}]},
                                                      {"conversation_id": session_id}]}}}],
        )
    return bool(updates)


async def _matching_trace(project_id, stop: dict) -> Optional[Trace]:
    query: dict = {"project_id": project_id, "metadata.source": "claude-code",
                   "metadata.session_id": stop["session_id"]}
    prompt_at = _parse_time(stop.get("prompt_at"))
    if prompt_at is not None:
        query["started_at"] = {"$gte": prompt_at - MATCH_SLACK, "$lte": prompt_at + MATCH_SLACK}
    return await Trace.find(query).sort("-started_at").first_or_none()


async def _finalize_later(trace_id) -> None:
    from otel.ingest import finalize_trace

    await asyncio.sleep(FINALIZE_DELAY_S)
    trace = await Trace.get(trace_id)
    if trace is not None and trace.status == TraceStatus.running:
        await finalize_trace(trace)


async def record_stop(project, stop: dict) -> str:
    """Keep the hook's payload and apply it to its trace if it exists. Returns "attached"
    or "pending" (the trace will pick it up when finalized)."""
    from evaluation.runner import evaluate_trace
    from evaluation.scheduler import bounded, schedule

    await OtelEvent(
        project_id=project.id,
        otel_trace_id=pending_key(stop["session_id"]),
        otel_span_id="stop",
        name=STOP_EVENT,
        data=json.dumps(stop, default=str),
    ).insert()
    trace = await _matching_trace(project.id, stop)
    if trace is None:
        return "pending"
    spans = await Span.find(Span.trace_id == trace.id).to_list()
    changed = await _apply(trace, spans, stop)
    if trace.status == TraceStatus.running:
        schedule(_finalize_later(trace.id))
    elif changed:
        schedule(bounded(evaluate_trace(trace.id)))
    return "attached"


async def apply_pending(trace, spans: list) -> bool:
    """At finalization: apply the latest Stop payload of this prompt, if the hook sent one."""
    session_id = (trace.metadata or {}).get("session_id")
    if not session_id:
        return False
    events = await OtelEvent.find({
        "project_id": trace.project_id, "otel_trace_id": pending_key(session_id), "name": STOP_EVENT,
    }).sort("-created_at").to_list()
    for event in events:
        try:
            stop = json.loads(event.data or "{}")
        except ValueError:
            continue
        if matches(trace, stop):
            return await _apply(trace, spans, stop)
    return False
