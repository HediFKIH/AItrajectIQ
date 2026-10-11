"""
Claude Code telemetry → AItrajectIQ traces.

Claude Code exports OpenTelemetry *log events* without trace or span ids
(claude_code.user_prompt, claude_code.api_request, claude_code.api_error,
claude_code.tool_result...). They are turned into synthetic OTLP spans so they go
through the regular OTel ingestion:

    one trace per prompt (`prompt.id`, or per `session.id` on versions without it)
    └─ agent span "Claude Code"               (user_prompt)
       ├─ llm span  <model>                   (api_request / api_error)
       ├─ tool span <tool>                    (tool_result)
       ├─ agent span <task description>       (tool_result of the Agent/Task tool)
       └─ agent span <subagent>               (events carrying an agent id, if any)

Enable it in Claude Code with:

    CLAUDE_CODE_ENABLE_TELEMETRY=1
    OTEL_LOGS_EXPORTER=otlp
    OTEL_EXPORTER_OTLP_PROTOCOL=http/json
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8000/otel
    OTEL_EXPORTER_OTLP_HEADERS=x-api-key=atq_...
    OTEL_LOG_USER_PROMPTS=1            (optional: prompt text in the trace name)

No span is a root: the telemetry never says when a prompt is done. The Stop hook does
(otel/claude_code_answer.py, which also brings the final answer and the tool arguments);
without it, traces are finalized by the idle sweeper (AITRAJECTIQ_CLAUDE_CODE_IDLE_S).
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Optional

from otel.decode import OtlpLogRecord, OtlpSpan

_PREFIX = "claude_code."
_SCOPE = "claude-code"
_PROMPT_PREVIEW = 80

TRACE_NAME = "aitrajectiq.trace_name"
RAW_PREFIX = "aitrajectiq.metadata."
_PRIVATE_KEYS = frozenset({"prompt", "user.email", "user.account_uuid", "user.id", "organization.id"})
_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})
# Attributes that tie an event to the subagent that emitted it, when the Claude Code
# version sends one. Without them, a subagent's calls stay under the main agent.
_AGENT_ID_KEYS = ("agent.id", "agent_id", "agentId", "subagent.id", "subagent_id")
_AGENT_NAME_KEYS = ("agent.name", "agent_name", "agent.type", "agent_type", "subagent_type", "subagent.type")


def _event(record: OtlpLogRecord) -> Optional[str]:
    """Short event name ("api_request") of a Claude Code record, else None."""
    for candidate in (record.body, record.event_name):
        if isinstance(candidate, str) and candidate.startswith(_PREFIX):
            return candidate[len(_PREFIX):]
    if record.resource.get("service.name") == _SCOPE and record.event_name:
        return str(record.event_name)
    return None


def is_claude_code(record: OtlpLogRecord) -> bool:
    return _event(record) is not None


def _hex(*parts: Any, length: int) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:length]


def _int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _span(trace_id, span_id, parent, name, start_ns, end_ns, attrs, record, error=None) -> OtlpSpan:
    return OtlpSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        kind=0,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={k: v for k, v in attrs.items() if v is not None},
        events=[],
        status_code=2 if error else 0,
        status_message=error,
        resource=record.resource,
        scope_name=_SCOPE,
    )


def _time_ns(record: OtlpLogRecord) -> int:
    if record.time_ns:
        return record.time_ns
    stamp = record.attributes.get("event.timestamp")
    if isinstance(stamp, str):
        try:
            return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1e9)
        except ValueError:
            pass
    return time.time_ns()


def _first(attrs: dict, keys: tuple) -> Optional[str]:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _raw_metadata(attrs: dict) -> dict:
    """The event's own attributes, kept as span tags (see mapping._TAG_PREFIXES)."""
    return {
        RAW_PREFIX + key.replace(".", "_"): value
        for key, value in attrs.items()
        if key not in _PRIVATE_KEYS and not isinstance(value, (dict, list))
    }


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _to_span(record: OtlpLogRecord, event: str) -> Optional[OtlpSpan]:
    a = record.attributes
    session = a.get("session.id") or "no-session"
    prompt = a.get("prompt.id")
    group = prompt or session
    trace_id = _hex("claude-code", session, group, length=32)
    session_span = _hex("session", session, length=16)
    main_agent = _hex("agent", group, length=16)
    parent = main_agent if prompt else session_span
    subagent = _first(a, _AGENT_ID_KEYS)
    if subagent:
        parent = _hex("subagent", group, subagent, length=16)
    end_ns = _time_ns(record)
    span_id = _hex(trace_id, event, end_ns, a.get("event.sequence"), length=16)

    duration = _int(a.get("duration_ms")) or 0
    start_ns = max(0, end_ns - duration * 1_000_000)
    common = {"session.id": a.get("session.id"), "claude_code.prompt_id": prompt, **_raw_metadata(a)}

    if event == "user_prompt":
        text = a.get("prompt")
        trace_name = "Claude Code"
        if isinstance(text, str) and text.strip():
            preview = " ".join(text.split())
            trace_name += ": " + (preview[:_PROMPT_PREVIEW] + "…" if len(preview) > _PROMPT_PREVIEW else preview)
        return _span(
            trace_id, main_agent if prompt else span_id, session_span, "Claude Code", end_ns, end_ns,
            {**common, "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Claude Code",
             "gen_ai.agent.description": text if isinstance(text, str) else None,
             TRACE_NAME: trace_name},
            record,
        )

    if event in ("api_request", "api_error"):
        model = a.get("model")
        tokens_in = sum(
            _int(a.get(key)) or 0 for key in ("input_tokens", "cache_read_tokens", "cache_creation_tokens")
        )
        error = None
        if event == "api_error":
            error = str(a.get("error") or "API error")
            if a.get("status_code"):
                error = f"{a['status_code']}: {error}"
        return _span(
            trace_id, span_id, parent, str(model or "claude"), start_ns, end_ns,
            {**common,
             "gen_ai.operation.name": "chat",
             "gen_ai.provider.name": "anthropic",
             "gen_ai.request.model": model,
             "gen_ai.usage.input_tokens": tokens_in if event == "api_request" else None,
             "gen_ai.usage.cache_read.input_tokens": _int(a.get("cache_read_tokens")),
             "gen_ai.usage.cache_creation.input_tokens": _int(a.get("cache_creation_tokens")),
             "gen_ai.usage.output_tokens": _int(a.get("output_tokens")),
             "gen_ai.usage.cost": _float(a.get("cost_usd"))},
            record,
            error,
        )

    if event == "tool_result":
        tool = str(a.get("tool_name") or "tool")
        failed = str(a.get("success", "true")).lower() == "false"
        error = str(a.get("error") or "tool failed") if failed else None
        arguments = a.get("tool_parameters")
        if tool in _SUBAGENT_TOOLS:
            # A subagent launch: shown as an agent named after its task, not as a tool call.
            params = _json_dict(arguments)
            name = _first(params, ("description", "subagent_type", "name")) or "Sous-agent"
            return _span(
                trace_id, span_id, parent, name, start_ns, end_ns,
                {**common, "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name,
                 "gen_ai.agent.description": _first(params, ("prompt",))},
                record,
                error,
            )
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        return _span(
            trace_id, span_id, parent, tool, start_ns, end_ns,
            {**common,
             "gen_ai.operation.name": "execute_tool",
             "gen_ai.tool.name": tool,
             "gen_ai.tool.call.arguments": arguments},
            record,
            error,
        )
    return None


def _subagent_span(record: OtlpLogRecord, child: OtlpSpan) -> OtlpSpan:
    """Agent span grouping the calls of one subagent, for events that name their agent."""
    a = record.attributes
    agent_id = _first(a, _AGENT_ID_KEYS)
    name = _first(a, _AGENT_NAME_KEYS) or f"Sous-agent {agent_id[:8]}"
    prompt = a.get("prompt.id")
    group = prompt or a.get("session.id") or "no-session"
    parent = _hex("agent", group, length=16) if prompt else _hex("session", a.get("session.id") or "no-session", length=16)
    return _span(
        child.trace_id, child.parent_span_id, parent, name, child.start_ns, child.end_ns,
        {"session.id": a.get("session.id"), "gen_ai.operation.name": "invoke_agent",
         "gen_ai.agent.name": name, RAW_PREFIX + "agent_id": agent_id},
        record,
    )


def claude_code_spans(records: list) -> list:
    """Synthetic spans for the Claude Code events among `records` (other events are skipped)."""
    spans = []
    subagents: dict = {}
    for record in records:
        event = _event(record)
        if event is None:
            continue
        span = _to_span(record, event)
        if span is None:
            continue
        spans.append(span)
        if _first(record.attributes, _AGENT_ID_KEYS):
            agent = subagents.get(span.parent_span_id)
            if agent is None:
                subagents[span.parent_span_id] = _subagent_span(record, span)
            else:
                agent.start_ns = min(agent.start_ns, span.start_ns)
                agent.end_ns = max(agent.end_ns, span.end_ns)
    return list(subagents.values()) + spans
