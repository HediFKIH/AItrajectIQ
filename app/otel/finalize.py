"""
Pure helpers used when an OpenTelemetry trace is finalized.

They work on any object exposing the Span attributes they read (Beanie documents in
production, SimpleNamespace in tests) and never touch the database.
"""

from datetime import datetime, timedelta
from typing import Iterable, Optional

from otel.mapping import messages_from_events


def span_type(span) -> str:
    return getattr(span.span_type, "value", span.span_type)


def nested_llm_span_ids(spans: list) -> set:
    """`sdk_span_id`s of LLM spans that have another LLM span somewhere below them.

    Two instrumentations often report the same call, e.g. a LangChain chat-model span
    wrapping the OpenAI SDK span. Only the innermost one should count tokens and cost.
    """
    by_id = {s.sdk_span_id: s for s in spans if s.sdk_span_id}
    outer = set()
    for span in spans:
        if span_type(span) != "llm":
            continue
        seen = set()
        parent_id = span.parent_span_id
        while parent_id and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id[parent_id]
            if span_type(parent) == "llm":
                outer.add(parent_id)
            parent_id = parent.parent_span_id
    return outer


def ordered(spans: Iterable) -> list:
    """Spans sorted by start time (ties keep their current step order)."""
    return sorted(spans, key=lambda s: (s.timestamp or datetime.min, s.step_index or 0))


def trace_totals(spans: Iterable) -> dict:
    spans = list(spans)
    llm = [s for s in spans if span_type(s) == "llm"]
    return {
        "total_spans": len(spans),
        "total_llm_spans": len(llm),
        "total_tool_spans": sum(1 for s in spans if span_type(s) == "tool"),
        "total_tokens_in": sum(s.tokens_in or 0 for s in llm),
        "total_tokens_out": sum(s.tokens_out or 0 for s in llm),
        "total_cost_usd": round(sum(s.cost_usd or 0.0 for s in llm), 6),
    }


def time_bounds(spans: Iterable) -> tuple:
    """(earliest start, latest end) of the spans, or (None, None)."""
    starts, ends = [], []
    for s in spans:
        if s.timestamp is None:
            continue
        starts.append(s.timestamp)
        ends.append(s.timestamp + timedelta(milliseconds=s.latency_ms or 0))
    return (min(starts), max(ends)) if starts else (None, None)


def apply_content_events(span, events: list) -> bool:
    """Fill a content-less LLM span from its GenAI content events. Returns True if changed."""
    if span_type(span) != "llm" or not events:
        return False
    messages, response = messages_from_events(events)
    changed = False
    if messages and not span.messages_in:
        span.messages_in = messages
        changed = True
    if response and not span.response:
        span.response = response
        changed = True
    return changed


def is_evaluable(span) -> bool:
    """LLM spans are evaluated only when content was captured: judging empty text is noise."""
    return span_type(span) == "llm" and bool(span.response or span.messages_in)


def duration_ms(started: Optional[datetime], ended: Optional[datetime]) -> Optional[int]:
    if started is None or ended is None:
        return None
    return max(0, int((ended - started).total_seconds() * 1000))
