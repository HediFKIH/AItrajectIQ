from datetime import datetime, timedelta
from types import SimpleNamespace

from otel.finalize import (
    apply_content_events,
    duration_ms,
    is_evaluable,
    nested_llm_span_ids,
    ordered,
    time_bounds,
    trace_totals,
)
from services.insights import span_depths

T0 = datetime(2026, 1, 1, 12, 0, 0)


def span(sid, parent=None, kind="llm", start=0, latency=100, tokens_in=10, tokens_out=5,
         cost=0.01, step=0, messages=None, response=None):
    return SimpleNamespace(
        sdk_span_id=sid,
        parent_span_id=parent,
        span_type=kind,
        timestamp=T0 + timedelta(milliseconds=start),
        latency_ms=latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        step_index=step,
        messages_in=messages,
        response=response,
    )


def test_only_the_innermost_of_nested_llm_spans_counts():
    spans = [
        span("agent", kind="agent"),
        span("langchain", parent="agent"),
        span("openai", parent="langchain"),
        span("tool", parent="agent", kind="tool"),
        span("other-llm", parent="agent"),
    ]

    assert nested_llm_span_ids(spans) == {"langchain"}


def test_nested_detection_terminates_on_cycles():
    assert nested_llm_span_ids([span("a", parent="b"), span("b", parent="a")]) == {"a", "b"}


def test_ordered_by_start_time_and_totals():
    spans = [
        span("late", start=500, step=0),
        span("early", start=0, step=1),
        span("tool", kind="tool", start=200, tokens_in=None, tokens_out=None, cost=None),
    ]

    assert [s.sdk_span_id for s in ordered(spans)] == ["early", "tool", "late"]
    assert trace_totals(spans) == {
        "total_spans": 3,
        "total_llm_spans": 2,
        "total_tool_spans": 1,
        "total_tokens_in": 20,
        "total_tokens_out": 10,
        "total_cost_usd": 0.02,
    }


def test_time_bounds_and_duration():
    started, ended = time_bounds([span("a", start=0, latency=100), span("b", start=50, latency=500)])

    assert started == T0
    assert ended == T0 + timedelta(milliseconds=550)
    assert duration_ms(started, ended) == 550
    assert time_bounds([]) == (None, None)
    assert duration_ms(None, ended) is None


def test_content_events_only_fill_missing_content():
    events = [
        {"name": "gen_ai.user.message", "time_ns": 1, "attributes": {}, "body": {"content": "Hi"}},
        {"name": "gen_ai.choice", "time_ns": 2, "attributes": {}, "body": {"message": {"content": "Hello!"}}},
    ]

    empty = span("x")
    assert apply_content_events(empty, events) is True
    assert empty.messages_in == [{"role": "user", "content": "Hi"}]
    assert empty.response == "Hello!"

    filled = span("y", messages=[{"role": "user", "content": "kept"}], response="kept")
    assert apply_content_events(filled, events) is False
    assert filled.response == "kept"


def test_only_llm_spans_with_content_are_evaluated():
    assert is_evaluable(span("a", response="text"))
    assert is_evaluable(span("b", messages=[{"role": "user", "content": "q"}]))
    assert not is_evaluable(span("c"))
    assert not is_evaluable(span("d", kind="tool", response="r"))


def test_span_depths_treat_unstored_parents_as_roots():
    spans = [
        SimpleNamespace(sdk_span_id="root", parent_span_id="http-span-not-stored"),
        SimpleNamespace(sdk_span_id="child", parent_span_id="root"),
        SimpleNamespace(sdk_span_id="grandchild", parent_span_id="child"),
    ]

    assert span_depths(spans) == {"root": 0, "child": 1, "grandchild": 2}
