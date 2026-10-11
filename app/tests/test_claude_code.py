from otel.claude_code import TRACE_NAME, claude_code_spans, is_claude_code
from otel.decode import OtlpLogRecord
from otel.mapping import map_span

T0 = 1_700_000_000_000_000_000
RESOURCE = {"service.name": "claude-code"}


def record(event, attributes, time_ns=T0):
    return OtlpLogRecord(
        trace_id=None,
        span_id=None,
        time_ns=time_ns,
        event_name=event,
        body=f"claude_code.{event}",
        attributes={"session.id": "sess-1", "prompt.id": "prompt-1", "event.name": event, **attributes},
        resource=RESOURCE,
    )


def test_detects_claude_code_events_only():
    assert is_claude_code(record("api_request", {}))
    other = OtlpLogRecord(None, None, T0, "gen_ai.choice", None, {}, {"service.name": "svc"})
    assert not is_claude_code(other)


def test_prompt_becomes_agent_span_parent_of_its_calls():
    spans = claude_code_spans([
        record("user_prompt", {"prompt": "Fix the   login bug"}),
        record("api_request", {
            "model": "claude-sonnet-4-5", "input_tokens": "100", "cache_read_tokens": "900",
            "output_tokens": "50", "cost_usd": "0.012", "duration_ms": "2000",
        }, time_ns=T0 + 3_000_000_000),
        record("tool_result", {
            "tool_name": "Bash", "success": "false", "error": "exit 1",
            "tool_parameters": '{"command": "pytest"}', "duration_ms": "500",
        }, time_ns=T0 + 4_000_000_000),
    ])
    agent, llm, tool = spans
    assert {s.trace_id for s in spans} == {agent.trace_id}
    assert not any(s.is_root for s in spans)        # finalized by the idle sweeper, not the first batch
    assert llm.parent_span_id == agent.span_id and tool.parent_span_id == agent.span_id
    assert agent.name == "Claude Code"
    assert agent.attributes[TRACE_NAME] == "Claude Code: Fix the login bug"
    assert map_span(tool).metadata["tags"]["tool_name"] == "Bash"      # raw attributes kept
    assert "prompt" not in map_span(agent).metadata.get("tags", {})

    mapped = map_span(llm)
    assert mapped.span_type == "llm" and mapped.model == "claude-sonnet-4-5"
    assert mapped.tokens_in == 1000 and mapped.tokens_out == 50 and mapped.cost_usd == 0.012
    assert mapped.cache_read_tokens == 900 and mapped.cache_write_tokens is None
    assert llm.end_ns - llm.start_ns == 2_000_000_000

    mapped = map_span(tool)
    assert mapped.span_type == "tool" and mapped.tool_name == "Bash"
    assert mapped.tool_input == {"command": "pytest"}
    assert mapped.status == "error" and mapped.tool_error == "exit 1"

    assert map_span(agent).span_type == "agent"


def test_subagent_launch_is_an_agent_named_after_its_task():
    (span,) = claude_code_spans([record("tool_result", {
        "tool_name": "Agent", "tool_parameters": '{"description": "Audit accessibilité", "prompt": "..."}',
    })])
    mapped = map_span(span)
    assert mapped.span_type == "agent" and mapped.agent_name == "Audit accessibilité"


def test_events_naming_their_subagent_are_grouped_under_it():
    spans = claude_code_spans([
        record("api_request", {"agent_id": "a1", "agent_type": "general-purpose"}, time_ns=T0),
        record("tool_result", {"agent_id": "a1", "tool_name": "Read"}, time_ns=T0 + 1),
        record("api_request", {"agent_id": "a2"}, time_ns=T0 + 2),
        record("api_request", {}, time_ns=T0 + 3),
    ])
    agents = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
    assert [a.name for a in agents] == ["general-purpose", "Sous-agent a2"]
    a1, a2 = agents
    calls = [s for s in spans if s not in agents]
    assert [c.parent_span_id for c in calls[:3]] == [a1.span_id, a1.span_id, a2.span_id]
    assert calls[3].parent_span_id == a1.parent_span_id                # main agent


def test_ids_are_stable_across_exporter_retries():
    first = claude_code_spans([record("api_request", {"model": "m"})])
    again = claude_code_spans([record("api_request", {"model": "m"})])
    assert first[0].span_id == again[0].span_id


def test_prompts_get_separate_traces():
    a = claude_code_spans([record("api_request", {"prompt.id": "p1"})])[0]
    b = claude_code_spans([record("api_request", {"prompt.id": "p2"})])[0]
    assert a.trace_id != b.trace_id
