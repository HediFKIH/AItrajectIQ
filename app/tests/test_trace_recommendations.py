from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from services.recommendations import COMPLEX, MEDIUM, SIMPLE, analyze_trace, cheaper_model, run_complexity
from utils.cost import compute_cost, set_web_prices

T0 = datetime(2026, 1, 1)


def span(i, kind="llm", **kw):
    fields = dict(
        id=f"s{i}", trace_id="t1", step_index=i, span_type=kind, name=None, status="completed", error=None,
        metadata=None, model=None, messages_in=None, tokens_in=None, tokens_out=None, cache_read_tokens=None,
        cache_write_tokens=None, cost_usd=None, temperature=None, tool_name=None, tool_input=None,
        tool_error=None, tool_output=None, timestamp=T0 + timedelta(seconds=i), metrics=[], parent_span_id=None,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def llm(i, model, tokens_in, tokens_out, **kw):
    return span(i, "llm", model=model, tokens_in=tokens_in, tokens_out=tokens_out, **kw)


def tool(i, name, args, **kw):
    return span(i, "tool", tool_name=name, tool_input=args, **kw)


def trace(source=None, status="completed", metrics=()):
    return SimpleNamespace(id="t1", metadata={"source": source} if source else {}, status=status, metrics=list(metrics))


def by_id(result):
    return {r["id"]: r for r in result["recommendations"]}


@pytest.mark.parametrize("tools, context, expected", [
    (0, 5_000, SIMPLE),
    (3, 20_000, SIMPLE),
    (4, 5_000, MEDIUM),
    (2, 30_000, MEDIUM),
    (15, 5_000, COMPLEX),
    (1, 100_000, COMPLEX),
])
def test_run_complexity(tools, context, expected):
    assert run_complexity(tools, context) == expected


def test_short_agent_run_on_a_premium_model_suggests_the_tier_below():
    spans = [
        tool(0, "Read", {"path": "a.py"}),
        tool(1, "Grep", {"pattern": "foo"}),
        *(llm(i, "claude-opus-4-5", 5_000, 300, cache_read_tokens=4_000) for i in range(2, 5)),
    ]
    result = analyze_trace(trace("cursor"), spans)

    assert result["agentic"] and result["complexity"] == SIMPLE
    rec = by_id(result)["trace-model:claude-opus-4.5"]
    per_call = (compute_cost("claude-opus-4.5", 5_000, 300, 4_000)
                - compute_cost("claude-sonnet-4.6", 5_000, 300, 4_000))
    assert rec["savings_usd"] == pytest.approx(3 * per_call, abs=1e-6)
    assert rec["evidence"]["alternative"] == "claude-sonnet-4.6"
    assert "Cursor" in rec["action"]
    assert [r["span_id"] for r in rec["spans"]] == ["s2", "s3", "s4"]
    assert all(s["alternative"] == "claude-sonnet-4.6" for s in result["steps"])
    assert result["optimized_cost_usd"] < result["cost_usd"]
    assert "claude-sonnet-4.6 would have been enough" in result["span_notes"]["s2"][0]
    assert not any(r["category"] == "caching" for r in result["recommendations"])


def test_long_agent_run_keeps_its_model():
    spans = [tool(i, "Read", {"path": f"f{i}.py"}) for i in range(20)]
    spans += [llm(20 + i, "claude-opus-4-5", 5_000, 300, cache_read_tokens=4_000) for i in range(3)]
    result = analyze_trace(trace("cursor"), spans)
    assert result["complexity"] == COMPLEX
    assert not any(r["category"] == "model" for r in result["recommendations"])
    assert all(s["alternative"] is None for s in result["steps"])
    assert result["steps"][0]["no_alternative_reason"].startswith("Task too long for a smaller model: 20 tool(s)")


def test_reason_when_no_cheaper_model():
    spans = [llm(0, "gpt-4o-mini", 1_000, 100), llm(1, "my-local-model", 1_000, 100),
             llm(2, "gpt-4o", 1_000, 100, status="failed", error="429"), llm(3, "gpt-4o", 9_000, 100)]
    steps = analyze_trace(trace(), spans)["steps"]

    assert steps[0]["no_alternative_reason"] == "No smaller model known from this provider."
    assert steps[1]["no_alternative_reason"].startswith("Unknown price")
    assert steps[2]["no_alternative_reason"] == "Failed call."
    assert steps[3]["no_alternative_reason"].startswith("Medium request: 9,000 input tokens")


def test_context_resent_without_cache():
    spans = [llm(i, "gpt-4o", 3_000, 100) for i in range(4)]
    result = analyze_trace(trace(), spans)

    rec = by_id(result)["trace-caching:gpt-4o"]
    assert rec["savings_usd"] == pytest.approx(3 * 3_000 * (2.5 - 0.25) / 1_000_000)
    assert rec["evidence"]["prefix_tokens"] == 3_000
    assert [r["span_id"] for r in rec["spans"]] == ["s1", "s2", "s3"]
    assert "Context resent without a cache read." in result["span_notes"]["s1"]
    assert "trace-model:gpt-4o" in by_id(result)


def test_repeated_tool_calls_and_tool_errors():
    spans = [
        tool(0, "Read", {"path": "a.py"}),
        tool(1, "Read", {"path": "a.py"}),
        tool(2, "Read", {"path": "a.py"}),
        tool(3, "Shell", {"command": "pytest"}, tool_error="exit code 1"),
        tool(4, "Shell", {"command": "pytest -x"}),
        tool(5, "Edit", {"path": "b.py"}, status="failed", error="file not found"),
    ]
    result = analyze_trace(trace("claude-code"), spans)
    recs = by_id(result)

    repeated = next(r for r in result["recommendations"] if r["id"].startswith("trace-repeated-tool:Read:"))
    assert repeated["severity"] == "medium"
    assert len(repeated["spans"]) == 3
    assert result["span_notes"]["s1"] == ["Same call as step 0."]

    errors = recs["trace-tool-errors"]
    assert errors["severity"] == "high"
    assert errors["evidence"] == {"tool_calls": 6, "failed": 2, "unrecovered": 1, "failure_rate": 0.333}
    assert [r["reason"] for r in errors["spans"]] == ["exit code 1", "file not found"]
    assert result["span_notes"]["s3"] == ["Tool error."]
    assert result["span_notes"]["s5"] == ["Tool error, never succeeded afterwards."]


def test_context_growth_over_an_agent_run():
    contexts = [10_000, 20_000, 40_000, 60_000, 80_000]
    spans = [llm(i, "claude-sonnet-4-5", c, 200, cache_read_tokens=int(c * 0.9)) for i, c in enumerate(contexts)]
    result = analyze_trace(trace("claude-code"), spans)

    rec = by_id(result)["trace-context-growth"]
    assert rec["evidence"]["first_context_tokens"] == 10_000
    assert rec["evidence"]["peak_context_tokens"] == 80_000
    assert rec["spans"][0]["span_id"] == "s4"
    assert result["complexity"] == MEDIUM
    assert "trace-cost-concentration" not in by_id(result)


def test_claude_code_spans_grouping_several_calls_use_the_average_context():
    spans = [llm(0, "claude-sonnet-4-5", 90_000, 600, metadata={"model_calls": 6, "avg_context_tokens": 15_000})]
    result = analyze_trace(trace("claude-code"), spans)
    assert result["max_context_tokens"] == 15_000
    assert result["steps"][0]["context_tokens"] == 15_000
    assert result["complexity"] == SIMPLE


def test_one_step_concentrating_the_cost():
    spans = [llm(0, "gpt-4o", 1_000, 100), llm(1, "gpt-4o", 40_000, 100), llm(2, "gpt-4o", 1_000, 100)]
    result = analyze_trace(trace(), spans)

    rec = by_id(result)["trace-cost-concentration"]
    assert rec["severity"] == "medium"
    assert rec["spans"][0]["span_id"] == "s1"
    assert "Context too large" in rec["action"]
    assert not any(r["category"] == "caching" for r in result["recommendations"])


def test_failed_model_calls():
    spans = [llm(0, "gpt-4o", 1_000, 100), llm(1, "gpt-4o", 0, 0, status="failed", error="429 rate limit")]
    result = analyze_trace(trace(), spans)

    rec = by_id(result)["trace-llm-errors"]
    assert rec["severity"] == "medium"
    assert rec["spans"] == [{"span_id": "s1", "step": 1, "label": "gpt-4o", "reason": "429 rate limit"}]
    assert result["steps"][1]["alternative"] is None


def test_trace_metric_verdicts():
    spans = [
        llm(0, "gpt-4o", 1_000, 100, metrics=[
            {"name": "hallucination", "value": ["Le fichier existe déjà"], "reason": "Affirmation non vérifiée",
             "judge": "new@v3"},
            {"name": "answer_relevance", "value": 0.1, "judge": "old@v1"},
        ]),
    ]
    metrics = [{"name": "task_completion", "value": False, "reason": "Les tests n'ont pas été lancés.",
                "judge": "new@v3"}]
    result = analyze_trace(trace(metrics=metrics), spans, tag="new@v3")
    recs = by_id(result)

    assert recs["metric:task_completion"]["severity"] == "high"
    assert recs["metric:task_completion"]["detail"] == "Les tests n'ont pas été lancés."
    assert recs["metric:hallucination"]["spans"][0]["reason"] == "Affirmation non vérifiée"
    assert "metric:answer_relevance" not in recs
    assert "Answers with unsupported claims." in result["span_notes"]["s0"]


def test_empty_trace():
    result = analyze_trace(trace(), [])
    assert result["recommendations"] == [] and result["steps"] == []
    assert result["cost_usd"] == 0 and result["complexity"] == SIMPLE
    assert result["prompts"] == []


def test_cursor_trace_gets_a_rule_to_paste():
    metrics = [{"name": "instruction_adherence", "value": 0.3, "reason": "A modifié un fichier non demandé.",
                "judge": "new@v4"}]
    spans = [
        tool(0, "Read", {"path": "a.py"}),
        *(tool(i, "Shell", {"command": "pytest"}, status="failed", tool_error="exit 1") for i in (1, 2, 3)),
        llm(4, "claude-opus-4-5", 5_000, 300),
    ]
    result = analyze_trace(trace("cursor", metrics=metrics), spans, tag="new@v4")
    recs = by_id(result)

    assert result["agent_target"] == "cursor"
    assert recs["metric:instruction_adherence"]["fix"]["file"] == ".cursor/rules/aitrajectiq.mdc"
    assert recs["trace-tool-errors"]["fix"]["kind"] == "rule"
    [prompt] = result["prompts"]
    assert prompt["id"] == "cursor"
    # The rule covers the metric verdict, the repeated failing command and the tool errors.
    assert set(prompt["changes"]) == {"User instructions poorly followed", "The agent is looping",
                                      "Too many failing tool calls"}


def test_cheaper_model_for_models_priced_from_the_web():
    set_web_prices({
        "claude-opus-5.5": {"in": 5.0, "out": 25.0},
        "claude-sonnet-5.5": {"in": 3.0, "out": 15.0},
    })
    try:
        assert cheaper_model("claude-opus-5.5") == "claude-sonnet-5.5"
        assert cheaper_model("claude-opus-4.5") == "claude-sonnet-4.6"
    finally:
        set_web_prices({})
    assert cheaper_model("claude-opus-5.5") is None
    assert cheaper_model("gpt-4o-mini") is None
    assert cheaper_model(None) is None
