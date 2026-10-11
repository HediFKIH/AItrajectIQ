import pytest

from services.recommendations import (
    CHEAPER_MODEL,
    COMPLEX,
    MEDIUM,
    SIMPLE,
    Call,
    Recommendation,
    build_calls,
    classify_complexity,
    combined_savings,
    complexity_breakdown,
    failing_models,
    RULES,
    agent_target,
    metric_recommendations,
    model_routing,
    model_upgrade,
    prompt_caching,
    prompt_findings,
    quality_score,
    recommended_prompts,
    severity_from_savings,
    sort_recommendations,
)
from utils.cost import PRICES, compute_cost, price_key


def call(model="gpt-4o", tokens_in=1_000, tokens_out=100, complexity=SIMPLE, trace_id="t", **kw) -> Call:
    cost = compute_cost(model, tokens_in, tokens_out, kw.get("cache_read", 0), kw.get("cache_write", 0))
    return Call(trace_id=trace_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost=cost,
                complexity=complexity, **kw)


def calls(n, **kw) -> list[Call]:
    return [call(trace_id=f"t{i}", **kw) for i in range(n)]


@pytest.mark.parametrize("tokens_in, tokens_out, agentic, tools, expected", [
    (1_000, 100, False, 0, SIMPLE),
    (4_000, 500, False, 0, SIMPLE),
    (4_001, 100, False, 0, MEDIUM),
    (1_000, 501, False, 0, MEDIUM),
    (30_000, 10, False, 0, COMPLEX),
    (1_000, 2_000, False, 0, COMPLEX),
    (500, 50, True, 0, MEDIUM),
    (500, 50, True, 5, COMPLEX),
    (30_000, 50, True, 1, COMPLEX),
])
def test_classify_complexity(tokens_in, tokens_out, agentic, tools, expected):
    assert classify_complexity(tokens_in, tokens_out, agentic, tools) == expected


def test_cheaper_models_are_priced_and_cheaper():
    for big, small in CHEAPER_MODEL.items():
        assert big in PRICES and small in PRICES, (big, small)
        assert PRICES[small]["in"] < PRICES[big]["in"], (big, small)
        assert price_key(big) == big and price_key(small) == small


def test_price_key_resolves_dated_ids_and_rejects_unknown():
    assert price_key("claude-sonnet-4-5-20250929") == "claude-sonnet-4.5"
    assert price_key("gpt-4o-2024-08-06") == "gpt-4o"
    assert price_key("my-local-model") is None
    assert price_key("") is None


def test_model_routing_savings():
    simple = calls(5, model="gpt-4o-2024-08-06")
    total = sum(c.cost for c in simple)
    [rec] = model_routing(simple, total)

    expected = sum(c.cost - compute_cost("gpt-4o-mini", c.tokens_in, c.tokens_out) for c in simple)
    assert rec.id == "model-routing:gpt-4o"
    assert rec.category == "model"
    assert rec.savings_usd == pytest.approx(expected, abs=1e-6)
    assert rec.severity == "high"
    assert rec.evidence["alternative"] == "gpt-4o-mini"
    assert rec.evidence["simple_calls"] == 5
    assert len(rec.examples) == 3


def test_model_routing_ignores_complex_failed_and_rare_calls():
    assert model_routing(calls(4), 1.0) == []
    assert model_routing(calls(5, complexity=COMPLEX), 1.0) == []
    assert model_routing(calls(5, failed=True), 1.0) == []
    assert model_routing(calls(5, model="gpt-4o-mini"), 1.0) == []
    assert model_routing(calls(5, model="my-local-model"), 1.0) == []


def test_model_routing_reports_alternative_quality():
    own = calls(5, model="gpt-4o", quality=0.9)
    alt = calls(5, model="gpt-4o-mini", quality=0.85)
    [rec] = model_routing(own + alt, 1.0)
    assert rec.evidence["alternative_quality"] == pytest.approx(0.85)
    assert "already used" in rec.detail


def test_model_upgrade_on_poor_complex_calls():
    group = calls(4, model="gpt-4o-mini", complexity=COMPLEX, quality=0.4) + \
        calls(4, model="gpt-4o-mini", complexity=COMPLEX, quality=0.9)
    [rec] = model_upgrade(group)
    assert rec.id == "model-upgrade:gpt-4o-mini"
    assert rec.evidence["bad_rate"] == 0.5
    assert rec.severity == "high"
    assert rec.evidence["alternative"] in {"gpt-4o", "gpt-4-turbo", "gpt-4"}

    assert model_upgrade(calls(8, model="gpt-4o-mini", complexity=COMPLEX, quality=0.9)) == []


def test_prompt_caching_savings():
    prompt = "x" * 8_000
    group = calls(6, model="claude-sonnet-4.5", tokens_in=3_000, tokens_out=200, system_prompt=prompt)
    [rec] = prompt_caching(group, sum(c.cost for c in group))

    expected = 5 * 2_000 * PRICES["claude-sonnet-4.5"]["in"] * 0.9 / 1_000_000
    assert rec.category == "caching"
    assert rec.savings_usd == pytest.approx(expected)
    assert rec.evidence["prompt_tokens_estimate"] == 2_000
    assert "cache_control" in rec.action


def test_prompt_caching_skips_cached_short_or_varied_prompts():
    prompt = "x" * 8_000
    cached = calls(6, model="claude-sonnet-4.5", tokens_in=3_000, cache_read=2_000, system_prompt=prompt)
    assert prompt_caching(cached, 1.0) == []
    assert prompt_caching(calls(6, system_prompt="short"), 1.0) == []
    varied = [call(trace_id=f"t{i}", system_prompt=f"{i}" + prompt) for i in range(6)]
    assert prompt_caching(varied, 1.0) == []


def test_failing_models():
    group = calls(8, model="gpt-4o") + calls(2, model="gpt-4o", failed=True)
    [rec] = failing_models(group)
    assert rec.category == "reliability"
    assert rec.evidence["failure_rate"] == 0.2
    assert rec.severity == "high"
    assert failing_models(calls(9, model="gpt-4o", failed=True)) == []


def test_metric_recommendations_thresholds():
    summary = {
        "span_metrics": {
            "hallucination": {"avg": 0.5, "evaluated_count": 10},
            "answer_relevance": {"avg": 0.9, "evaluated_count": 10},
            "is_json": {"avg": 0.2, "evaluated_count": 3},
            "completeness": {"avg": None, "evaluated_count": 10},
        },
        "trace_metrics": {
            "tool_error_rate": {"avg": 0.3, "evaluated_count": 6},
        },
    }
    fired = {rec.id: rec for _, rec in metric_recommendations(summary)}
    assert set(fired) == {"metric:hallucination", "metric:tool_error_rate"}
    assert fired["metric:hallucination"].severity == "high"
    assert fired["metric:hallucination"].category == "prompt"
    assert "50%" in fired["metric:hallucination"].detail
    assert fired["metric:tool_error_rate"].severity == "medium"
    assert fired["metric:tool_error_rate"].category == "agent"
    assert fired["metric:hallucination"].fix["kind"] == "prompt"
    assert "context provided" in fired["metric:hallucination"].fix["content"]


def test_agent_metric_fix_goes_where_the_agent_reads_it():
    summary = {
        "span_metrics": {"hallucination": {"avg": 0.5, "evaluated_count": 10}},
        "trace_metrics": {"instruction_adherence": {"avg": 0.5, "evaluated_count": 14}},
    }
    fired = {rec.id: rec for _, rec in metric_recommendations(summary, "cursor")}

    rule = fired["metric:instruction_adherence"]
    assert rule.fix["kind"] == "rule" and rule.fix["file"] == ".cursor/rules/aitrajectiq.mdc"
    assert rule.fix["content"].startswith("---\n") and "alwaysApply: true" in rule.fix["content"]
    assert "Cursor" in rule.action
    # Answer metrics are not computed on agent runs: their fix stays a system prompt instruction.
    assert fired["metric:hallucination"].fix["kind"] == "prompt"

    claude = {rec.id: rec for _, rec in metric_recommendations(summary, "claude-code")}
    assert claude["metric:instruction_adherence"].fix["file"] == "CLAUDE.md"


def test_agent_target():
    assert agent_target(["cursor", "cursor", "claude-code", None]) == "cursor"
    assert agent_target(["sdk", None]) == "sdk"
    assert agent_target([]) == "sdk"


def test_recommended_prompts_for_a_cursor_project():
    rules = [RULES["instruction_adherence"], RULES["claims_grounded"]]
    [prompt] = recommended_prompts(rules, "cursor", calls(3, model="claude-opus-4-5"))

    assert prompt["id"] == "cursor" and prompt["file"] == ".cursor/rules/aitrajectiq.mdc"
    assert RULES["instruction_adherence"].snippet in prompt["content"]
    assert RULES["claims_grounded"].snippet in prompt["content"]
    assert prompt["changes"] == [RULES["instruction_adherence"].title, RULES["claims_grounded"].title]


def test_recommended_system_prompt_appends_the_missing_instructions():
    system = "Tu es l'assistant du support. Réponds en français."
    group = calls(4, model="gpt-4.1", system_prompt=system) + calls(1, model="gpt-4.1", system_prompt="Autre prompt")
    rules = [RULES["hallucination"], RULES["answer_relevance"], RULES["context_precision"]]
    [prompt] = recommended_prompts(rules, "sdk", group)

    assert prompt["id"] == "system" and prompt["current"] == system
    assert prompt["content"].startswith(system) and "## Instructions" in prompt["content"]
    assert RULES["hallucination"].snippet in prompt["content"]
    assert prompt["calls"] == 4 and prompt["model"] == "gpt-4.1"
    # context_precision has no instruction: it is a retrieval setting.
    assert prompt["changes"] == [RULES["hallucination"].title, RULES["answer_relevance"].title]


def test_recommended_system_prompt_skips_instructions_already_present():
    system = f"Rôle : support.\n{RULES['hallucination'].snippet}"
    [prompt] = recommended_prompts([RULES["hallucination"]], "sdk", calls(2, system_prompt=system))
    assert prompt["content"] == system and prompt["changes"] == []


def test_no_recommended_prompt_without_instructions_or_system_prompt():
    assert recommended_prompts([], "cursor", calls(3)) == []
    assert recommended_prompts([RULES["context_recall"]], "sdk", calls(3)) == []


def test_baseline_rule_for_a_coding_agent_without_fired_rules():
    [prompt] = recommended_prompts([], "cursor", calls(3), baseline=True)
    assert prompt["baseline"] is True and prompt["changes"] == []
    assert prompt["file"] == ".cursor/rules/aitrajectiq.mdc"
    assert RULES["claims_grounded"].snippet in prompt["content"]
    assert RULES["task_completion"].snippet in prompt["content"]


def test_baseline_gives_way_to_fired_agent_rules():
    [prompt] = recommended_prompts([RULES["tool_error_rate"]], "claude-code", calls(3), baseline=True)
    assert prompt["baseline"] is False
    assert prompt["changes"] == [RULES["tool_error_rate"].title]
    assert RULES["claims_grounded"].snippet not in prompt["content"]


def test_no_baseline_for_an_app_without_system_prompt():
    assert recommended_prompts([], "sdk", calls(3), baseline=True) == []


def test_combined_savings_do_not_count_the_same_dollars_twice():
    # Half the cost moves to a model 10x cheaper; caching saved 20 on the current model.
    assert combined_savings(100.0, 45.0, 20.0) == pytest.approx(45.0 + 20.0 * 0.55)
    assert combined_savings(100.0, 0.0, 20.0) == 20.0
    assert combined_savings(100.0, 90.0, 80.0) <= 100.0
    assert combined_savings(0.0, 5.0, 5.0) == 0.0


def test_prompt_findings_keep_the_judge_reasons():
    result = {"recommendations": [
        {"category": "agent", "title": "Instructions poorly followed", "detail": "50%",
         "examples": [{"trace_id": "t1", "reason": "Modified a file that was not requested"}], "spans": []},
        {"category": "model", "title": "Routing", "detail": "x", "examples": [], "spans": []},
    ]}
    assert prompt_findings(result) == ["Instructions poorly followed: 50%",
                                       "  example: Modified a file that was not requested"]


def test_severity_from_savings():
    assert severity_from_savings(10, 100) == "high"
    assert severity_from_savings(2, 100) == "medium"
    assert severity_from_savings(1, 100) == "low"
    assert severity_from_savings(1, 0) == "low"


def test_sort_recommendations():
    def rec(id, severity, savings=None):
        return Recommendation(id=id, category="model", severity=severity, title="", detail="", action="",
                              savings_usd=savings)

    ordered = sort_recommendations([rec("a", "low", 5), rec("b", "high"), rec("c", "medium", 1),
                                    rec("d", "high", 2), rec("e", "medium", 3)])
    assert [r.id for r in ordered] == ["d", "b", "e", "c", "a"]


def test_complexity_breakdown():
    data = calls(3, model="gpt-4o") + calls(1, model="gpt-4o", complexity=COMPLEX) + calls(2, model="my-local-model")
    overall, rows = complexity_breakdown(data)

    assert overall[SIMPLE]["calls"] == 5
    assert overall[COMPLEX]["calls"] == 1
    assert overall[MEDIUM]["calls"] == 0
    gpt = next(r for r in rows if r["model"] == "gpt-4o")
    assert gpt["calls"] == 4
    assert gpt[SIMPLE]["calls"] == 3
    assert gpt["alternative"] == "gpt-4o-mini"
    assert gpt["price_known"] is True
    local = next(r for r in rows if r["model"] == "my-local-model")
    assert local["price_known"] is False
    assert local["alternative"] is None
    assert local["no_alternative_reason"].startswith("Unknown price")
    assert gpt["no_alternative_reason"] is None
    assert rows[0]["cost_usd"] >= rows[-1]["cost_usd"]


def test_quality_score_skips_stale_judge():
    metrics = [
        {"name": "answer_relevance", "value": 0.8, "judge": "judge@v3"},
        {"name": "answer_relevance", "value": 0.0, "judge": "judge@v2"},
        {"name": "is_json", "value": True},
        {"name": "latency", "value": 999},
    ]
    assert quality_score(metrics, "judge@v3") == pytest.approx(0.9)
    assert quality_score(metrics, None) == pytest.approx(0.6)
    assert quality_score([], "judge@v3") is None


def test_build_calls():
    spans = [
        {"trace_id": "t1", "model": "gpt-4o", "tokens_in": 1_000, "tokens_out": 100, "cost_usd": None,
         "first_message": {"role": "system", "content": [{"type": "text", "text": "Tu es un assistant."}]}},
        {"trace_id": "t2", "model": "gpt-4o", "tokens_in": 1_000, "tokens_out": 100, "cost_usd": 0.5,
         "first_message": {"role": "user", "content": "bonjour"}},
        {"trace_id": "t3", "model": "gpt-4o", "tokens_in": 0, "tokens_out": 0},
        {"trace_id": "t3", "model": "gpt-4o", "tokens_in": 0, "tokens_out": 0, "status": "failed"},
        {"trace_id": "t4", "model": None, "tokens_in": 1_000, "tokens_out": 100},
    ]
    traces = [
        {"_id": "t1", "total_tool_spans": 6},
        {"_id": "t2", "metadata": {"source": "cursor"}},
        {"_id": "t3"},
    ]
    built = build_calls(spans, traces)

    assert len(built) == 4
    first, second, failed, unknown = built
    assert first.agentic and first.complexity == COMPLEX
    assert first.system_prompt == "Tu es un assistant."
    assert first.cost == compute_cost("gpt-4o", 1_000, 100)
    assert second.agentic and second.complexity == MEDIUM
    assert second.system_prompt is None
    assert second.cost == 0.5
    assert failed.failed and not failed.agentic
    assert unknown.model == "unknown" and unknown.complexity == SIMPLE
