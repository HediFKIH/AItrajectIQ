import asyncio
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from evaluation.runner import _trace_loop_detection, _trace_step_efficiency
from otel.claude_code_answer import (
    ANSWER_FLAG,
    MAX_ARG,
    answer_span,
    compact_input,
    match_tool_inputs,
    matches,
    stop_updates,
)

T0 = datetime(2026, 10, 11, 7, 22, 52)
INTEGRATIONS = Path(__file__).resolve().parents[2] / "sdk" / "aitrajectiq" / "integrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, INTEGRATIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def span(i, span_type="tool", **fields):
    base = dict(
        id=f"s{i}", span_type=span_type, name=None, agent_name=None, status="ok", error=None,
        tool_name=None, tool_input=None, tool_output=None, tool_error=None, metadata=None,
        response=None, messages_in=None, sdk_span_id=f"o{i}", parent_span_id="main",
        step_index=i, timestamp=T0 + timedelta(seconds=i),
    )
    base.update(fields)
    return SimpleNamespace(**base)


def claude_run():
    return [
        span(0, "agent", name="Claude Code", sdk_span_id="main", parent_span_id="session"),
        span(1, "llm"),
        span(2, tool_name="Read", name="Read"),
        span(3, tool_name="Read", name="Read"),
        span(4, "agent", name="Explorer", sdk_span_id="sub", parent_span_id="main"),
        span(5, "llm", parent_span_id="sub"),
        span(6, tool_name="Bash", name="Bash", tool_input={"command": "pytest"}),
        span(7, "llm"),
    ]


TOOLS = [
    {"name": "Read", "input": {"file_path": "app/main.py"}},
    {"name": "Task", "input": {"description": "Explorer"}},
    {"name": "Read", "input": {"file_path": "app/db.py"}},
    {"name": "Bash", "input": {"command": "pytest"}},
]


def test_compact_input_cuts_long_values():
    data = compact_input({"file_path": "a.py", "content": "x" * 5000, "limit": 10, "edits": [{"a": 1}]})
    assert data["file_path"] == "a.py" and data["limit"] == 10
    assert len(data["content"]) == MAX_ARG + 1 and data["edits"] == '[{"a": 1}]'
    assert compact_input("not a dict") == {}


def test_tool_inputs_are_matched_by_name_in_order():
    spans = claude_run()
    tools = [s for s in spans if s.span_type == "tool"]
    assert match_tool_inputs(tools, TOOLS) == {
        "s2": {"file_path": "app/main.py"},
        "s3": {"file_path": "app/db.py"},
        "s6": {"command": "pytest"},
    }


def test_answer_goes_on_the_main_agents_last_call():
    spans = claude_run()
    assert answer_span(spans).id == "s7"
    spans[-1].error = "529 overloaded"
    assert answer_span(spans).id == "s1"


def test_stop_updates():
    spans = claude_run()
    spans[1].metadata = {ANSWER_FLAG: True}
    spans[1].response = "older answer"
    updates = stop_updates(None, spans, {"answer": "Voici l'analyse.", "prompt": "analyser le code", "tools": TOOLS})

    assert updates["s7"]["response"] == "Voici l'analyse."
    assert updates["s7"]["messages_in"] == [{"role": "user", "content": "analyser le code"}]
    assert updates["s7"]["metadata"] == {ANSWER_FLAG: True}
    assert updates["s1"] == {"response": None, "messages_in": None, "metadata": {}}
    assert updates["s2"] == {"tool_input": {"file_path": "app/main.py"}}
    assert "s6" not in updates   # the telemetry already gave its arguments


def test_a_stop_matches_the_trace_of_its_prompt():
    trace = SimpleNamespace(started_at=T0)
    assert matches(trace, {"prompt_at": "2026-10-11T07:22:50Z"})
    assert not matches(trace, {"prompt_at": "2026-10-11T07:30:00Z"})
    assert matches(trace, {})


def test_unknown_targets_are_neither_repeats_nor_loops():
    trace = SimpleNamespace(id="t", metrics=[])
    reads = [span(i, tool_name="Read") for i in range(7)]
    efficiency = asyncio.run(_trace_step_efficiency(trace, reads))
    loop = asyncio.run(_trace_loop_detection(trace, reads))
    assert efficiency.value == 1.0 and "unknown target for 7" in efficiency.reason
    assert loop.value is False

    same = [span(i, tool_name="Read", tool_input={"file_path": "a.py"}) for i in range(3)]
    assert asyncio.run(_trace_loop_detection(trace, same)).value is True
    assert asyncio.run(_trace_step_efficiency(trace, same)).value == round(1 - 2 / 3, 2)


def test_hook_reads_the_last_prompt_of_the_transcript(tmp_path):
    hook = load("claude_code_hook")
    entries = [
        {"type": "user", "message": {"role": "user", "content": "première demande"}, "timestamp": "2026-10-11T07:00:00Z"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ancienne réponse"}]}},
        {"type": "user", "message": {"content": "analyser le code"}, "timestamp": "2026-10-11T07:22:52Z"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Je lis les fichiers."},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "app/main.py"}},
        ]}},
        {"type": "user", "isSidechain": True, "message": {"content": "sous-agent"}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "..."}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Voici l'analyse."}]}},
        {"type": "user", "isMeta": True, "message": {"content": "meta"}},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\nnot json\n", encoding="utf-8")

    turn = hook.last_turn(hook.read_transcript(str(path)))
    assert turn["prompt"] == "analyser le code" and turn["prompt_at"] == "2026-10-11T07:22:52Z"
    assert turn["answer"] == "Voici l'analyse."
    assert turn["tools"] == [{"name": "Read", "input": {"file_path": "app/main.py"}}]
    assert hook.last_turn([]) == {}


def test_setup_merges_claude_settings_without_losing_other_hooks():
    setup = load("claude_code_setup")
    other = {"hooks": [{"type": "command", "command": "notify-send done"}]}
    old = {"hooks": [{"type": "command", "command": "python /x/claude_code_hook.py"}]}
    settings = {"model": "opus", "env": {"FOO": "1"}, "hooks": {"Stop": [other, old], "PreToolUse": [other]}}

    merged = setup.merge_settings(settings, "http://localhost:8000", "atq_k", '"py" "hook"')
    assert merged["model"] == "opus" and merged["env"]["FOO"] == "1"
    assert merged["env"]["AITRAJECTIQ_API_KEY"] == "atq_k"
    assert merged["env"]["OTEL_EXPORTER_OTLP_HEADERS"] == "x-api-key=atq_k"
    assert merged["hooks"]["PreToolUse"] == [other]
    assert merged["hooks"]["Stop"][0] == other and len(merged["hooks"]["Stop"]) == 2
    assert merged["hooks"]["Stop"][1]["hooks"][0]["command"] == '"py" "hook"'
