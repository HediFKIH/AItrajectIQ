from datetime import datetime, timedelta
from types import SimpleNamespace

from evaluation.actions import Turn, action_log, actions, earlier_turns_text, files_changed, is_agentic, run_mode

T0 = datetime(2026, 10, 8, 12, 0, 0)


def span(i, span_type="tool", **fields):
    base = dict(
        span_type=span_type, name=None, agent_name=None, status=None, error=None,
        tool_name=None, tool_input=None, tool_output=None, tool_error=None,
        response=None, parent_span_id="root", step_index=i, timestamp=T0 + timedelta(seconds=i),
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_edit_shell_and_dispatched_tools_are_classified():
    acts = actions([
        span(1, tool_name="StrReplace", tool_input={"path": "app/main.py", "old_string": "a"}),
        span(2, tool_name="Shell", tool_input={"command": "pytest app/tests -q"}, tool_output="3 passed"),
        span(3, tool_name="CallDynamicTool", tool_input={"toolName": "ReadLints", "arguments": {"paths": ["x"]}}),
        span(4, tool_name="Grep", tool_input={"pattern": "foo", "path": "app"}),
    ])
    assert [(a.tool, a.kind, a.verifies) for a in acts] == [
        ("StrReplace", "edit", False),
        ("Shell", "command", True),
        ("ReadLints", "tool", True),
        ("Grep", "tool", False),
    ]
    assert acts[0].target == "app/main.py"
    assert acts[1].target == "pytest app/tests -q"


def test_failed_command_does_not_count_as_verification():
    acts = actions([span(1, tool_name="Shell", tool_input={"command": "npm run build"}, error="blocked by hook")])
    assert acts[0].failed and not acts[0].verifies
    assert acts[0].error == "blocked by hook"


def test_root_agent_is_not_an_action_but_subagents_are():
    acts = actions([
        span(0, "agent", name="Cursor Agent", parent_span_id=None),
        span(2, "agent", name="Explication du PHP", response="PHP est..."),
        span(1, tool_name="Read", tool_input={"path": "a.py"}),
    ])
    assert [(a.tool, a.kind) for a in acts] == [("Read", "tool"), ("Explication du PHP", "subagent")]


def test_files_changed_keeps_successful_edits_once():
    acts = actions([
        span(1, tool_name="Write", tool_input={"path": "a.py"}),
        span(2, tool_name="StrReplace", tool_input={"path": "a.py"}),
        span(3, tool_name="StrReplace", tool_input={"path": "b.py"}, error="no match"),
    ])
    assert files_changed(acts) == ["a.py"]


def test_is_agentic_by_origin_or_steps():
    chatbot = SimpleNamespace(metadata={})
    assert not is_agentic(chatbot, [span(1, "llm")])
    assert is_agentic(chatbot, [span(1, "llm"), span(2, tool_name="search")])
    assert is_agentic(SimpleNamespace(metadata={"source": "cursor"}), [])


def test_action_log_keeps_start_and_end_when_too_long():
    acts = actions([span(i, tool_name="Read", tool_input={"path": f"file_{i}.py"}) for i in range(200)])
    log = action_log(acts, limit=1000)
    assert len(log) <= 1100
    assert log.startswith("1. [tool] Read file_0.py")
    assert "actions omises" in log
    assert log.rstrip().endswith("file_199.py")


def test_run_mode_from_the_latest_reminder_then_the_metadata():
    trace = SimpleNamespace(metadata={"mode": "agent"})
    request = ("<user_query>analyse ces améliorations</user_query>\n"
               "<system_reminder>Ask mode is active. You MUST NOT make any edits.</system_reminder>")
    assert run_mode(trace, request) == "ask"
    later = request + "\n<system_reminder>You are now in Agent mode.</system_reminder>"
    assert run_mode(trace, later) == "agent"
    assert run_mode(trace, "<user_query>continuer</user_query>") == "agent"
    assert run_mode(SimpleNamespace(metadata={"mode": "chat"}), "") == "ask"
    assert run_mode(SimpleNamespace(metadata=None), "") is None


def test_earlier_turns_keep_the_most_recent_when_too_long():
    turns = [
        Turn(request=f"demande {i}", reports=[("Revue", f"rapport {i} " + "x" * 600)], answer=f"réponse {i}")
        for i in range(10)
    ]
    text = earlier_turns_text(turns, limit=2500)
    assert "Turn 10 request: demande 9" in text
    assert "subagent Revue reported: rapport 9" in text
    assert "demande 0" not in text
    assert len(text) <= 2600
    assert earlier_turns_text([]) == ""
