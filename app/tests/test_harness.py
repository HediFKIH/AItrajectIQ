import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from services.harness import SHELL_READ_GUARD, Signals, add, analyze, collect, findings
from services.recommendations import analyze_trace

T0 = datetime(2026, 1, 1)


def span(i, kind="tool", **kw):
    fields = dict(
        id=f"s{i}", trace_id="t1", step_index=i, span_type=kind, name=None, status="completed", error=None,
        metadata=None, model=None, messages_in=None, tokens_in=None, tokens_out=None, cache_read_tokens=None,
        cache_write_tokens=None, cost_usd=None, temperature=None, tool_name=None, tool_input=None,
        tool_error=None, tool_output=None, timestamp=T0 + timedelta(seconds=i), metrics=[], parent_span_id=None,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def tool(i, name, args, **kw):
    return span(i, tool_name=name, tool_input=args, **kw)


def shell(i, command, **kw):
    return tool(i, "Shell", {"command": command}, **kw)


def trace(source="cursor", tid="t1", workspace=None, status="completed"):
    metadata = {"source": source} if source else {}
    if workspace:
        metadata["workspace"] = workspace
    return SimpleNamespace(id=tid, metadata=metadata, status=status, metrics=[])


def by_id(result):
    return {f["id"]: f for f in result}


def test_code_changed_without_a_check_gets_a_stop_hook():
    spans = [tool(0, "Read", {"path": "app/main.py"}), tool(1, "StrReplace", {"path": "app/main.py"})]
    rec = by_id(analyze(trace(workspace="c:\\dev\\proj"), spans, "cursor"))["harness-verify"]

    assert rec["category"] == "harness"
    assert rec["fix"]["file"] == ".cursor/hooks.json"
    assert json.loads(rec["fix"]["content"])["hooks"]["stop"][0]["command"] == "py -3 .cursor/hooks/verify.py"
    [script] = rec["extra_fixes"]
    assert script["file"] == ".cursor/hooks/verify.py"
    assert "CHECK = 'python -m pytest -q'" in script["content"]
    compile(script["content"], "verify.py", "exec")
    assert rec["spans"][0]["span_id"] == "s1"


def test_a_passing_check_after_the_last_edit_is_enough():
    spans = [tool(0, "StrReplace", {"path": "a.py"}), shell(1, "pytest -q", tool_output="3 passed")]
    assert "harness-verify" not in by_id(analyze(trace(), spans, "cursor"))
    docs_only = [tool(0, "Write", {"path": "README.md"})]
    assert "harness-verify" not in by_id(analyze(trace(), docs_only, "cursor"))
    running = [tool(0, "Write", {"path": "a.py"})]
    assert "harness-verify" not in by_id(analyze(trace(status="running"), running, "cursor"))


def test_project_hook_uses_the_check_seen_in_the_runs():
    checked = collect(trace(tid="a"), [tool(0, "Write", {"file_path": "a.py"}),
                                       shell(1, "cd app; python -m pytest tests -q", tool_output="ok")])
    total = Signals()
    for sig in (checked, *(collect(trace("claude-code", tid=f"r{i}"), [tool(0, "Edit", {"file_path": "b.py"})])
                           for i in range(3))):
        add(total, sig)
    rec = by_id(findings(total, "claude-code", "project"))["harness-verify"]

    assert rec["severity"] == "high"
    assert "3 of the 4 runs" in rec["detail"]
    assert rec["fix"]["file"] == ".claude/settings.json"
    assert "CHECK = 'cd app; python -m pytest tests -q'" in rec["extra_fixes"][0]["content"]
    assert [e["trace_id"] for e in rec["examples"]] == ["r0", "r1", "r2"]
    assert rec["spans"] == []


def test_powershell_errors_and_missing_commands_get_a_terminal_rule():
    spans = [
        shell(0, "cd app && pytest", error="The token '&&' is not a valid statement separator in this version."),
        shell(1, "rg foo", tool_output="rg : The term 'rg' is not recognized as the name of a cmdlet"),
    ]
    rec = by_id(analyze(trace(), spans, "cursor"))["harness-terminal"]

    assert rec["fix"]["file"] == ".cursor/rules/terminal.mdc"
    assert "PowerShell" in rec["fix"]["content"] and "`rg`" in rec["fix"]["content"]
    assert rec["severity"] == "medium"
    assert len(rec["spans"]) == 2


def test_blocked_checks_get_an_allow_rule():
    spans = [shell(i, "npm run build", error="blocked by hook") for i in range(2)]
    rec = by_id(analyze(trace("claude-code"), spans, "claude-code"))["harness-blocked"]

    assert rec["severity"] == "high"
    assert rec["title"] == "The agent can't run its checks"
    assert json.loads(rec["extra_fixes"][0]["content"]) == {"permissions": {"allow": ["Bash(npm run build:*)"]}}
    assert "`npm run build`" in rec["fix"]["content"]


def test_reads_in_generated_files():
    spans = [tool(0, "Read", {"path": "dashboard/node_modules/react/index.js"}),
             shell(1, "Get-Content package-lock.json")]
    cursor = by_id(analyze(trace(), spans, "cursor"))["harness-generated"]
    assert cursor["fix"]["file"] == ".cursorignore"
    assert set(cursor["fix"]["content"].split()) == {"node_modules/", "package-lock.json"}

    claude = by_id(analyze(trace("claude-code"), spans, "claude-code"))["harness-generated"]
    deny = json.loads(claude["fix"]["content"])["permissions"]["deny"]
    assert set(deny) == {"Read(**/node_modules/**)", "Read(**/package-lock.json)"}


def test_failing_mcp_servers():
    spans = [
        tool(0, "mcp__github__create_issue", {}, error="401 Unauthorized"),
        tool(1, "mcp__github__list_issues", {}, error="HTTP 401"),
        tool(2, "CallDynamicTool", {"namespace": "user-GitLab", "toolName": "get_issue", "arguments": {}},
             error="ECONNREFUSED"),
        tool(3, "CallDynamicTool", {"namespace": "user-GitLab", "toolName": "list_issues", "arguments": {}}),
        tool(4, "CallDynamicTool", {"namespace": "user-GitLab", "toolName": "get_branch", "arguments": {}},
             error="connection refused"),
        tool(5, "CallDynamicTool", {"namespace": "cursor", "toolName": "ReadLints", "arguments": {}}, error="x"),
    ]
    recs = by_id(analyze(trace("claude-code"), spans, "claude-code"))

    github = recs["harness-mcp:github"]
    assert github["severity"] == "high" and "authentication" in github["detail"]
    assert "claude mcp get github" in github["fix"]["content"]
    gitlab = recs["harness-mcp:user-GitLab"]
    assert gitlab["severity"] == "medium" and gitlab["evidence"] == {"calls": 3, "failed": 2}
    assert not any(key.startswith("harness-mcp:cursor") for key in recs)


def test_long_tool_outputs():
    spans = [shell(i, "pytest", tool_output="x" * 16_000) for i in range(2)]
    rec = by_id(analyze(trace(), spans, "cursor"))["harness-output"]
    assert rec["evidence"] == {"outputs": 2, "tokens_estimate": 8_000}
    assert "pytest -q --tb=short" in rec["fix"]["content"]

    sdk = by_id(analyze(trace(None), spans, "sdk"))["harness-output"]
    assert sdk["fix"]["kind"] == "code" and "clip_tool_output" in sdk["fix"]["content"]


def test_long_exploration_without_subagent():
    # Claude Code's main agent hangs under its session, which is not a span.
    main = span(0, "agent", name="Claude Code", sdk_span_id="main", parent_span_id="session")
    reads = [main, *(tool(i, "Read", {"path": f"f{i}.py"}, parent_span_id="main") for i in range(1, 21))]
    rec = by_id(analyze(trace("claude-code"), reads, "claude-code"))["harness-explore"]
    assert rec["fix"]["file"] == ".claude/agents/explorer.md"
    assert "model: haiku" in rec["fix"]["content"]

    delegated = [*reads, span(30, "agent", name="explore", sdk_span_id="sub", parent_span_id="main")]
    assert "harness-explore" not in by_id(analyze(trace("claude-code"), delegated, "claude-code"))


def test_failed_replacements():
    spans = [tool(i, "StrReplace", {"path": "a.py"}, error="old_string not found in file") for i in range(2)]
    rec = by_id(analyze(trace(), spans, "cursor"))["harness-edit-miss"]
    assert rec["fix"]["file"] == ".cursor/rules/edits.mdc"
    assert "formatter" not in rec["detail"]


def test_shell_used_to_read_files():
    spans = [shell(0, "cat a.py"), shell(1, "cd app; grep -r foo ."), shell(2, "Get-Content b.py"),
             shell(3, "npm test | tail -n 20")]
    rec = by_id(analyze(trace("claude-code"), spans, "claude-code"))["harness-shell-read"]
    assert rec["evidence"] == {"commands": 3}
    guard = rec["extra_fixes"][1]
    assert guard["file"] == ".claude/hooks/shell_read_guard.py"
    compile(SHELL_READ_GUARD, "guard.py", "exec")
    assert "harness-shell-read" not in by_id(analyze(trace(None), spans, "sdk"))


def test_invalid_tool_arguments_on_an_sdk_agent():
    spans = [tool(i, "search_orders", {"q": 1}, error="1 validation error for SearchOrders") for i in range(2)]
    rec = by_id(analyze(trace(None), spans, "sdk"))["harness-schema"]
    assert json.loads(rec["fix"]["content"])["function"]["strict"] is True
    assert "harness-schema" not in by_id(analyze(trace(), spans, "cursor"))


def test_project_map_from_files_reread_in_most_runs():
    total = Signals()
    for i in range(5):
        add(total, collect(trace(tid=f"r{i}", workspace="c:\\dev\\proj"), [
            tool(0, "Read", {"path": "c:\\dev\\proj\\app\\main.py"}),
            tool(1, "Read", {"path": f"c:\\dev\\proj\\other_{i}.py"}),
        ]))
    rec = by_id(findings(total, "cursor", "project"))["harness-memory"]
    assert rec["fix"]["file"] == ".cursor/rules/project.mdc"
    assert "`app/main.py`: role to fill in (read in 5 of 5 runs)" in rec["fix"]["content"]
    assert "other_0.py" not in rec["fix"]["content"]
    assert "harness-memory" not in by_id(findings(total, "cursor", "trace"))


def test_trace_recommendations_include_the_harness():
    spans = [tool(0, "StrReplace", {"path": "a.py"}), span(1, "llm", model="gpt-4o", tokens_in=1_000, tokens_out=100)]
    result = analyze_trace(trace(), spans)
    rec = next(r for r in result["recommendations"] if r["id"] == "harness-verify")
    assert rec["extra_fixes"][0]["file"] == ".cursor/hooks/verify.py"
    assert "Code handed back unverified." in result["span_notes"]["s0"]

    chatbot = analyze_trace(trace(None), [span(0, "llm", model="gpt-4o", tokens_in=1_000, tokens_out=100)])
    assert not any(r["category"] == "harness" for r in chatbot["recommendations"])
