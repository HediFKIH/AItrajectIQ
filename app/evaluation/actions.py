"""
What an agent actually did, read from its tool and subagent spans.

An agent's LLM spans only carry the request and the answer: the code it read or changed and
the commands it ran live in its tool spans. The agentic metrics judge the final answer
against this action log instead of judging it in a vacuum.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

TOOL_ORIGINS = ("cursor", "claude-code")

_EDIT_WORDS = ("edit", "write", "replace", "patch")
_EDIT_NAMES = {"delete", "deletefile", "createfile"}
_SHELL_WORDS = ("shell", "bash", "terminal", "command", "powershell", "exec")
_CHECK_TOOL_WORDS = ("lint", "diagnostic", "typecheck", "runtest")
_CHECK_COMMAND = re.compile(
    r"\b(pytest|unittest|tox|nox|jest|vitest|mocha|playwright|cypress|tsc|eslint|ruff|flake8|mypy|pylint|pyright"
    r"|(npm|pnpm|yarn|bun)\s+(run\s+)?(test|build|lint|typecheck|check)"
    r"|cargo\s+(test|check|build|clippy)|go\s+(test|vet|build)|mvn|gradle|dotnet\s+(test|build)|make\s+test)\b",
    re.I,
)
_PATH_KEYS = ("path", "file_path", "target_file", "filePath", "notebook_path", "file")
_COMMAND_KEYS = ("command", "cmd", "script")
_FAILED_STATUS = {"failed", "error"}


@dataclass
class Action:
    tool: str
    kind: str                 # edit | command | subagent | tool
    target: Optional[str]     # file, command, or a short view of the arguments
    failed: bool
    error: Optional[str]
    output: Optional[str]
    verifies: bool            # a test, lint, type check or build that succeeded

    @property
    def signature(self) -> tuple:
        return self.tool, self.target


def _type(span) -> str:
    return getattr(span.span_type, "value", span.span_type)


def tool_failed(span) -> bool:
    return bool(span.tool_error or span.error or (span.status or "").lower() in _FAILED_STATUS)


def _arguments(span) -> tuple[str, dict]:
    """Tool name and arguments; a dispatcher call (e.g. CallDynamicTool) names the real
    tool in `toolName` and nests its arguments."""
    data = span.tool_input if isinstance(span.tool_input, dict) else {}
    name = span.tool_name or span.name or "tool"
    if isinstance(data.get("toolName"), str):
        name = data["toolName"]
        nested = data.get("arguments")
        data = nested if isinstance(nested, dict) else data
    return name, data


def _kind(name: str) -> str:
    n = re.sub(r"[^a-z]", "", name.lower())
    if any(w in n for w in _EDIT_WORDS) or n in _EDIT_NAMES:
        return "edit"
    if any(w in n for w in _SHELL_WORDS):
        return "command"
    return "tool"


def _first_str(data: dict, keys: tuple) -> Optional[str]:
    for key in keys:
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()
    return None


def _tool_action(span) -> Action:
    name, args = _arguments(span)
    kind = _kind(name)
    if kind == "command":
        target = _first_str(args, _COMMAND_KEYS)
    else:
        target = _first_str(args, _PATH_KEYS)
    if target is None and args:
        target = json.dumps(args, ensure_ascii=False, default=str)[:160]
    failed = tool_failed(span)
    normalized = re.sub(r"[^a-z]", "", name.lower())
    checks = (kind == "command" and bool(_CHECK_COMMAND.search(target or ""))) or any(
        w in normalized for w in _CHECK_TOOL_WORDS
    )
    return Action(
        tool=name, kind=kind, target=target, failed=failed,
        error=(span.tool_error or span.error) if failed else None,
        output=span.tool_output, verifies=checks and not failed,
    )


def actions(spans: list) -> list[Action]:
    """Tool calls and subagent runs in execution order. The root agent span of a run (no
    parent) is the run itself, not an action."""
    steps = [
        s for s in spans
        if _type(s) == "tool" or (_type(s) == "agent" and s.parent_span_id)
    ]
    steps.sort(key=lambda s: (s.timestamp, s.step_index))
    result = []
    for s in steps:
        if _type(s) == "tool":
            result.append(_tool_action(s))
        else:
            result.append(Action(
                tool=s.name or s.agent_name or "sous-agent", kind="subagent", target=None,
                failed=tool_failed(s), error=s.error, output=s.response, verifies=False,
            ))
    return result


def is_agentic(trace, spans: list) -> bool:
    """A run of an agent (Cursor, Claude Code, an SDK agent with tools) rather than a chatbot
    or Q&A call: generic answer metrics do not apply to it."""
    if (getattr(trace, "metadata", None) or {}).get("source") in TOOL_ORIGINS:
        return True
    return any(_type(s) == "tool" or _type(s) == "agent" for s in spans)


def files_changed(acts: list[Action]) -> list[str]:
    seen: dict[str, None] = {}
    for a in acts:
        if a.kind == "edit" and not a.failed and a.target:
            seen.setdefault(a.target, None)
    return list(seen)


def _line(i: int, a: Action) -> str:
    if a.kind == "subagent":
        head = f"{i}. [sous-agent] {a.tool}"
        tail = f" -> {' '.join((a.output or '').split())[:600]}" if a.output else ""
    else:
        head = f"{i}. [{a.kind}] {a.tool}" + (f" {a.target[:200]}" if a.target else "")
        if a.failed:
            tail = f" -> ERREUR: {' '.join((a.error or 'échec').split())[:200]}"
        elif a.output:
            tail = f" -> {' '.join(a.output.split())[:300]}"
        else:
            tail = ""
    return head + tail


_MODE = re.compile(
    r"\b(ask|plan|agent|debug) mode is (?:still )?active\b|\byou are now in (ask|plan|agent|debug) mode\b", re.I,
)
_MODE_ALIASES = {"chat": "ask"}

MODE_NOTES = {
    "ask": ("ASK mode (read-only): the agent was not allowed to modify files or run commands that change "
            "the system. Reading code, answering, explaining and proposing changes without applying them is "
            "the expected behaviour; asking the user to switch to Agent mode to apply them is correct."),
    "plan": "PLAN mode: the agent was expected to produce a plan, not to implement it.",
    "agent": "AGENT mode: the agent was allowed to modify files and run commands.",
    "debug": ("DEBUG mode: the agent was expected to find the cause of a bug; adding temporary logs, running "
              "the code and asking the user to reproduce the issue are expected steps before the fix."),
}


def run_mode(trace, request_text: Optional[str]) -> Optional[str]:
    """Mode the agent ran in: the latest mode reminder of the request (Cursor adds one to each
    turn), else the mode recorded by the integration."""
    found = None
    for match in _MODE.finditer(request_text or ""):
        found = (match.group(1) or match.group(2)).lower()
    if found:
        return found
    mode = (getattr(trace, "metadata", None) or {}).get("mode")
    if not isinstance(mode, str) or not mode:
        return None
    return _MODE_ALIASES.get(mode.lower(), mode.lower())


@dataclass
class Turn:
    """An earlier request of the same conversation (one trace each with Cursor)."""
    request: str
    reports: list             # (subagent name, result)
    answer: Optional[str]


def _flat(text: Optional[str], limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def earlier_turns_text(turns: list[Turn], limit: int = 5000) -> str:
    """Earlier turns, oldest first, for a judge; when too long the most recent are kept."""
    blocks = []
    for n, turn in enumerate(turns, 1):
        lines = [f"Turn {n} request: {_flat(turn.request, 300)}"]
        lines += [f"  - subagent {name} reported: {_flat(result, 800)}" for name, result in turn.reports if result]
        if turn.answer:
            lines.append(f"  - agent answer: {_flat(turn.answer, 800)}")
        blocks.append("\n".join(lines))
    kept, size = [], 0
    for block in reversed(blocks):
        if size + len(block) > limit:
            break
        kept.insert(0, block)
        size += len(block) + 1
    return "\n".join(kept)


def action_log(acts: list[Action], limit: int = 8000) -> str:
    """Numbered action list for a judge; when too long, the start and the end are kept."""
    lines = [_line(i, a) for i, a in enumerate(acts, 1)]
    if sum(len(line) + 1 for line in lines) <= limit:
        return "\n".join(lines)
    head, size = [], 0
    for line in lines:
        if size + len(line) > limit // 3:
            break
        head.append(line)
        size += len(line) + 1
    tail, size = [], 0
    for line in reversed(lines[len(head):]):
        if size + len(line) > limit - limit // 3:
            break
        tail.insert(0, line)
        size += len(line) + 1
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... ({omitted} actions omises) ...", *tail])
