"""
Harness recommendations: what to change around the agent rather than in its prompt.

The harness is everything a coding agent works through: the hooks that check its work, the
files it may read, the shell its commands run in, the size of what its tools send back, its
subagents, its permissions and the MCP servers it calls. Every finding is read from the tool
spans of agent runs and comes with the files to add for the agent that produced them: Cursor
hooks, rules and ignore file, Claude Code settings, hooks and subagents, or code for an agent
built with the SDK.

    verify      code changed, then the run ended without a passing test, lint or build
                -> a stop hook that runs the project's check and sends its errors back
    terminal    commands written for another shell, or for tools not installed
                -> a rule describing the terminal
    shell_read  cat / grep / Get-Content where the agent has Read, Grep and Glob
    output      tool outputs of tens of thousands of characters, paid again at every call
    generated   reads in node_modules, dist, .venv, lock files...
                -> .cursorignore, or Read denials in .claude/settings.json
    mcp         an MCP server failing (authentication, connection)
    blocked     commands refused by a permission or a hook, retried anyway
    edit_miss   replacements failing because the text to replace is not found or not unique
    explore     long explorations in the main agent's context instead of a subagent
    schema      SDK agents calling their tools with invalid arguments
    memory      (project) the same files reread in most runs -> a project map always loaded

collect() reads the signals of one run. Signals of many runs are added together, and
findings() turns them into recommendations worded for one trace or for a whole project.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

from beanie import PydanticObjectId

from evaluation.actions import _CHECK_COMMAND, _PATH_KEYS, _arguments, _tool_action
from models.span import Span
from models.trace import Trace

MAX_REFS = 10
OUTPUT_HEAD = 2_000                 # characters of a tool output scanned for shell errors
BIG_OUTPUT_CHARS = 15_000           # about 4 000 tokens
BIG_OUTPUT_TOTAL = 40_000
HEAVY_OUTPUT_TOTAL = 100_000
EXPLORATION_MIN_READS = 20
SHELL_READ_MIN = 3
REPEATED_MIN = 2
PROJECT_RUN_LIMIT = 200
PROJECT_MIN_RUNS = 3
MEMORY_MIN_RUNS = 5
MEMORY_MIN_FILE_RUNS = 3
MEMORY_FILE_SHARE = 0.3
MEMORY_MAX_FILES = 8
AGENT_SOURCES = ("cursor", "claude-code")

CODE_EXTS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".go", ".rs", ".java", ".kt",
    ".cs", ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".swift", ".scala",
})
_CHECK_BY_EXT = (
    ((".py",), "python -m pytest -q"),
    ((".ts", ".tsx", ".vue", ".svelte"), "npx tsc --noEmit"),
    ((".js", ".jsx", ".mjs", ".cjs"), "npm test"),
    ((".go",), "go test ./..."),
    ((".rs",), "cargo check"),
    ((".java", ".kt", ".scala"), "./gradlew test"),
    ((".cs",), "dotnet build"),
)

_DRIVE = re.compile(r"^/?[a-zA-Z]:[\\/]")
_POWERSHELL_HINT = re.compile(r"\$env:|\b(?:Get-ChildItem|Get-Content|Select-String|Select-Object|Remove-Item)\b", re.I)
_POWERSHELL_SYNTAX = re.compile(
    r"is not a valid statement separator|n'est pas un séparateur d'instruction valide"
    r"|ParserError|Unexpected token .{1,40}? in expression or statement|Jeton inattendu.{1,40}?dans l'expression"
    r"|A parameter cannot be found that matches parameter name|Impossible de trouver un paramètre correspondant",
    re.I,
)
_WINDOWS_MISSING = re.compile(r"is not recognized as|n'est pas reconnu", re.I)
_MISSING_COMMAND = re.compile(
    r"The term '([^']+)' is not recognized"
    r"|Le terme «\s*([^»]+?)\s*» n'est pas reconnu"
    r"|'([^']+)' is not recognized as an internal or external command"
    r"|'([^']+)' n'est pas reconnu en tant que commande interne"
    r"|(?:^|[\s:])([\w.-]+): (?:command )?not found",
    re.I,
)
_COMMAND_NAME = re.compile(r"^[\w.-]{1,30}$")
_NOT_COMMANDS = frozenset({"error", "fatal", "warning", "file", "path", "module", "npm", "err"})
_POWERSHELL_EQUIVALENT = {
    "grep": "Select-String", "head": "Select-Object -First 20", "tail": "Select-Object -Last 20",
    "touch": "New-Item", "export": "$env:NAME = \"value\"", "which": "Get-Command", "wc": "Measure-Object",
    "sed": "-replace", "rm": "Remove-Item -Recurse -Force", "find": "Get-ChildItem -Recurse",
}
_BLOCKED = re.compile(
    r"blocked|denied|not allowed|rejected|refused|refus|bloqu|interdit|requires approval|user declined", re.I,
)
_EDIT_MISS = re.compile(
    r"not found|not unique|no match|did not match|does not match|multiple (?:matches|occurrences)"
    r"|found \d+ (?:matches|occurrences)|introuvable|modified since|has been modified|has changed since"
    r"|must read|read it first",
    re.I,
)
_FILE_CHANGED = re.compile(r"modified since|has been modified|has changed since|must read|read it first", re.I)
_SCHEMA_ERROR = re.compile(
    r"validation error|invalid (?:argument|parameter|input|json)|missing (?:\d+ )?required|unexpected keyword"
    r"|required propert|is not of type|JSONDecodeError|TypeError",
    re.I,
)
_MCP_AUTH = re.compile(
    r"\b40[13]\b|unauthori[sz]ed|forbidden|authenticat|needs ?auth|token.{0,20}expired|invalid.{0,10}token"
    r"|log ?in required",
    re.I,
)
_MCP_DOWN = re.compile(
    r"ECONNREFUSED|not connected|connection (?:refused|closed|reset|error)|timed? ?out|\b50[234]\b|unavailable"
    r"|not running|disconnected",
    re.I,
)
_READ_COMMANDS = frozenset({
    "cat", "head", "tail", "less", "more", "grep", "egrep", "rg", "find", "findstr", "type", "ls", "dir", "tree",
    "get-content", "gc", "select-string", "sls", "get-childitem", "gci",
})
_MOVE_COMMANDS = frozenset({"cd", "pushd", "set-location", "sl"})
_GROUPED_COMMANDS = frozenset({
    "npm", "pnpm", "yarn", "npx", "bun", "python", "python3", "py", "git", "go", "cargo", "dotnet", "uv", "poetry",
    "pip", "docker", "make",
})
_EDGE = r"(?:^|[/\s\"'])"
_GENERATED = (
    (re.compile(_EDGE + r"node_modules/"), "node_modules/"),
    (re.compile(_EDGE + r"\.venv[\w.-]*/"), ".venv*/"),
    (re.compile(_EDGE + r"venv/"), "venv/"),
    (re.compile(_EDGE + r"site-packages/"), "site-packages/"),
    (re.compile(_EDGE + r"__pycache__/"), "__pycache__/"),
    (re.compile(_EDGE + r"\.pytest_cache/"), ".pytest_cache/"),
    (re.compile(_EDGE + r"dist/"), "dist/"),
    (re.compile(_EDGE + r"build/"), "build/"),
    (re.compile(_EDGE + r"\.next/"), ".next/"),
    (re.compile(_EDGE + r"coverage/"), "coverage/"),
    (re.compile(_EDGE + r"\.git/"), ".git/"),
    (re.compile(r"\.min\.(?:js|css)\b"), "*.min.*"),
    (re.compile(r"\.(?:js|css)\.map\b"), "*.map"),
)
_LOCK_FILE = re.compile(_EDGE + r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|uv\.lock|Cargo\.lock)\b")


@dataclass
class Signals:
    """What the tool spans of one or several runs show about the harness. Adding the signals
    of two sets of runs (add) gives the signals of their union. Lists hold (trace id, step
    reference) pairs."""
    runs: int = 0
    windows: bool = False
    edit_runs: int = 0
    unverified: list = field(default_factory=list)          # last edit of a run never checked after
    edited_exts: Counter = field(default_factory=Counter)
    checks_ok: Counter = field(default_factory=Counter)     # check commands that passed
    terminal: list = field(default_factory=list)            # commands failing because of the shell
    missing: Counter = field(default_factory=Counter)       # commands not installed
    failing_commands: Counter = field(default_factory=Counter)
    shell_reads: list = field(default_factory=list)
    shell_read_commands: Counter = field(default_factory=Counter)
    big_outputs: list = field(default_factory=list)
    big_chars: int = 0
    big_kinds: Counter = field(default_factory=Counter)     # command | read | tool
    generated: list = field(default_factory=list)
    generated_paths: Counter = field(default_factory=Counter)
    mcp_calls: Counter = field(default_factory=Counter)
    mcp_failed: dict = field(default_factory=dict)          # server -> [(hit, kind)]
    blocked: list = field(default_factory=list)
    blocked_commands: Counter = field(default_factory=Counter)
    blocked_checks: Counter = field(default_factory=Counter)
    edit_misses: list = field(default_factory=list)
    file_changed: bool = False
    explore: list = field(default_factory=list)
    explore_reads: int = 0
    schema_errors: list = field(default_factory=list)
    schema_tools: Counter = field(default_factory=Counter)
    files_read: Counter = field(default_factory=Counter)    # path -> runs that read it


def add(total: Signals, other: Signals) -> Signals:
    for f in fields(Signals):
        a, b = getattr(total, f.name), getattr(other, f.name)
        if isinstance(a, bool):
            setattr(total, f.name, a or b)
        elif isinstance(a, (int, list, Counter)):
            setattr(total, f.name, a + b)
        else:
            for key, items in b.items():
                a.setdefault(key, []).extend(items)
    return total


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------

def _type(span) -> str:
    return getattr(span.span_type, "value", span.span_type)


def _order(span) -> tuple:
    return (span.timestamp or datetime.min, span.step_index or 0)


def _status(trace) -> str:
    return str(getattr(trace.status, "value", trace.status) or "")


def _flat(text: Optional[str], limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def _ref(span, label: str, reason: Optional[str] = None) -> dict:
    return {"span_id": str(span.id), "step": span.step_index, "label": _flat(label, 120),
            "reason": _flat(reason, 300) or None}


def _int(n: int) -> str:
    return f"{n:,}"


def _output_len(span) -> int:
    return getattr(span, "output_chars", None) or len(span.tool_output or "")


def _relative(path: str, workspace: str) -> str:
    path = re.sub(r"^/([a-zA-Z]:)", r"\1", path.replace("\\", "/"))
    root = re.sub(r"^/([a-zA-Z]:)", r"\1", (workspace or "").replace("\\", "/")).rstrip("/")
    if root and path.lower().startswith(root.lower() + "/"):
        return path[len(root) + 1:]
    return path


def _extension(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _segments(command: str) -> list[str]:
    """Commands chained with ; && ||, each cut at its first pipe."""
    return [part.split("|", 1)[0].strip() for part in re.split(r"&&|\|\||;", command or "") if part.strip()]


def _program(segment: str) -> str:
    words = segment.lstrip("(& ").split()
    return words[0].lower() if words else ""


def _read_programs(command: str) -> list[str]:
    """File reading programs a command starts with (`cd app; grep x` counts, `npm test | tail` does not)."""
    programs = [_program(s) for s in _segments(command)]
    return [p for p in programs if p in _READ_COMMANDS]


def _command_name(command: str) -> str:
    """`npm run build`, `pytest`: what a permission would name, without the arguments."""
    for segment in _segments(command):
        words = segment.split()
        if not words or words[0].lower() in _MOVE_COMMANDS:
            continue
        check = _CHECK_COMMAND.search(segment)
        if check:
            return segment[:check.end()].strip()
        return " ".join(words[:2] if words[0].lower() in _GROUPED_COMMANDS else words[:1])
    return _flat(command, 40)


def _generated(text: str) -> set[str]:
    text = text.replace("\\", "/")
    found = {line for pattern, line in _GENERATED if pattern.search(text)}
    found.update(m.group(1) for m in _LOCK_FILE.finditer(text))
    return found


def _terminal_error(text: str) -> Optional[tuple[str, Optional[str]]]:
    """("missing", command) for a command not installed, ("syntax", None) for a command
    written for another shell, None otherwise."""
    match = _MISSING_COMMAND.search(text)
    if match:
        name = next(g for g in match.groups() if g).strip()
        if not _COMMAND_NAME.match(name) or name.lower() in _NOT_COMMANDS:
            return None
        return "missing", name
    if _POWERSHELL_SYNTAX.search(text):
        return "syntax", None
    return None


def _mcp_server(span) -> Optional[str]:
    raw = span.tool_input if isinstance(span.tool_input, dict) else {}
    namespace = raw.get("namespace")
    if isinstance(namespace, str) and namespace:
        return None if namespace == "cursor" else namespace
    name = span.tool_name or span.name or ""
    if name.startswith("mcp__"):
        parts = name.split("__")
        return parts[1] if len(parts) >= 3 else name
    if name.lower().startswith(("mcp:", "mcp_")):
        return name[4:] or name
    return None


def _mcp_kind(error: str) -> str:
    if _MCP_AUTH.search(error):
        return "auth"
    if _MCP_DOWN.search(error):
        return "down"
    return "error"


def _reads(name: str) -> bool:
    n = re.sub(r"[^a-z]", "", name.lower())
    if "web" in n or "lint" in n:
        return False
    return n in ("ls", "view") or any(w in n for w in ("read", "grep", "glob", "search", "listdir", "find"))


def collect(trace, spans: list) -> Signals:
    """Harness signals of one agent run, from its tool and subagent spans."""
    sig = Signals(runs=1)
    tid = str(trace.id)
    workspace = (getattr(trace, "metadata", None) or {}).get("workspace") or ""
    sig.windows = bool(_DRIVE.match(workspace))
    tools = sorted((s for s in spans if _type(s) == "tool"), key=_order)
    # A subagent is an agent span under another agent span: Claude Code's main agent has a
    # parent too, the session, which is not a span.
    agents = {s.sdk_span_id: s for s in spans if _type(s) == "agent" and getattr(s, "sdk_span_id", None)}
    delegates = any(a.parent_span_id in agents for a in agents.values())
    last_edit = last_check = None
    reads, read_refs, read_files, edited = 0, [], set(), set()

    for i, span in enumerate(tools):
        act = _tool_action(span)
        name, args = _arguments(span)
        label = f"{name} {act.target}" if act.target else name
        error = act.error or ""
        size = _output_len(span)
        if size >= BIG_OUTPUT_CHARS:
            kind = "command" if act.kind == "command" else "read" if _reads(name) else "tool"
            sig.big_outputs.append((tid, _ref(span, label, f"{_int(size)} characters")))
            sig.big_chars += size
            sig.big_kinds[kind] += 1

        server = _mcp_server(span)
        if server:
            sig.mcp_calls[server] += 1
            if act.failed:
                sig.mcp_failed.setdefault(server, []).append(((tid, _ref(span, label, error)), _mcp_kind(error)))
            continue

        if act.kind == "edit":
            path = act.target or ""
            sig.windows = sig.windows or bool(_DRIVE.match(path))
            if act.failed:
                if _EDIT_MISS.search(error):
                    sig.edit_misses.append((tid, _ref(span, label, error)))
                    sig.file_changed = sig.file_changed or bool(_FILE_CHANGED.search(error))
            elif _extension(path) in CODE_EXTS:
                last_edit = (i, span, label)
                edited.add(_extension(path))
            continue

        if act.verifies:
            last_check = i
            if act.kind == "command":
                sig.checks_ok[_flat(act.target, 160)] += 1

        if act.kind == "command":
            command = act.target or ""
            sig.windows = sig.windows or bool(_POWERSHELL_HINT.search(command))
            if act.failed and _BLOCKED.search(error):
                sig.blocked.append((tid, _ref(span, label, error)))
                sig.blocked_commands[_command_name(command)] += 1
                if _CHECK_COMMAND.search(command):
                    sig.blocked_checks[_command_name(command)] += 1
                continue
            text = f"{error}\n{(span.tool_output or '')[:OUTPUT_HEAD]}"
            problem = _terminal_error(text)
            if problem:
                sig.terminal.append((tid, _ref(span, label, error or (span.tool_output or "")[:300])))
                sig.failing_commands[_flat(command, 120)] += 1
                if problem[0] == "missing":
                    sig.missing[problem[1]] += 1
                sig.windows = sig.windows or bool(_POWERSHELL_SYNTAX.search(text) or _WINDOWS_MISSING.search(text))
            programs = _read_programs(command)
            if programs:
                reads += 1
                read_refs.append((tid, _ref(span, label)))
                sig.shell_reads.append((tid, _ref(span, label)))
                sig.shell_read_commands.update(programs)
                found = _generated(command)
                if found:
                    sig.generated.append((tid, _ref(span, label)))
                    sig.generated_paths.update(found)
            continue

        if act.failed and _SCHEMA_ERROR.search(error):
            sig.schema_errors.append((tid, _ref(span, label, error)))
            sig.schema_tools[name] += 1
        if _reads(name):
            reads += 1
            read_refs.append((tid, _ref(span, label)))
            path = next((args[k] for k in (*_PATH_KEYS, "target_directory", "directory")
                         if isinstance(args.get(k), str) and args[k].strip()), None)
            if path:
                sig.windows = sig.windows or bool(_DRIVE.match(path))
                found = _generated(path.rstrip("/\\") + "/")
                if found:
                    sig.generated.append((tid, _ref(span, label)))
                    sig.generated_paths.update(found)
                if "read" in name.lower() and not found:
                    read_files.add(_relative(path, workspace))

    if last_edit is not None:
        sig.edit_runs = 1
        sig.edited_exts.update(edited)
        if (last_check is None or last_check < last_edit[0]) and _status(trace) != "running":
            sig.unverified.append((tid, _ref(last_edit[1], last_edit[2], "Last code change, with no passing "
                                                                         "check afterwards.")))
    if reads >= EXPLORATION_MIN_READS and not delegates:
        sig.explore.extend(read_refs[:MAX_REFS])
        sig.explore_reads += reads
    sig.files_read.update(read_files)
    return sig


# ---------------------------------------------------------------------------
# Files to add
# ---------------------------------------------------------------------------

VERIFY_SCRIPT = '''"""End-of-task hook suggested by AItrajectIQ. When files have changed since the last
check, runs CHECK and sends its errors back to the agent, which fixes them before handing
back control. Works with Cursor (stop hook) and Claude Code (Stop hook)."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

CHECK = __CHECK__
MAX_ROUNDS = 2          # retries per conversation before letting the agent finish
MAX_OUTPUT = 3000
STATE = os.path.join(tempfile.gettempdir(), "aitrajectiq-verify.json")


def run(args, timeout=60):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def load():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(state):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def main():
    event = json.load(sys.stdin)
    claude = event.get("hook_event_name") == "Stop"
    if not claude and event.get("status", "completed") != "completed":
        return {}
    status = run(["git", "status", "--porcelain"])
    if status.returncode != 0 or not status.stdout.strip():
        return {}
    fingerprint = hashlib.sha1((status.stdout + run(["git", "diff"]).stdout).encode()).hexdigest()
    session = str(event.get("session_id") or event.get("conversation_id") or "default")
    state = load()
    entry = state.get(session) or {}
    if entry.get("checked") == fingerprint:
        return {}
    shell = ["powershell", "-NoProfile", "-Command", CHECK] if os.name == "nt" else ["sh", "-c", CHECK]
    try:
        result = run(shell, timeout=900)
    except subprocess.TimeoutExpired:
        return {}
    rounds = entry.get("rounds", 0)
    if result.returncode == 0 or rounds >= MAX_ROUNDS:
        state[session] = {"checked": fingerprint, "rounds": 0}
        save(state)
        return {}
    state[session] = {"checked": None, "rounds": rounds + 1}
    save(state)
    output = (result.stdout + result.stderr).strip()[-MAX_OUTPUT:]
    message = f"The check `{CHECK}` fails after your changes. Fix it before finishing:\\n\\n{output}"
    return {"decision": "block", "reason": message} if claude else {"followup_message": message}


if __name__ == "__main__":
    print(json.dumps(main()))
'''

SHELL_READ_GUARD = '''"""PreToolUse hook suggested by AItrajectIQ: refuses commands that read or search files, so
that Claude uses Read, Grep and Glob (bounded output, no permission prompt)."""

import json
import re
import sys

READ_COMMANDS = re.compile(r"^\\s*(cat|head|tail|less|more|grep|rg|find|findstr|Get-Content|Select-String)\\b", re.I)

command = (json.load(sys.stdin).get("tool_input") or {}).get("command") or ""
if READ_COMMANDS.match(command) and "|" not in command:
    print("Use the Read, Grep and Glob tools to read and search files.", file=sys.stderr)
    sys.exit(2)
'''

EXPLORER_AGENT = """---
name: explorer
description: Explores the code read-only to answer a precise question (where is X, how does Y work) and returns a short summary with the paths. Use it as soon as a search spans more than five files.
tools: Read, Grep, Glob
model: haiku
---

You explore the code without changing anything. Return:
- the useful files and line numbers;
- the answer to the question in ten lines at most.
Don't copy long code excerpts.
"""

CLIP_OUTPUT_CODE = (
    "MAX_TOOL_OUTPUT = 8_000\n\n"
    "def clip_tool_output(text: str) -> str:\n"
    "    \"\"\"Start and end of a long tool output, before sending it back to the model.\"\"\"\n"
    "    if len(text) <= MAX_TOOL_OUTPUT:\n"
    "        return text\n"
    "    cut = len(text) - 7_000\n"
    "    return f\"{text[:2_000]}\\n... [{cut} characters cut] ...\\n{text[-5_000:]}\"\n"
)


def _file(kind: str, file: Optional[str], content: str, label: Optional[str] = None) -> dict:
    return {"kind": kind, "label": label or f"Create: {file}", "file": file, "content": content}


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _rule(name: str, title: str, lines: list[str], target: str) -> dict:
    bullets = "\n".join(f"- {line}" for line in lines)
    if target == "cursor":
        path = f".cursor/rules/{name}.mdc"
        return _file("rule", path, f"---\ndescription: {title}\nalwaysApply: true\n---\n\n{bullets}\n")
    if target == "claude-code":
        return _file("rule", "CLAUDE.md", f"## {title}\n\n{bullets}\n", "Add to CLAUDE.md")
    return _file("prompt", None, bullets, "Add to the system prompt")


def _python(windows: bool) -> str:
    return "py -3" if windows else "python3"


def _verify_files(target: str, check: str, windows: bool) -> tuple[Optional[dict], list[dict]]:
    script = VERIFY_SCRIPT.replace("__CHECK__", repr(check))
    if target == "cursor":
        config = {"version": 1, "hooks": {"stop": [
            {"command": f"{_python(windows)} .cursor/hooks/verify.py", "timeout": 900}]}}
        return (_file("config", ".cursor/hooks.json", _json(config), "Add to .cursor/hooks.json"),
                [_file("file", ".cursor/hooks/verify.py", script)])
    if target == "claude-code":
        config = {"hooks": {"Stop": [{"hooks": [{
            "type": "command", "command": 'python "$CLAUDE_PROJECT_DIR/.claude/hooks/verify.py"', "timeout": 900}]}]}}
        return (_file("config", ".claude/settings.json", _json(config), "Add to .claude/settings.json"),
                [_file("file", ".claude/hooks/verify.py", script)])
    code = (
        "import subprocess\n\n"
        f"CHECK = {check!r}\n\n"
        "def verify() -> str | None:\n"
        "    \"\"\"Errors of the check, or None when it passes.\"\"\"\n"
        "    r = subprocess.run(CHECK, shell=True, capture_output=True, text=True, timeout=900)\n"
        "    return None if r.returncode == 0 else (r.stdout + r.stderr)[-3000:]\n\n"
        "# before the agent's final answer\n"
        "if (errors := verify()):\n"
        "    messages.append({\"role\": \"user\", \"content\": f\"The check fails, fix it:\\n{errors}\"})\n"
    )
    return _file("code", None, code, "Check before the final answer (Python)"), []


def _deny_rule(line: str) -> str:
    return f"Read(**/{line}**)" if line.endswith("/") else f"Read(**/{line})"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _runs(hits: list) -> int:
    return len({tid for tid, _ in hits})


def _examples(hits: list, limit: int = 3) -> list:
    seen, out = set(), []
    for tid, ref in hits:
        if tid not in seen:
            seen.add(tid)
            out.append({"trace_id": tid, "reason": ref.get("reason") or ref.get("label")})
        if len(out) == limit:
            break
    return out


def _enough(hits: list, sig: Signals, scope: str, trace_min: int, project_min: int = 2,
            project_share: float = 0.0) -> bool:
    if scope == "trace":
        return len(hits) >= trace_min
    runs = _runs(hits)
    return runs >= project_min and runs / max(sig.runs, 1) >= project_share


def _in_runs(hits: list, sig: Signals, scope: str) -> str:
    return f" In {_runs(hits)} of the {sig.runs} runs analyzed." if scope == "project" else ""


def _finding(key: str, severity: str, title: str, detail: str, action: str, hits: list, scope: str,
             evidence: dict, fix: Optional[dict] = None, extra: Optional[list] = None) -> dict:
    return {
        "id": f"harness-{key}",
        "category": "harness",
        "severity": severity,
        "title": title,
        "detail": detail,
        "action": action,
        "savings_usd": None,
        "evidence": evidence,
        "examples": _examples(hits) if scope == "project" else [],
        "spans": [ref for _, ref in hits][:MAX_REFS] if scope == "trace" else [],
        "fix": fix,
        "extra_fixes": extra or [],
    }


def _tool(target: str) -> str:
    return {"cursor": "Cursor", "claude-code": "Claude Code"}.get(target, "the agent")


def check_command(sig: Signals) -> tuple[str, bool]:
    """The check to run after changes: the one that passed most often in the runs, else one
    guessed from the edited files. Second value: whether it was seen in the runs."""
    if sig.checks_ok:
        return sig.checks_ok.most_common(1)[0][0], True
    for ext, _ in sig.edited_exts.most_common():
        for exts, command in _CHECK_BY_EXT:
            if ext in exts:
                return command, False
    return "npm test", False


def _verify(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.unverified
    if not hits:
        return None
    if scope == "project":
        if sig.edit_runs < PROJECT_MIN_RUNS or _runs(hits) / sig.edit_runs < 0.2:
            return None
        share = _runs(hits) / sig.edit_runs
        severity = "high" if share >= 0.5 else "medium"
        detail = (f"{_runs(hits)} of the {sig.edit_runs} runs that change code end without a passing test, lint "
                  f"or build after the last change.")
    else:
        severity = "medium"
        detail = "The agent changed code, then handed back control without a passing test, lint or build afterwards."
    check, seen = check_command(sig)
    origin = "already run successfully in your runs" if seen else "inferred from the changed files, adjust it"
    detail += f" Check {origin}: {check}."
    action = {
        "cursor": ("Add a stop hook: if files have changed, it runs the check and sends the errors back to the "
                   "agent, which fixes them before handing back control (at most 2 retries)."),
        "claude-code": ("Add a Stop hook: if files have changed, it runs the check and keeps Claude from "
                        "finishing while it fails (at most 2 retries)."),
    }.get(target, "Run the check before the final answer and send its errors back to the model.")
    fix, extra = _verify_files(target, check, sig.windows)
    return _finding("verify", severity, "Code handed back unverified", detail, action, hits, scope,
                    {"runs_with_edits": sig.edit_runs, "unverified_runs": _runs(hits), "check": check},
                    fix, extra)


def _terminal(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.terminal
    if not _enough(hits, sig, scope, 1, 1):
        return None
    lines, causes = [], []
    if sig.windows:
        lines.append("The terminal is PowerShell on Windows: chain commands with `;` (not `&&`), set a "
                     "variable with `$env:NAME = \"value\"` and keep the end of an output with "
                     "`| Select-Object -Last 50`.")
        causes.append("syntax of another shell")
    missing = [name for name, _ in sig.missing.most_common(8)]
    if missing:
        line = f"These commands don't exist here: {', '.join(f'`{m}`' for m in missing)}."
        equivalents = [f"`{m}` → `{_POWERSHELL_EQUIVALENT[m]}`" for m in missing
                       if sig.windows and m in _POWERSHELL_EQUIVALENT]
        if equivalents:
            line += f" Equivalents: {', '.join(equivalents)}."
        lines.append(line)
        causes.append(f"missing command(s): {', '.join(missing[:4])}")
    if not lines:
        return None
    n = len(hits)
    severity = (("high" if _runs(hits) >= PROJECT_MIN_RUNS else "medium") if scope == "project"
                else "medium" if n >= 2 else "low")
    return _finding(
        "terminal", severity, "Commands unsuited to the terminal",
        f"{n} command(s) failed because of the terminal ({'; '.join(causes)}).{_in_runs(hits, sig, scope)}",
        "Describe the terminal in an always-applied rule: the agent stops running commands that can't work "
        "here, and stops wasting steps on them.",
        hits, scope, {"failed_commands": n, "windows": "yes" if sig.windows else "no"},
        _rule("terminal", "Project terminal", lines, target),
    )


def _shell_reads(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.shell_reads
    if target not in AGENT_SOURCES or not _enough(hits, sig, scope, SHELL_READ_MIN, PROJECT_MIN_RUNS, 0.2):
        return None
    programs = [p for p, _ in sig.shell_read_commands.most_common(5)]
    extra = []
    action = ("Ask for the reading tools: their output is bounded and structured, and they need no "
              "permission.")
    if target == "claude-code":
        config = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{
            "type": "command", "command": 'python "$CLAUDE_PROJECT_DIR/.claude/hooks/shell_read_guard.py"'}]}]}}
        extra = [_file("config", ".claude/settings.json", _json(config),
                       "To enforce it: add to .claude/settings.json"),
                 _file("file", ".claude/hooks/shell_read_guard.py", SHELL_READ_GUARD)]
        action += " To enforce it, add the hook that refuses these commands."
    return _finding(
        "shell-read", "medium" if len(hits) >= 2 * SHELL_READ_MIN else "low",
        "The terminal is used to read files",
        f"{len(hits)} commands ({', '.join(programs)}) read or search files even though "
        f"{_tool(target)} has Read, Grep and Glob.{_in_runs(hits, sig, scope)}",
        action, hits, scope, {"commands": len(hits)},
        _rule("tools", "Reading tools", [
            "To read, search or list files, use Read, Grep and Glob, never cat, grep, find, "
            "Get-Content or Select-String in the terminal."], target),
        extra,
    )


def _outputs(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.big_outputs
    if scope == "trace":
        if len(hits) < 2 and sig.big_chars < BIG_OUTPUT_TOTAL:
            return None
    elif not _enough(hits, sig, scope, 0, 2, 0.2):
        return None
    tokens = sig.big_chars // 4
    detail = (f"{len(hits)} tool outputs over {_int(BIG_OUTPUT_CHARS)} characters, about "
              f"{_int(tokens)} tokens sent to the model and paid again on every following call."
              f"{_in_runs(hits, sig, scope)}")
    evidence = {"outputs": len(hits), "tokens_estimate": tokens}
    severity = "medium" if scope == "project" or sig.big_chars >= HEAVY_OUTPUT_TOTAL else "low"
    if target not in AGENT_SOURCES:
        return _finding("output", severity, "Tool outputs too long", detail,
                        "Clip long tool outputs before sending them back to the model (start and end).",
                        hits, scope, evidence, _file("code", None, CLIP_OUTPUT_CODE, "Tool outputs (Python)"))
    lines = []
    if sig.big_kinds["command"]:
        tail = "| Select-Object -Last 60" if sig.windows else "| tail -n 60"
        lines.append(f"Run tests and builds in quiet mode (`pytest -q --tb=short`, `npm run build --silent`) and "
                     f"keep only the end of a long output (`{tail}`).")
    if sig.big_kinds["read"]:
        lines.append("Read large files by line ranges instead of reading them whole.")
    if sig.big_kinds["tool"]:
        lines.append("Ask tools for filtered or paginated results rather than full lists.")
    return _finding("output", severity, "Tool outputs too long", detail,
                    "Have the tools produce short outputs: context is what costs, on every call.",
                    hits, scope, evidence, _rule("outputs", "Short outputs", lines, target))


def _generated_files(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.generated
    if target not in AGENT_SOURCES or not _enough(hits, sig, scope, REPEATED_MIN):
        return None
    paths = [p for p, _ in sig.generated_paths.most_common(10)]
    if target == "cursor":
        fix = _file("config", ".cursorignore", "\n".join(paths) + "\n", "Add to .cursorignore")
        action = "Exclude them with .cursorignore: the agent and the index no longer see them."
    else:
        fix = _file("config", ".claude/settings.json",
                    _json({"permissions": {"deny": [_deny_rule(p) for p in paths]}}),
                    "Add to .claude/settings.json")
        action = "Deny reading them in .claude/settings.json."
    return _finding(
        "generated", "medium" if len(hits) >= 5 else "low", "The agent reads generated files",
        f"{len(hits)} reads in {', '.join(paths)}: dependencies and generated files that fill the "
        f"context without teaching the agent anything.{_in_runs(hits, sig, scope)}",
        action, hits, scope, {"reads": len(hits)}, fix,
    )


_MCP_CAUSE = {"auth": "authentication refused", "down": "server unreachable", "error": "tool errors"}


def _mcp(sig: Signals, target: str, scope: str) -> list[dict]:
    out = []
    for server, failures in sig.mcp_failed.items():
        if len(failures) < REPEATED_MIN:
            continue
        calls = sig.mcp_calls[server]
        kind = Counter(k for _, k in failures).most_common(1)[0][0]
        action = {
            "auth": f"Reconnect the MCP server {server}: its authentication has expired and every refused call "
                    f"costs the agent a step.",
            "down": f"Check that the MCP server {server} is running, or disable it for this project: its tools "
                    f"stay described in the context of every call.",
        }.get(kind, f"Read the error of the server {server}; if it isn't used in this project, disable it.")
        fix = None
        if target == "claude-code":
            fix = _file("code", None, f"claude mcp get {server}\n# if it isn't used in this project:\n"
                                      f"claude mcp remove {server}\n", "Claude Code commands")
        hits = [hit for hit, _ in failures]
        out.append(_finding(
            f"mcp:{server}", "high" if len(failures) == calls else "medium", f"MCP server {server} failing",
            f"{len(failures)} of the {calls} calls to {server} failed ({_MCP_CAUSE[kind]}).{_in_runs(hits, sig, scope)}",
            action, hits, scope, {"calls": calls, "failed": len(failures)}, fix,
        ))
    return out


def _blocked(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.blocked
    if not _enough(hits, sig, scope, REPEATED_MIN):
        return None
    names = [c for c, _ in sig.blocked_commands.most_common(6)]
    checks = [c for c, _ in sig.blocked_checks.most_common(4)]
    listed = ", ".join(f"`{c}`" for c in names)
    rule = _rule("commands", "Blocked commands", [
        f"These commands are blocked in this project: {listed}. Don't retry them: give them to the user and "
        f"continue with what you can verify."], target)
    extra = []
    if checks:
        title = "The agent can't run its checks"
        action = "Allow the check commands (tests, build, lint): without them the agent hands back unverified code."
        if target == "claude-code":
            extra = [_file("config", ".claude/settings.json",
                           _json({"permissions": {"allow": [f"Bash({c}:*)" for c in checks]}}),
                           "Allow the checks: add to .claude/settings.json")]
        elif target == "cursor":
            action += f" In Cursor, add them to the allowed commands: {', '.join(checks)}."
        action += " Tell the agent which other commands are blocked."
    else:
        title = "Blocked commands retried"
        action = "Tell the agent which commands are blocked, so it doesn't retry them."
    return _finding(
        "blocked", "high" if checks else "medium", title,
        f"{len(hits)} commands refused by a permission or a hook ({listed}).{_in_runs(hits, sig, scope)}",
        action, hits, scope, {"blocked": len(hits)}, rule, extra,
    )


def _edit_misses(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.edit_misses
    if not _enough(hits, sig, scope, REPEATED_MIN, 2, 0.1):
        return None
    lines = ["Right before editing a file, reread the exact area and copy the text to replace verbatim, with "
             "two or three surrounding lines so that it is unique."]
    detail = f"{len(hits)} edits failed: text to replace not found or ambiguous."
    action = "Ask the agent to reread the area right before editing it."
    if sig.file_changed:
        lines.append("After a failed replacement, reread the file: it may have been reformatted in the meantime.")
        detail += " Some files had changed since they were read: a formatter or another process rewrites them."
        action += " If a formatter runs on save, run it in a post-edit hook instead."
    return _finding("edit-miss", "medium", "Failed file edits", detail + _in_runs(hits, sig, scope),
                    action, hits, scope, {"failed_edits": len(hits)},
                    _rule("edits", "Editing a file", lines, target))


def _explore(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.explore
    if not _enough(hits, sig, scope, 1, PROJECT_MIN_RUNS, 0.2):
        return None
    runs = _runs(hits)
    reads = sig.explore_reads // max(runs, 1)
    detail = (f"{reads} reads and searches on average in the main agent's context, with no subagent: "
              f"everything it reads stays there and is paid again on every call.{_in_runs(hits, sig, scope)}")
    if target == "claude-code":
        fix = _file("file", ".claude/agents/explorer.md", EXPLORER_AGENT)
        action = ("Add a read-only exploration subagent on Haiku: Claude hands it the searches and keeps only "
                  "its summary.")
    elif target == "cursor":
        fix = _rule("exploration", "Exploration", [
            "For a question that requires going through more than five files, launch an explore subagent and "
            "keep only its summary (paths and conclusions)."], target)
        action = "Hand broad searches to an explore subagent: only its summary enters the context."
    else:
        fix = None
        action = ("Hand searches to a subagent on a cheaper model, which returns only a summary to the main "
                  "agent.")
    return _finding("explore", "medium", "Exploration in the main context", detail, action, hits, scope,
                    {"reads": reads}, fix)


def _schema(sig: Signals, target: str, scope: str) -> Optional[dict]:
    hits = sig.schema_errors
    if target in AGENT_SOURCES or not _enough(hits, sig, scope, REPEATED_MIN):
        return None
    tool = sig.schema_tools.most_common(1)[0][0]
    code = _json({
        "type": "function",
        "function": {
            "name": tool,
            "description": "What the tool does, when to call it, and the exact format of each parameter.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "..."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    })
    return _finding(
        "schema", "medium", "Invalid tool arguments",
        f"{len(hits)} tool calls rejected for invalid arguments (types, missing fields), mostly "
        f"{tool}.{_in_runs(hits, sig, scope)}",
        "Declare strict schemas and describe each parameter: the provider then guarantees valid "
        "arguments.",
        hits, scope, {"failed_calls": len(hits)}, _file("code", None, code, f"Strict schema for {tool} (OpenAI)"),
    )


def _memory(sig: Signals, target: str) -> Optional[dict]:
    if target not in AGENT_SOURCES or sig.runs < MEMORY_MIN_RUNS:
        return None
    files = [(path, n) for path, n in sig.files_read.most_common(MEMORY_MAX_FILES)
             if n >= MEMORY_MIN_FILE_RUNS and n / sig.runs >= MEMORY_FILE_SHARE]
    if not files:
        return None
    sections = ["## Key files", "Reread in most tasks: start from them instead of exploring the repository.",
                *(f"- `{path}`: role to fill in (read in {n} of {sig.runs} runs)" for path, n in files)]
    checks = [c for c, _ in sig.checks_ok.most_common(3)]
    if checks:
        sections += ["", "## Checking your work", *(f"- `{c}`" for c in checks)]
    failing = [c for c, _ in sig.failing_commands.most_common(3)]
    if failing:
        sections += ["", "## Commands that fail here", *(f"- `{c}`" for c in failing)]
    body = "\n".join(sections) + "\n"
    if target == "cursor":
        fix = _file("rule", ".cursor/rules/project.mdc",
                    f"---\ndescription: Project map, learned from the agent's runs\nalwaysApply: true\n---\n\n{body}")
    else:
        fix = _file("rule", "CLAUDE.md", f"# Project map\n\n{body}", "Add to CLAUDE.md")
    top, n = files[0]
    return _finding(
        "memory", "medium", "Every run rediscovers the project",
        f"{len(files)} files are reread in at least {round(MEMORY_FILE_SHARE * 100)}% of the {sig.runs} runs, "
        f"including {top} ({n} runs).",
        "Load a project map on every run: the key files with their role, the commands that work and the "
        "ones that fail. Fill in the role of each file in one line.",
        [], "project", {"runs": sig.runs, "central_files": len(files)}, fix,
    )


def findings(sig: Signals, target: str, scope: str = "trace") -> list[dict]:
    """Harness recommendations from the signals of one run (scope "trace") or of the runs of
    a project (scope "project"), with their files written for `target`."""
    out = [
        _verify(sig, target, scope),
        _blocked(sig, target, scope),
        _terminal(sig, target, scope),
        *_mcp(sig, target, scope),
        _generated_files(sig, target, scope),
        _outputs(sig, target, scope),
        _explore(sig, target, scope),
        _edit_misses(sig, target, scope),
        _shell_reads(sig, target, scope),
        _schema(sig, target, scope),
        _memory(sig, target) if scope == "project" else None,
    ]
    return [f for f in out if f]


def analyze(trace, spans: list, target: str, notes: Optional[dict] = None) -> list[dict]:
    """Harness recommendations of one agent run; `notes` gets a line per step concerned."""
    result = findings(collect(trace, spans), target, "trace")
    if notes is not None:
        for f in result:
            for ref in f["spans"]:
                notes[ref["span_id"]].append(f"{f['title']}.")
    return result


# ---------------------------------------------------------------------------
# A project's runs
# ---------------------------------------------------------------------------

_SPAN_FIELDS = dict(
    name=None, tool_name=None, tool_input=None, tool_output=None, tool_error=None, error=None, status=None,
    step_index=0, timestamp=None, parent_span_id=None, sdk_span_id=None, span_type=None, output_chars=0,
)
_HEAVY_INPUTS = ("contents", "content", "new_string", "old_string", "arguments.contents", "arguments.content")


async def project_findings(pid: PydanticObjectId, since: datetime, target: str) -> list[dict]:
    """Harness recommendations over the latest agent runs of a project."""
    traces = await Trace.get_motor_collection().find(
        {"project_id": pid, "started_at": {"$gte": since},
         "$or": [{"total_tool_spans": {"$gt": 0}}, {"metadata.source": {"$in": list(AGENT_SOURCES)}}]},
        {"metadata.source": 1, "metadata.workspace": 1, "status": 1},
    ).sort("started_at", -1).limit(PROJECT_RUN_LIMIT).to_list(None)
    if not traces:
        return []
    is_text = {"$eq": [{"$type": "$tool_output"}, "string"]}
    docs = await Span.aggregate([
        {"$match": {"trace_id": {"$in": [t["_id"] for t in traces]}, "span_type": {"$in": ["tool", "agent"]}}},
        {"$addFields": {
            "output_chars": {"$cond": [is_text, {"$strLenCP": "$tool_output"}, 0]},
            "output_head": {"$cond": [is_text, {"$substrCP": ["$tool_output", 0, OUTPUT_HEAD]}, None]},
        }},
        {"$project": {"tool_output": 0, "messages_in": 0, "response": 0, "metrics": 0,
                      **{f"tool_input.{key}": 0 for key in _HEAVY_INPUTS}}},
    ]).to_list()
    spans: dict = defaultdict(list)
    for doc in docs:
        spans[doc["trace_id"]].append(SimpleNamespace(**{
            **_SPAN_FIELDS, **{k: v for k, v in doc.items() if k in _SPAN_FIELDS},
            "id": doc["_id"], "tool_output": doc.get("output_head"),
        }))
    total = Signals()
    for t in traces:
        run = SimpleNamespace(id=t["_id"], metadata=t.get("metadata") or {}, status=t.get("status"))
        add(total, collect(run, spans.get(t["_id"], [])))
    return findings(total, target, "project")
