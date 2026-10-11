"""
Cursor agent → AItrajectIQ traces, through Cursor hooks.

One trace per agent run (Cursor `generation_id`): the prompt opens it, every tool call
becomes a tool span, subagents become agent spans, the agent's answer an LLM span,
and `stop` closes it. Tokens come from `afterAgentResponse` / `stop` (input, output, cache
read and cache write), so costs account for prompt caching.

Standard library only: Cursor runs it with whatever Python is on the PATH.

    ~/.cursor/hooks.json
    {
      "version": 1,
      "hooks": {
        "beforeSubmitPrompt": [{"command": "py -3 <path>/cursor_hook.py", "timeout": 5}],
        "postToolUse":        [{"command": "py -3 <path>/cursor_hook.py", "timeout": 5}],
        ...same command for postToolUseFailure, subagentStart, subagentStop,
           afterAgentResponse, stop
      }
    }

Configuration: AITRAJECTIQ_API_KEY and AITRAJECTIQ_BASE_URL (default http://localhost:8000),
from the environment or from ~/.aitrajectiq/config.json ({"api_key": ..., "base_url": ...}).
The hook never blocks the agent: any failure is written to ~/.aitrajectiq/cursor_hook.log.
"""

import json
import os
import re
import sys
import tempfile
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HTTP_TIMEOUT_S = 4.0
MAX_TEXT = 20_000
PROMPT_PREVIEW = 80
STATE_TTL_S = 24 * 3600
ATTACH_WINDOW_S = 30 * 60      # an unknown run id joins a trace active this recently
SUBAGENT_ATTACH_S = 120        # a subagent's first event follows its launch within this delay
HOME = os.path.join(os.path.expanduser("~"), ".aitrajectiq")
STATE_DIR = os.path.join(tempfile.gettempdir(), "aitrajectiq-cursor")
STOP_STATUS = {"completed": "completed", "error": "failed", "aborted": "failed"}


def _config() -> dict:
    conf = {}
    path = os.path.join(HOME, "config.json")
    try:
        # utf-8-sig: Windows PowerShell writes UTF-8 files with a BOM.
        with open(path, encoding="utf-8-sig") as f:
            conf = json.load(f)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        _log(f"cannot read {path}: {e}")
    return {
        "api_key": os.environ.get("AITRAJECTIQ_API_KEY") or conf.get("api_key"),
        "base_url": (os.environ.get("AITRAJECTIQ_BASE_URL") or conf.get("base_url") or "http://localhost:8000").rstrip("/"),
    }


def _log(message: str) -> None:
    try:
        os.makedirs(HOME, exist_ok=True)
        with open(os.path.join(HOME, "cursor_hook.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def _post(conf: dict, path: str, payload: dict):
    req = Request(
        conf["base_url"] + path,
        data=json.dumps(payload, default=str).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": conf["api_key"]},
        method="POST",
    )
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read() or b"{}")
    except HTTPError as e:
        _log(f"POST {path} -> HTTP {e.code}: {e.read()[:300]!r}")
    except (URLError, OSError, ValueError) as e:
        _log(f"POST {path} failed: {e}")
    return None


# --- Run state, shared by the hook processes of one agent run -------------------

def _state_path(run_id: str) -> str:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_")[:80] or "default"
    return os.path.join(STATE_DIR, safe + ".json")


class _Lock:
    """Cross-process lock on a run (tool hooks of one run can fire concurrently)."""

    def __init__(self, path: str):
        self.path = path + ".lock"

    def __enter__(self):
        deadline = time.time() + 4
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
                return self
            except (FileExistsError, PermissionError):   # Windows: lock file being deleted
                try:
                    if time.time() - os.path.getmtime(self.path) > 10:
                        os.remove(self.path)          # left behind by a killed hook
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    return self                       # better a race than a stuck agent
                time.sleep(0.02)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save(path: str, state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:               # Windows: another hook is reading the file
            time.sleep(0.05)
    os.replace(tmp, path)


def _cleanup_old_states() -> None:
    try:
        for name in os.listdir(STATE_DIR):
            full = os.path.join(STATE_DIR, name)
            if time.time() - os.path.getmtime(full) > STATE_TTL_S:
                os.remove(full)
    except OSError:
        pass


# --- Event → span ----------------------------------------------------------------

def _clip(value, limit: int = MAX_TEXT):
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text[:limit]


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {} if value in (None, "") else {"input": _clip(value, 2000)}


def _first(event: dict, *keys):
    for key in keys:
        value = event.get(key)
        if value not in (None, ""):
            return value
    return None


def _int(value):
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _duration_ms(event: dict):
    value = _first(event, "duration_ms", "duration")
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _span(conf: dict, state: dict, **fields) -> None:
    state["step"] = state.get("step", 0) + 1
    payload = {
        "trace_id": state["trace_id"],
        "span_id": f"span-{uuid.uuid4().hex[:12]}",
        "step_index": state["step"],
        **fields,
    }
    _post(conf, "/ingest/span", payload)


def _mode(event: dict):
    return _first(event, "composer_mode", "mode", "agent_mode", "chat_mode", "unified_mode")


def _start_run(conf: dict, event: dict) -> dict:
    prompt = event.get("prompt") or ""
    preview = " ".join(str(prompt).split())
    notified = _notified_tasks(str(prompt))
    if notified:
        preview = "notification · " + ", ".join(t.get("title") or t["tool_call_id"] for t in notified)
    elif "<user_query>" in str(prompt):
        query = re.search(r"<user_query>(.*?)(?:</user_query>|\Z)", str(prompt), re.S)
        preview = " ".join(query.group(1).split()) if query else preview
    name = "Cursor: " + (preview[:PROMPT_PREVIEW] + "…" if len(preview) > PROMPT_PREVIEW else preview or "agent")
    metadata = {
        "source": "cursor",
        "conversation_id": event.get("conversation_id"),
        "generation_id": event.get("generation_id"),
        "model": event.get("model"),
        "mode": _mode(event),
        "cursor_version": event.get("cursor_version"),
        "workspace": (event.get("workspace_roots") or [None])[0],
    }
    started = _post(conf, "/ingest/trace/start", {"name": name, "metadata": metadata})
    if not started or not started.get("trace_id"):
        if started is not None:
            _log(f"trace/start answered without trace_id: {started!r}")
        return {}
    _log(f"trace {started['trace_id']} started for generation {event.get('generation_id')} ({conf['base_url']})")
    state = {
        "trace_id": started["trace_id"],
        "root_span": f"span-{uuid.uuid4().hex[:12]}",
        "prompt": _clip(prompt),
        "model": event.get("model"),
        "mode": _mode(event),
        "conversation_id": event.get("conversation_id"),
        "started": time.time(),
        "subagents": {},
        "step": 0,
    }
    _root_span(conf, state, status="running")
    return state


def _root_span(conf: dict, state: dict, status: str, latency_ms: int = None) -> None:
    _post(conf, "/ingest/span", {
        "trace_id": state["trace_id"],
        "span_id": state["root_span"],
        "span_type": "agent",
        "name": "Cursor Agent",
        "agent_name": "Cursor Agent",
        "step_index": 0,
        "status": status,
        "latency_ms": latency_ms,
        "metadata": {"model": state.get("model"), "mode": state.get("mode")},
    })


def _parent(state: dict, sub_key: str = None) -> str:
    """The subagent that emitted the event, else the main agent."""
    sub = state.get("subagents", {}).get(sub_key) if sub_key else None
    return sub["span_id"] if sub else state["root_span"]


TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _tokens(event: dict):
    """Usage Cursor reports on afterAgentResponse / stop, or None when it has none."""
    usage = {}
    for field in TOKEN_FIELDS:
        try:
            usage[field] = int(event.get(field) or 0)
        except (TypeError, ValueError):
            usage[field] = 0
    return usage if any(usage.values()) else None


def _llm_span(conf: dict, state: dict, sub_key, response, usage, event: dict) -> None:
    sub = state.get("subagents", {}).get(sub_key) if sub_key else None
    model = (sub or {}).get("model") or event.get("model") or state.get("model")
    usage = usage or {}
    reported = state.setdefault("reported", {}).setdefault(sub_key or "", dict.fromkeys(TOKEN_FIELDS, 0))
    for field in TOKEN_FIELDS:
        reported[field] += usage.get(field, 0)
    if sub is not None:
        sub["llm"] = True
    cache_read = usage.get("cache_read_tokens", 0)
    cache_write = usage.get("cache_write_tokens", 0)
    calls = _int(event.get("loop_count"))
    span_id = f"span-{uuid.uuid4().hex[:12]}"
    metadata = {
        "usage": usage or None,
        "model_calls": calls,
        "avg_context_tokens": usage.get("input_tokens", 0) // calls if calls else None,
        "model_id": event.get("model_id"),
        "model_params": event.get("model_params"),
        "mode": state.get("mode"),
    }
    state.setdefault("last_llm", {})[sub_key or ""] = {"span_id": span_id, "metadata": metadata}
    _span(
        conf, state,
        span_id=span_id,
        span_type="llm",
        name=model or "cursor",
        provider="cursor",
        model=model,
        parent_span_id=_parent(state, sub_key),
        messages_in=[{"role": "user", "content": (sub.get("task") if sub else state.get("prompt")) or ""}],
        response=_clip(response),
        # Cursor's input_tokens is the whole context sent to the model, cache included,
        # summed over the model calls of the run; the backend bills the cached part at
        # the cache rate.
        tokens_in=usage.get("input_tokens") or None,
        tokens_out=usage.get("output_tokens") or None,
        cache_read_tokens=cache_read or None,
        cache_write_tokens=cache_write or None,
        metadata=metadata,
    )


def _report_remaining_usage(conf: dict, state: dict, sub_key, event: dict) -> None:
    """`stop` carries the run's usage: whatever afterAgentResponse did not report yet, and
    the number of model calls, which only `stop` gives."""
    total = _tokens(event)
    if not total:
        return
    reported = state.get("reported", {}).get(sub_key or "", {})
    rest = {f: max(0, total[f] - reported.get(f, 0)) for f in TOKEN_FIELDS}
    if any(rest.values()):
        _llm_span(conf, state, sub_key, None, rest, event)
        return
    calls = _int(event.get("loop_count"))
    last = state.get("last_llm", {}).get(sub_key or "")
    if calls and last and not last["metadata"].get("model_calls"):
        last["metadata"].update(
            model_calls=calls, avg_context_tokens=total["input_tokens"] // calls)
        _post(conf, "/ingest/span", {
            "trace_id": state["trace_id"], "span_id": last["span_id"], "span_type": "llm",
            "metadata": last["metadata"],
        })


def _subagent_span(conf: dict, state: dict, sub: dict, status: str) -> None:
    _post(conf, "/ingest/span", {
        "trace_id": state["trace_id"],
        "span_id": sub["span_id"],
        "span_type": "agent",
        "step_index": sub.get("step", 0),
        "name": sub["name"],
        "agent_name": sub["name"],
        "parent_span_id": state["root_span"],
        "status": status,
        "latency_ms": sub.get("latency_ms"),
        "response": sub.get("response"),
        "metadata": {
            "subagent_type": sub.get("type"),
            "model": sub.get("model"),
            "parallel": sub.get("parallel"),
            "task": _clip(sub.get("task"), 2000),
        },
    })


def _finish_subagent(conf: dict, state: dict, sub_key: str, event: dict) -> None:
    sub = state.get("subagents", {}).get(sub_key)
    if sub is None or sub.get("done"):
        return
    sub["done"] = True
    sub["latency_ms"] = int((time.time() - sub["started"]) * 1000)
    sub["response"] = _clip(_first(event, "result", "output", "summary", "text"), 4000)
    _report_remaining_usage(conf, state, sub_key, event)
    _subagent_span(conf, state, sub, "ok")


def _task_args(tool_input: dict) -> dict:
    """Arguments of a subagent launch: the Task tool's input, or nested in `arguments`
    when the tool is invoked through a generic dispatcher (e.g. CallDynamicTool)."""
    nested = tool_input.get("arguments")
    if isinstance(nested, str):
        nested = _as_dict(nested)
    if isinstance(nested, dict) and ("prompt" in nested or "description" in nested):
        return nested
    return tool_input


def _subagent_of_tool(state: dict, event: dict, tool_input: dict):
    """Key of the subagent a Task tool call launched (same call id, or same prompt)."""
    subagents = state.get("subagents", {})
    call_id = str(_first(event, "tool_use_id", "tool_call_id") or "")
    if call_id in subagents:
        return call_id
    prompt = " ".join(str(_task_args(tool_input).get("prompt") or "").split())
    if prompt:
        for key, sub in subagents.items():
            if " ".join(str(sub.get("task") or "").split()) == prompt:
                return key
    return None


def _handle(conf: dict, event: dict, name: str, state: dict, sub_key: str = None) -> None:
    if name in ("postToolUse", "postToolUseFailure"):
        tool = str(_first(event, "tool_name", "tool") or "tool")
        failed = name == "postToolUseFailure"
        error = _clip(_first(event, "error", "error_message"), 2000) if failed else None
        tool_input = _as_dict(_first(event, "tool_input", "input"))
        tool_output = _clip(_first(event, "tool_output", "output", "result", "result_json"))
        launched = _subagent_of_tool(state, event, tool_input)
        if not launched and ("subagent_type" in _task_args(tool_input) or tool.lower() in ("task", "agent", "subagent")):
            _log(f"unmatched subagent tool {tool!r}: input keys {sorted(tool_input)}, event keys {sorted(event)}")
        if launched:
            # The Task call that launched a subagent: name the subagent node after the
            # call's description instead of adding a duplicate tool node.
            sub = state["subagents"][launched]
            args = _task_args(tool_input)
            if args.get("description"):
                sub["name"] = " ".join(str(args["description"]).split())[:PROMPT_PREVIEW]
            if args.get("run_in_background"):
                sub["background"] = True
            if not sub.get("response"):
                sub["response"] = _clip(tool_output, 4000)
            _subagent_span(conf, state, sub, "ok" if sub.get("done") else "running")
            return
        _span(
            conf, state,
            span_type="tool",
            name=tool,
            tool_name=tool,
            parent_span_id=_parent(state, sub_key),
            tool_input=tool_input,
            tool_output=tool_output,
            error=error or ("tool failed" if failed else None),
            latency_ms=_duration_ms(event),
        )
    elif name == "subagentStart":
        key = str(_first(event, "subagent_id", "tool_call_id", "tool_use_id") or uuid.uuid4().hex)
        task = str(_first(event, "task", "description") or "")
        subagents = state.setdefault("subagents", {})
        kind = str(event.get("subagent_type") or "sous-agent")
        # Parallel subagents often share a prompt opening: number them until the Task
        # call's description names them.
        label = event.get("description") or f"{kind} #{1 + sum(s.get('type') == event.get('subagent_type') for s in subagents.values())}"
        state["step"] = state.get("step", 0) + 1
        sub = {
            "span_id": f"span-{uuid.uuid4().hex[:12]}",
            "step": state["step"],
            "name": " ".join(str(label).split())[:PROMPT_PREVIEW],
            "task": task,
            "type": event.get("subagent_type"),
            "model": event.get("subagent_model") or event.get("model"),
            "parallel": bool(event.get("is_parallel_worker")),
            "started": time.time(),
        }
        subagents[key] = sub
        _subagent_span(conf, state, sub, "running")
    elif name == "subagentStop":
        subagents = state.get("subagents", {})
        key = str(_first(event, "subagent_id", "tool_call_id", "tool_use_id") or "")
        if key not in subagents:
            key = next((k for k, s in subagents.items() if not s.get("done")), None)
        if key:
            _finish_subagent(conf, state, key, event)
    elif name == "stop" and sub_key:
        # A subagent's own run ended: close its node, not the prompt's trace.
        _finish_subagent(conf, state, sub_key, event)
    elif name == "afterAgentResponse":
        _llm_span(conf, state, sub_key, _first(event, "text", "response"), _tokens(event), event)


WORK_EVENTS = {"postToolUse", "postToolUseFailure", "subagentStart", "subagentStop", "afterAgentResponse"}
RESUME_EVENTS = {"postToolUse", "postToolUseFailure", "subagentStart"}
SUBAGENT_QUIET_S = 300         # a subagent silent for this long is not waited for


def _running_subagents(state: dict) -> list:
    """Subagents still sending events. Background subagents send none at all to the
    hooks, so the trace cannot wait for them."""
    now = time.time()
    return [k for k, s in state.get("subagents", {}).items()
            if not s.get("done") and s.get("seen_at") and now - s["seen_at"] < SUBAGENT_QUIET_S]


def _close_if_done(conf: dict, state: dict) -> None:
    """End the trace when the main agent has stopped and no subagent is still running.
    Background subagents stay open: their completion notification closes them later."""
    if state.get("closed") or not state.get("root_stopped") or _running_subagents(state):
        return
    state["closed"] = True
    status = state.get("stop_status") or "completed"
    for key, sub in list(state.get("subagents", {}).items()):
        if not sub.get("background"):
            _finish_subagent(conf, state, key, {})
    started = state.get("started") or time.time()
    _root_span(conf, state, "ok" if status == "completed" else "error", int((time.time() - started) * 1000))
    _post(conf, "/ingest/trace/end", {"trace_id": state["trace_id"], "status": status})


NOTIFICATION = re.compile(r"<task>(.*?)</task>", re.S)
NOTIFIED_STATUS = {"success": "ok", "completed": "ok", "error": "error", "failed": "error", "aborted": "error"}


def _notified_tasks(prompt: str) -> list:
    """Subagent completions announced in a system notification turn:
    [{"tool_call_id", "status", "title", "detail"}]."""
    tasks = []
    for block in NOTIFICATION.findall(prompt or ""):
        fields = dict(re.findall(r"^\s*(tool_call_id|status|title|agent_id):\s*(.+?)\s*$", block, re.M))
        detail = re.search(r"detail:\s*(.*?)(?:\n\s*output_path:|\Z)", block, re.S)
        if fields.get("tool_call_id"):
            fields["detail"] = detail.group(1).strip() if detail else None
            tasks.append(fields)
    return tasks


def _complete_notified_subagents(conf: dict, prompt: str) -> None:
    """Close the nodes of background subagents whose completion Cursor just announced, in
    the trace that launched them (which may already be ended)."""
    for task in _notified_tasks(prompt):
        key = task["tool_call_id"]
        for path in _open_runs(6 * 3600):
            state = _load(path)
            if not state or key not in state.get("subagents", {}):
                continue
            with _Lock(path):
                state = _load(path) or state
                sub = state["subagents"][key]
                if not sub.get("done"):
                    sub["done"] = True
                    sub["latency_ms"] = int((time.time() - sub["started"]) * 1000)
                    sub["response"] = _clip(task.get("detail"), 4000)
                    if task.get("title") and sub["name"].split(" #")[0] == str(sub.get("type")):
                        sub["name"] = task["title"][:PROMPT_PREVIEW]
                    _subagent_span(conf, state, sub, NOTIFIED_STATUS.get(task.get("status", ""), "ok"))
                    if state.get("closed"):
                        # The trace lasted until its last subagent finished.
                        _post(conf, "/ingest/trace/end", {
                            "trace_id": state["trace_id"],
                            "status": state.get("stop_status") or "completed",
                        })
                    _save(path, state)
            break


def _reopen(conf: dict, state: dict) -> None:
    state["closed"] = False
    _post(conf, "/ingest/trace/reopen", {"trace_id": state["trace_id"]})
    _root_span(conf, state, "running")


def _fix_mojibake(text: str) -> str:
    """Undo UTF-8 read as Windows-1252 then re-encoded ("lancÃ©" → "lancé"), which
    PowerShell does to piped input."""
    if "Ã" not in text and "â€" not in text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _read_event(raw: bytes) -> dict:
    """Hook input, whatever encoding the shell between Cursor and Python used
    (PowerShell may re-encode piped input as UTF-16 or add a BOM)."""
    if not raw.strip():
        _log("empty hook input on stdin")
        return {}
    encodings = ["utf-8-sig", "utf-16", "cp1252"]
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:8]:
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            text = raw.decode(encoding).strip().strip("\x00")
            event = json.loads(_fix_mojibake(text))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(event, dict):
            return event
    _log(f"unreadable hook input ({len(raw)} bytes): {raw[:120]!r}")
    return {}


def _alias_path(run_id: str) -> str:
    return _state_path(run_id)[: -len(".json")] + ".alias"


def _open_runs(max_age_s: float = ATTACH_WINDOW_S) -> list:
    """State files of the runs active within `max_age_s`, most recently active first."""
    try:
        names = os.listdir(STATE_DIR)
    except OSError:
        return []
    runs = []
    for file in names:
        if not file.endswith(".json"):
            continue
        full = os.path.join(STATE_DIR, file)
        try:
            age = time.time() - os.path.getmtime(full)
        except OSError:
            continue
        if age < max_age_s:
            runs.append((age, full))
    return [full for _, full in sorted(runs)]


def _waiting_subagents(state: dict) -> list:
    """Subagents launched moments ago whose own run id is not known yet."""
    now = time.time()
    return [
        k for k, s in state.get("subagents", {}).items()
        if not s.get("done") and not s.get("run_id") and now - s.get("started", 0) < SUBAGENT_ATTACH_S
    ]


def _run_of_conversation(conversation: str, include_closed: bool = False):
    """Most recently active run of a conversation (open only, unless include_closed)."""
    for path in _open_runs():
        state = _load(path)
        if (state and (include_closed or not state.get("closed"))
                and state.get("conversation_id") == conversation):
            return path
    return None


def _subagent_of(conversation: str):
    """(state path, subagent key) of the subagent whose id is this conversation id."""
    for path in _open_runs():
        state = _load(path)
        if state and conversation in state.get("subagents", {}):
            return path, conversation
    return None


def _attach(run_id: str, event: dict, only_waiting: bool = False):
    """Run id Cursor gave to a subagent (or a run whose prompt hook was missed): find
    the open trace it belongs to. Returns (state path, subagent key) or (None, None)."""
    conversation = event.get("conversation_id")
    candidates = []
    for path in _open_runs():
        state = _load(path)
        if not state:
            continue
        waiting = _waiting_subagents(state)
        if (only_waiting or state.get("closed")) and not waiting:
            continue
        same_conversation = bool(conversation) and state.get("conversation_id") == conversation
        candidates.append((not waiting, not same_conversation, path))
    if not candidates:
        return None, None
    path = sorted(candidates)[0][2]           # a run waiting for a subagent first, then same conversation
    with _Lock(path):
        state = _load(path)
        if not state:
            return None, None
        sub_key = next(iter(_waiting_subagents(state)), None)
        if sub_key:
            state["subagents"][sub_key]["run_id"] = run_id
            _save(path, state)
    with open(_alias_path(run_id), "w", encoding="utf-8") as f:
        json.dump({"path": path, "subagent": sub_key}, f)
    return path, sub_key


def _resolve(run_id: str, name: str, event: dict):
    """(state path, subagent key, owns the trace) for this event."""
    conversation = event.get("conversation_id")
    if name in ("subagentStart", "subagentStop") and event.get("parent_conversation_id"):
        # A background subagent can stop after the turn that launched it ended.
        path = _run_of_conversation(str(event["parent_conversation_id"]), include_closed=name == "subagentStop")
        if path:
            return path, None, False
    if conversation:
        # A subagent's own events: Cursor gives the subagent its own conversation id,
        # announced as `subagent_id` when the parent launched it.
        found = _subagent_of(str(conversation))
        if found:
            return found[0], found[1], False
    if not event.get("generation_id") and conversation:
        path = _run_of_conversation(str(conversation))
        if path:
            return path, None, False
    own = _state_path(run_id)
    if os.path.exists(own):
        return own, None, True
    alias = _load(_alias_path(run_id))
    if alias and os.path.exists(alias.get("path", "")):
        return alias["path"], alias.get("subagent"), False
    # A new prompt opens its own trace, unless it is the task of a subagent just launched.
    path, sub_key = _attach(run_id, event, only_waiting=name == "beforeSubmitPrompt")
    if path:
        return path, sub_key, False
    return own, None, True                    # nothing open: a run whose prompt hook was missed


def main() -> None:
    event = _read_event(sys.stdin.buffer.read())
    name = event.get("hook_event_name") or ""
    output = {"continue": True} if name == "beforeSubmitPrompt" else {}
    try:
        conf = _config()
        run_id = str(_first(event, "generation_id", "conversation_id") or "")
        if name in ("afterAgentResponse", "stop", "subagentStop"):
            _log(f"{name} {event.get('generation_id')}: tokens {_tokens(event)}, fields {sorted(event)}")
        if conf["api_key"] and run_id:
            os.makedirs(STATE_DIR, exist_ok=True)
            if name == "beforeSubmitPrompt":
                _complete_notified_subagents(conf, str(event.get("prompt") or ""))
            path, sub_key, owner = _resolve(run_id, name, event)
            with _Lock(path):
                state = _load(path)
                if state and state.get("closed") and owner and name == "beforeSubmitPrompt":
                    state = None                      # a new prompt after the run ended
                if state is None and owner and name != "stop":
                    _cleanup_old_states()
                    state = _start_run(conf, event)
                if state and name == "stop" and owner:
                    # The main agent stopped (finished, or waiting for the user to allow a
                    # command). The trace ends once no subagent is running; Cursor keeps
                    # sending events after `stop`, so the state stays to catch them.
                    state["root_stopped"] = True
                    state["stop_status"] = STOP_STATUS.get(str(event.get("status") or "completed"), "completed")
                    _report_remaining_usage(conf, state, None, event)
                    _save(path, state)
                    _close_if_done(conf, state)
                    _save(path, state)
                elif state:
                    if _mode(event) and not state.get("mode"):
                        state["mode"] = _mode(event)
                    if sub_key and sub_key in state.get("subagents", {}):
                        state["subagents"][sub_key]["seen_at"] = time.time()
                    if name in WORK_EVENTS and not (name == "subagentStop" and not _running_subagents(state)):
                        if state.get("closed"):
                            _reopen(conf, state)
                        if owner and name in RESUME_EVENTS:
                            state["root_stopped"] = False     # resumed, e.g. after "allow"
                    _handle(conf, event, name, state, sub_key)
                    _close_if_done(conf, state)
                    _save(path, state)
        elif not conf["api_key"]:
            _log("AITRAJECTIQ_API_KEY is not set: Cursor events are not sent")
        else:
            _log(f"{name or '?'}: no generation_id/conversation_id in the event (keys: {sorted(event)})")
    except Exception as e:  # never break the agent
        _log(f"{name}: {e!r}")
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()
