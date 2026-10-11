"""
Claude Code Stop hook → AItrajectIQ: the final answer and tool arguments of each prompt.

Claude Code's OpenTelemetry export (see app/otel/claude_code.py) carries tokens, costs and
tool names, but not the agent's answer nor the file a tool read. When Claude stops, this
hook reads the session transcript and posts, for the last prompt: its text and time, the
final answer, and the tool calls with their arguments. AItrajectIQ attaches them to the
prompt's trace, judges the answer and finalizes the trace.

    ~/.claude/settings.json
    {
      "env": {"AITRAJECTIQ_API_KEY": "atq_...", "AITRAJECTIQ_BASE_URL": "http://localhost:8000", ...},
      "hooks": {
        "Stop": [{"hooks": [{"type": "command", "command": "python3 \"$HOME/.aitrajectiq/claude_code_hook.py\"",
                             "timeout": 10}]}]
      }
    }

Configuration: AITRAJECTIQ_API_KEY and AITRAJECTIQ_BASE_URL (default http://localhost:8000),
from the environment or from ~/.aitrajectiq/config.json ({"api_key": ..., "base_url": ...}).
Standard library only. The hook never blocks Claude: failures go to
~/.aitrajectiq/claude_code_hook.log.
"""

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HTTP_TIMEOUT_S = 5.0
MAX_TEXT = 20_000
MAX_TOOLS = 300
HOME = os.path.join(os.path.expanduser("~"), ".aitrajectiq")
_COMMAND_PREFIXES = ("<command-", "<local-command-", "<bash-", "<user-memory-input")


def _log(message: str) -> None:
    try:
        os.makedirs(HOME, exist_ok=True)
        with open(os.path.join(HOME, "claude_code_hook.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def _config() -> dict:
    conf = {}
    path = os.path.join(HOME, "config.json")
    try:
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


def _blocks(entry: dict) -> list:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _text(blocks: list) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text" and b.get("text")).strip()


def _is_prompt(entry: dict) -> bool:
    """A message the user typed: not a tool result, a slash-command echo or a meta message."""
    if entry.get("type") != "user" or entry.get("isSidechain") or entry.get("isMeta"):
        return False
    blocks = _blocks(entry)
    if any(b.get("type") == "tool_result" for b in blocks):
        return False
    text = _text(blocks)
    return bool(text) and not text.startswith(_COMMAND_PREFIXES)


def read_transcript(path: str) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def last_turn(entries: list) -> dict:
    """Prompt, prompt time, final answer and tool calls of the last prompt of the main agent."""
    start = max((i for i, e in enumerate(entries) if _is_prompt(e)), default=None)
    if start is None:
        return {}
    turn = [e for e in entries[start + 1:] if not e.get("isSidechain")]
    tools, answer_from = [], 0
    for index, entry in enumerate(turn):
        blocks = _blocks(entry)
        if entry.get("type") == "assistant":
            tools += [{"name": b.get("name"), "input": b.get("input") if isinstance(b.get("input"), dict) else {}}
                      for b in blocks if b.get("type") == "tool_use"]
        elif entry.get("type") == "user" and any(b.get("type") == "tool_result" for b in blocks):
            answer_from = index + 1
    answer = "\n\n".join(filter(None, (_text(_blocks(e)) for e in turn[answer_from:] if e.get("type") == "assistant")))
    if not answer:
        answer = next((t for t in (_text(_blocks(e)) for e in reversed(turn) if e.get("type") == "assistant") if t), "")
    return {
        "prompt": _text(_blocks(entries[start]))[:MAX_TEXT],
        "prompt_at": entries[start].get("timestamp"),
        "answer": answer[:MAX_TEXT],
        "tools": tools[:MAX_TOOLS],
    }


def _post(conf: dict, payload: dict) -> None:
    request = Request(
        f"{conf['base_url']}/otel/v1/claude-code/stop",
        data=json.dumps(payload, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": conf["api_key"]},
        method="POST",
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            response.read()
    except HTTPError as e:
        _log(f"AItrajectIQ answered {e.code}: {e.read()[:300]!r}")
    except (URLError, OSError) as e:
        _log(f"cannot reach {conf['base_url']}: {e}")


def main() -> None:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        if event.get("hook_event_name", "Stop") != "Stop":
            return
        conf = _config()
        if not conf["api_key"]:
            _log("no API key: set AITRAJECTIQ_API_KEY in the env block of ~/.claude/settings.json")
            return
        session_id, transcript = event.get("session_id"), event.get("transcript_path")
        if not session_id or not transcript:
            return
        turn = last_turn(read_transcript(os.path.expanduser(transcript)))
        if turn:
            _post(conf, {"session_id": session_id, "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         **turn})
    except Exception as e:  # never block Claude Code
        _log(f"hook failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
