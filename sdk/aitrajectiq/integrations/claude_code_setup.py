"""
Connect Claude Code to AItrajectIQ in one command (standard library only).

    curl -fsSL http://localhost:8000/integrations/claude-code-setup.py | python3 - http://localhost:8000 atq_...

Downloads the Stop hook to ~/.aitrajectiq/claude_code_hook.py and merges into
~/.claude/settings.json (previous version kept as settings.json.bak):
    env    the OpenTelemetry export of Claude Code to AItrajectIQ, and the hook's key
    hooks  a Stop hook running claude_code_hook.py with this Python
Other settings and hooks are left as they are; running it again updates the values.
"""

import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

HOOK_NAME = "claude_code_hook.py"


def settings_env(base_url: str, api_key: str) -> dict:
    return {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"{base_url}/otel",
        "OTEL_EXPORTER_OTLP_HEADERS": f"x-api-key={api_key}",
        "OTEL_LOG_USER_PROMPTS": "1",
        "AITRAJECTIQ_API_KEY": api_key,
        "AITRAJECTIQ_BASE_URL": base_url,
    }


def merge_settings(settings: dict, base_url: str, api_key: str, command: str) -> dict:
    """`settings` with the AItrajectIQ env and Stop hook, replacing an earlier AItrajectIQ hook."""
    merged = dict(settings)
    merged["env"] = {**(settings.get("env") or {}), **settings_env(base_url, api_key)}
    hooks = dict(settings.get("hooks") or {})
    stop = [
        group for group in hooks.get("Stop") or []
        if not any(HOOK_NAME in str(h.get("command", "")) for h in (group or {}).get("hooks") or [])
    ]
    stop.append({"hooks": [{"type": "command", "command": command, "timeout": 10}]})
    hooks["Stop"] = stop
    merged["hooks"] = hooks
    return merged


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: claude_code_setup.py <AITRAJECTIQ_URL> <API_KEY>")
    base_url, api_key = sys.argv[1].rstrip("/"), sys.argv[2].strip()

    home = Path.home()
    hook = home / ".aitrajectiq" / HOOK_NAME
    hook.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{base_url}/integrations/claude-code-hook.py", timeout=15) as response:
        hook.write_bytes(response.read())

    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {}
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        shutil.copyfile(path, path.with_name("settings.json.bak"))
    command = f'"{Path(sys.executable).as_posix()}" "{hook.as_posix()}"'
    path.write_text(json.dumps(merge_settings(settings, base_url, api_key, command), indent=2), encoding="utf-8")
    print(f"Claude Code connected to {base_url}. Restart claude: each prompt becomes a trace.")


if __name__ == "__main__":
    main()
