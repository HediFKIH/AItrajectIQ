"""
Files the dashboard's setup instructions download (public, no secret inside).

GET /integrations/cursor-hook.py        the Cursor hook script (standard library only)
GET /integrations/claude-code-hook.py   the Claude Code Stop hook script (standard library only)
GET /integrations/claude-code-setup.py  installs that hook and the telemetry settings of Claude Code
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/integrations", tags=["integrations"])

INTEGRATIONS = Path(__file__).resolve().parents[2] / "sdk" / "aitrajectiq" / "integrations"


def _script(name: str) -> FileResponse:
    path = INTEGRATIONS / name
    if not path.is_file():
        raise HTTPException(404, f"{name} is not deployed next to the backend (expected sdk/ beside app/)")
    return FileResponse(path, media_type="text/x-python", filename=name)


@router.get("/cursor-hook.py")
async def cursor_hook():
    return _script("cursor_hook.py")


@router.get("/claude-code-hook.py")
async def claude_code_hook():
    return _script("claude_code_hook.py")


@router.get("/claude-code-setup.py")
async def claude_code_setup():
    return _script("claude_code_setup.py")
