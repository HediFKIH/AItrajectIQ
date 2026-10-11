"""
Project API keys: generation, hashing, lookup and FastAPI dependencies.

Keys look like ``atq_<43 url-safe characters>``. Only their SHA-256 hash is stored,
so a key is shown once when it is created and can never be recovered afterwards.
Keys created before the AItrajectIQ rename start with ``lli_`` and remain valid.
A key identifies its project on its own: callers don't need to send a project_id.

Accepted headers (first match wins):
    X-API-Key: atq_...
    Authorization: Bearer atq_...
    api-key: atq_...            (legacy /records header)

AITRAJECTIQ_AUTH=enforce (default) rejects every request without a valid key.
AITRAJECTIQ_AUTH=permissive keeps the old behavior on the legacy SDK routes only:
an unknown key is accepted when the request names an existing project.
"""

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

from beanie import PydanticObjectId
from fastapi import HTTPException, Request

from models.apikey import ApiKey, ApiKeyScope
from models.project import Project

log = logging.getLogger("aitrajectiq.auth")

KEY_PREFIX = "atq_"
INGEST = ApiKeyScope.ingest.value
READ = ApiKeyScope.read.value

_CACHE_TTL_S = 60.0
_CACHE_MAX_ENTRIES = 10_000
_LAST_USED_MIN_INTERVAL_S = 60.0


@dataclass(frozen=True)
class ApiKeyInfo:
    key_id: PydanticObjectId
    project_id: PydanticObjectId
    scopes: frozenset
    name: str
    prefix: str


@dataclass
class AuthContext:
    project: Project
    key: Optional[ApiKeyInfo]   # None only for legacy calls accepted in permissive mode


# key_hash -> (monotonic time of lookup, info or None for unknown/revoked keys)
_cache: dict = {}
_last_used_written: dict = {}
_permissive_warned: set = set()


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def display_prefix(raw_key: str) -> str:
    return raw_key[: len(KEY_PREFIX) + 8]


def extract_api_key(headers: Mapping[str, str]) -> Optional[str]:
    """Return the raw key from the request headers, or None.

    ``headers`` must be case-insensitive (Starlette ``Headers``) or use lowercase names.
    """
    value = headers.get("x-api-key")
    if not value:
        scheme, _, token = (headers.get("authorization") or "").partition(" ")
        if scheme.lower() == "bearer":
            value = token
    if not value:
        value = headers.get("api-key")
    value = (value or "").strip()
    return value or None


def has_scope(info: ApiKeyInfo, scope: Optional[str]) -> bool:
    return scope is None or scope in info.scopes


def invalidate_cached_key(key_hash: str) -> None:
    _cache.pop(key_hash, None)


def auth_mode() -> str:
    mode = os.getenv("AITRAJECTIQ_AUTH", "enforce").strip().lower()
    return "permissive" if mode == "permissive" else "enforce"


async def resolve_api_key(raw_key: Optional[str]) -> Optional[ApiKeyInfo]:
    """Look up an active key. Results (including misses) are cached for a minute."""
    if not raw_key:
        return None
    key_hash = hash_api_key(raw_key)
    now = time.monotonic()

    cached = _cache.get(key_hash)
    if cached is not None and now - cached[0] < _CACHE_TTL_S:
        info = cached[1]
    else:
        doc = await ApiKey.find_one(ApiKey.key_hash == key_hash)
        info = None
        if doc is not None and doc.revoked_at is None:
            info = ApiKeyInfo(
                key_id=doc.id,
                project_id=doc.project_id,
                scopes=frozenset(s.value if isinstance(s, ApiKeyScope) else str(s) for s in doc.scopes),
                name=doc.name,
                prefix=doc.prefix,
            )
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.clear()
        _cache[key_hash] = (now, info)

    if info is not None:
        await _touch_last_used(info, key_hash, now)
    return info


async def _touch_last_used(info: ApiKeyInfo, key_hash: str, now: float) -> None:
    last = _last_used_written.get(key_hash)
    if last is not None and now - last < _LAST_USED_MIN_INTERVAL_S:
        return
    _last_used_written[key_hash] = now
    try:
        await ApiKey.get_motor_collection().update_one(
            {"_id": info.key_id}, {"$set": {"last_used_at": datetime.utcnow()}}
        )
    except Exception:
        log.debug("Could not update last_used_at for key %s", info.prefix, exc_info=True)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


async def _project_or_404(project_id: str) -> Project:
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, f"Invalid project_id: {project_id!r}")
    project = await Project.get(pid)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


async def authenticate(
    request: Request,
    scope: Optional[str],
    project_id: Optional[str] = None,
    allow_legacy: bool = False,
) -> AuthContext:
    """Resolve the calling project from the request's API key.

    ``project_id`` (sent in a body, or owning the trace being written to) must match
    the key's project. ``allow_legacy`` lets the old SDK routes through without a
    known key when ``AITRAJECTIQ_AUTH=permissive``.
    """
    raw_key = extract_api_key(request.headers)
    info = await resolve_api_key(raw_key)

    if info is not None:
        if not has_scope(info, scope):
            raise HTTPException(403, f"API key {info.prefix}... is missing the '{scope}' scope")
        if project_id and str(project_id) != str(info.project_id):
            raise HTTPException(403, "API key does not belong to this project")
        project = await Project.get(info.project_id)
        if project is None:
            raise _unauthorized("The project of this API key no longer exists")
        return AuthContext(project=project, key=info)

    if allow_legacy and project_id and auth_mode() == "permissive":
        project = await _project_or_404(project_id)
        if project_id not in _permissive_warned:
            _permissive_warned.add(project_id)
            log.warning(
                "AITRAJECTIQ_AUTH=permissive: accepting calls for project %s without a valid "
                "API key. Create one with POST /projects/%s/api-keys.",
                project_id, project_id,
            )
        return AuthContext(project=project, key=None)

    if raw_key:
        raise _unauthorized("Invalid or revoked API key")
    raise _unauthorized(
        "Missing API key: send 'X-API-Key: <key>' or 'Authorization: Bearer <key>'"
    )


def require_api_key(scope: Optional[str] = None):
    """FastAPI dependency factory, e.g. ``Depends(require_api_key(INGEST))``.

    ``scope=None`` accepts any valid key.
    """
    async def dependency(request: Request) -> AuthContext:
        return await authenticate(request, scope)

    return dependency
