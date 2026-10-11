"""
Project API key management.

POST   /projects/quickstart                       create a project and its first key (dashboard onboarding)
POST   /projects/{project_id}/api-keys            create a key (the raw key is returned once);
                                                  open only while the project has no active key
GET    /projects/{project_id}/api-keys            list keys (masked), needs a read key of the project
DELETE /projects/{project_id}/api-keys/{key_id}   revoke a key, needs a key of the project
GET    /auth/whoami                               project and scopes of the calling key
"""

from datetime import datetime
from typing import List, Optional

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from models.apikey import ApiKey, ApiKeyScope
from models.metric import MetricName
from models.project import Project, ProjectCategory, ProjectTask
from security.api_keys import (
    READ,
    AuthContext,
    authenticate,
    display_prefix,
    generate_api_key,
    hash_api_key,
    invalidate_cached_key,
    require_api_key,
)

router = APIRouter(tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    name: str = "default"
    scopes: List[ApiKeyScope] = Field(
        default_factory=lambda: [ApiKeyScope.ingest, ApiKeyScope.read],
        min_length=1,
    )


class ApiKeyOut(BaseModel):
    id: str
    project_id: str
    name: str
    prefix: str
    scopes: List[ApiKeyScope]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    active: bool


class CreatedApiKey(ApiKeyOut):
    key: str = Field(description="The full API key. It is only returned once.")


def _to_out(key: ApiKey) -> ApiKeyOut:
    return ApiKeyOut(
        id=str(key.id),
        project_id=str(key.project_id),
        name=key.name,
        prefix=key.prefix,
        scopes=key.scopes,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
        active=key.revoked_at is None,
    )


async def _get_project(project_id: str) -> Project:
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, f"Invalid project_id: {project_id!r}")
    project = await Project.get(pid)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.post(
    "/projects/{project_id}/api-keys",
    response_model=CreatedApiKey,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(project_id: str, request: Request, body: Optional[CreateApiKeyRequest] = None):
    """The first key of a project can be created freely; later ones need a key of the
    project holding every scope being granted."""
    project = await _get_project(project_id)
    body = body or CreateApiKeyRequest()
    has_active_key = await ApiKey.find(ApiKey.project_id == project.id, ApiKey.revoked_at == None).count()
    if has_active_key:
        auth = await authenticate(request, None, project_id=project_id)
        missing = {s.value for s in body.scopes} - set(auth.key.scopes)
        if missing:
            raise HTTPException(403, f"API key {auth.key.prefix}... cannot grant scopes it lacks: {sorted(missing)}")
    raw_key = generate_api_key()
    key = ApiKey(
        project_id=project.id,
        name=body.name,
        key_hash=hash_api_key(raw_key),
        prefix=display_prefix(raw_key),
        scopes=body.scopes,
    )
    await key.insert()
    return CreatedApiKey(**_to_out(key).model_dump(), key=raw_key)


class QuickstartRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class QuickstartProject(CreatedApiKey):
    project_name: str


@router.post("/projects/quickstart", response_model=QuickstartProject, status_code=status.HTTP_201_CREATED)
async def quickstart(body: QuickstartRequest):
    """A new project and its first key (ingest + read) in one call, for the dashboard's
    onboarding: the same access as POST /projects/ followed by the project's first key.
    Agent runs get the agent metrics automatically; plain LLM calls get relevance and
    hallucination checks."""
    project = Project(
        name=body.name.strip(),
        client_id="dashboard",
        type=ProjectCategory.LLMObserve,
        task=ProjectTask.agent,
        enabled_metrics=[MetricName.answer_relevance, MetricName.hallucination],
        created_at=datetime.utcnow(),
    )
    await project.insert()
    raw_key = generate_api_key()
    key = ApiKey(
        project_id=project.id,
        name="dashboard",
        key_hash=hash_api_key(raw_key),
        prefix=display_prefix(raw_key),
        scopes=[ApiKeyScope.ingest, ApiKeyScope.read],
    )
    await key.insert()
    return QuickstartProject(**_to_out(key).model_dump(), key=raw_key, project_name=project.name)


@router.get("/projects/{project_id}/api-keys", response_model=List[ApiKeyOut])
async def list_api_keys(project_id: str, request: Request):
    project = await _get_project(project_id)
    await authenticate(request, READ, project_id=project_id)
    keys = await ApiKey.find(ApiKey.project_id == project.id).sort("-created_at").to_list()
    return [_to_out(k) for k in keys]


@router.delete(
    "/projects/{project_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_api_key(project_id: str, key_id: str, request: Request):
    project = await _get_project(project_id)
    await authenticate(request, None, project_id=project_id)
    try:
        kid = PydanticObjectId(key_id)
    except Exception:
        raise HTTPException(400, f"Invalid key_id: {key_id!r}")
    key = await ApiKey.get(kid)
    if key is None or key.project_id != project.id:
        raise HTTPException(404, "API key not found")
    if key.revoked_at is None:
        key.revoked_at = datetime.utcnow()
        await key.save()
    invalidate_cached_key(key.key_hash)
    return None


@router.get("/auth/whoami")
async def whoami(auth: AuthContext = Depends(require_api_key())):
    return {
        "project_id": str(auth.project.id),
        "project_name": auth.project.name,
        "key_name": auth.key.name,
        "key_prefix": auth.key.prefix,
        "scopes": sorted(auth.key.scopes),
    }
