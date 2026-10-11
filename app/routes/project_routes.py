# app/api/projects.py

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Literal
from models.project import Project, ProjectCategory, ProjectTask
from models.metric import MetricName
from models.user import User
from beanie import Link, PydanticObjectId
from typing import Optional
from datetime import datetime
from utils.time import now_utc_plus_one
from fastapi import Query
from security.api_keys import INGEST, READ, auth_mode, authenticate, extract_api_key
import json

router = APIRouter(
    prefix="/projects",
    tags=["projects"]
)

# Pydantic request schema
class CreateProjectRequest(BaseModel):
    name:str
    # user_id: PydanticObjectId
    client_id: str
    type: ProjectCategory
    task: Optional[ProjectTask] = None
    enabled_metrics: Optional[List[MetricName]] = []
    created_at: datetime = now_utc_plus_one()

class UpdateProjectRequest(BaseModel):
    name: Optional[str] =None
    client_id: Optional[str] =None
    type:Optional[ProjectCategory] = None
    task: Optional[ProjectTask] = None
    enabled_metrics:Optional[List[MetricName]] = None

# ---------- Export response schema (no client_id — portable file) ----------
class ProjectExport(BaseModel):
    """Project configuration for export — no client_id so any user can reuse the file."""
    name: str
    type: ProjectCategory
    task: Optional[ProjectTask] = None
    enabled_metrics: List[MetricName] = []


# ---------- Import request schema (client_id included — set by frontend) ----------
class ProjectImport(BaseModel):
    """Project configuration for import — client_id is provided by the frontend (current user)."""
    name: str
    client_id: str
    type: ProjectCategory
    task: Optional[ProjectTask] = None
    enabled_metrics: List[MetricName] = []

@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: CreateProjectRequest):
    data = project_data.model_dump()
    project = Project(**data)
    await project.insert()
    return project

async def _owned_project(request: Request, project_id: str, scope: str) -> Project:
    """The project, once the caller's key is proven to belong to it."""
    return (await authenticate(request, scope, project_id=project_id, allow_legacy=True)).project


@router.get("/", response_model=List[Project])
async def list_projects(request: Request, client_id: Optional[str] = Query(None)):
    """Without AITRAJECTIQ_AUTH=permissive, only the calling key's project is listed."""
    if auth_mode() == "permissive" and not extract_api_key(request.headers):
        if client_id:
            return await Project.find(Project.client_id == client_id).to_list()
        return await Project.find_all().to_list()
    auth = await authenticate(request, READ)
    if client_id and auth.project.client_id != client_id:
        return []
    return [auth.project]

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str, request: Request):
    return await _owned_project(request, project_id, READ)

@router.patch("/{project_id}", response_model=Project)
async def update_project(project_id: str, project_update: UpdateProjectRequest, request: Request):
    project = await _owned_project(request, project_id, INGEST)
    
    update_data = project_update.model_dump(exclude_unset=True)  # only fields sent in request
    for field, value in update_data.items():
        setattr(project, field, value)
    
    await project.save()
    return project


# ---------- Export project configuration (download as JSON file) ----------
@router.get("/{project_id}/export")
async def export_project(project_id: str, request: Request):
    """Export a project's configuration as a downloadable JSON file."""
    project = await _owned_project(request, project_id, READ)

    export_data = {
        "name": project.name,
        "type": project.type.value if hasattr(project.type, 'value') else project.type,
        "task": project.task.value if project.task and hasattr(project.task, 'value') else project.task,
        "enabled_metrics": [m.value if hasattr(m, 'value') else m for m in project.enabled_metrics],
    }

    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    filename = f"{project.name.replace(' ', '_')}_config.json"

    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# ---------- Import project from exported JSON ----------
@router.post("/import", response_model=Project, status_code=status.HTTP_201_CREATED)
async def import_project(
    import_data: ProjectImport,
):
    """Import a project configuration from an exported JSON payload."""
    new_project = Project(
        name=import_data.name,
        client_id=import_data.client_id,
        type=import_data.type,
        task=import_data.task,
        enabled_metrics=import_data.enabled_metrics,
        created_at=now_utc_plus_one(),
    )
    await new_project.insert()
    return new_project

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, request: Request):
    project = await _owned_project(request, project_id, INGEST)
    await project.delete()
    return None  # 204 No Content usually returns no body
