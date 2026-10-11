
# app/api/logs.py

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from typing import List
from models.log import Log  # Beanie Document
from models.metric import Metric
from typing import Optional, Dict, Any
from datetime import datetime
from beanie import PydanticObjectId

from security.api_keys import INGEST, READ, authenticate
from utils.time import now_utc_plus_one


router = APIRouter(
    prefix="/logs",
    tags=["logs"]
)


class RegisterLogRequest(BaseModel):
    project_id : PydanticObjectId
    prompt: str
    retrieved_contexts: Optional[List[str]] =None
    temperature: Optional[float]=None
    history: list
    timestamp: datetime = now_utc_plus_one()


async def _owned_log(request: Request, log_id: str, scope: str) -> Log:
    try:
        log = await Log.get(PydanticObjectId(log_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid log_id")
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    await authenticate(request, scope, project_id=str(log.project_id), allow_legacy=True)
    return log


async def _update_log(request: Request, log_id: str, data: BaseModel) -> Log:
    log = await _owned_log(request, log_id, INGEST)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(log, field, value)
    await log.save()
    return log


@router.post("/", response_model=Log)
async def create_log(log_data: RegisterLogRequest, request: Request):
    await authenticate(request, INGEST, project_id=str(log_data.project_id), allow_legacy=True)
    log = Log(**log_data.model_dump())
    await log.insert()
    return log


class RequestAnalysisData(BaseModel):
    blocked: bool
    blocked_reason:Optional[str] =None
    contains_pii: bool
    anonymized: bool

@router.patch("/{log_id}/request_analysis", response_model=Log)
async def add_request_analysis(log_id: str, data: RequestAnalysisData, request: Request):
    return await _update_log(request, log_id, data)


class ResponseData(BaseModel):
    Output: Optional[str] =None
    Model: Optional[str]=None
    latency: Optional[str]=None
    n_tokens: Optional[int]=None


@router.patch("/{log_id}/response", response_model=Log)
async def add_response(log_id: str, data: ResponseData, request: Request):
    return await _update_log(request, log_id, data)


class MetricsData(BaseModel):
    metrics: List[Metric]
    final_score: Optional[float] = None

@router.patch("/{log_id}/final_analysis", response_model=Log)
async def add_final_analysis(log_id: str, data: MetricsData, request: Request):
    return await _update_log(request, log_id, data)


@router.get("/{log_id}", response_model=Log)
async def get_log(log_id: str, request: Request):
    return await _owned_log(request, log_id, READ)


@router.get("/", response_model=List[Log])
async def list_logs(request: Request, project_id: Optional[str] = None):
    """Logs of the calling key's project (`project_id` must match it if given)."""
    auth = await authenticate(request, READ, project_id=project_id or None, allow_legacy=True)
    return await Log.find(Log.project_id == auth.project.id).to_list()


@router.post("/filter", response_model=List[Log])
async def filter_logs(filters: Dict[str, Any], request: Request, project_id: Optional[str] = None):
    project_id = project_id or filters.get("project_id")
    auth = await authenticate(request, READ, project_id=str(project_id) if project_id else None, allow_legacy=True)
    for key in filters:
        if key not in Log.model_fields:
            raise HTTPException(status_code=400, detail=f"Invalid filter field: {key}")
    filters["project_id"] = auth.project.id
    return await Log.find(filters).to_list()


@router.delete("/{log_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_log(log_id: str, request: Request):
    log = await _owned_log(request, log_id, INGEST)
    await log.delete()
    return None  # 204 No Content usually returns no body
