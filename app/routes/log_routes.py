
# app/api/logs.py

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ValidationError
from typing import List
from models.log import Log  # Beanie Document
from models.project import Project
from models.metric import Metric
from typing import Optional, Dict, Any
from datetime import datetime
from beanie import PydanticObjectId

from utils.time import now_utc_plus_one
from bson import ObjectId
from sys import exception


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

    
   


@router.post("/", response_model=Log)
async def create_log(log_data: RegisterLogRequest):
    project = await Project.get(log_data.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    log = Log(**log_data.model_dump())
    await log.insert()
    return log


class RequestAnalysisData(BaseModel):
    blocked: bool
    blocked_reason:Optional[str] =None
    contains_pii: bool
    anonymized: bool

@router.patch("/{log_id}/request_analysis", response_model=Log)
async def add_request_analysis(log_id: str, data: RequestAnalysisData):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    update_data = data.model_dump(exclude_unset=True)  # only fields sent in request
    for field, value in update_data.items():
        setattr(log, field, value)
    
    await log.save()
    return log


class ResponseData(BaseModel):
    Output: Optional[str] =None
    Model: Optional[str]=None
    latency: Optional[str]=None
    n_tokens: Optional[int]=None


@router.patch("/{log_id}/response", response_model=Log)
async def add_response(log_id: str, data: ResponseData):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    update_data = data.model_dump(exclude_unset=True) 
    for field, value in update_data.items():
        setattr(log, field, value)
    
    await log.save()
    return log


class MetricsData(BaseModel):
    metrics: List[Metric]
    final_score: Optional[float] = None

@router.patch("/{log_id}/final_analysis", response_model=Log)
async def add_final_analysis(log_id: str, data: MetricsData):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    update_data = data.model_dump(exclude_unset=True)  
    for field, value in update_data.items():
        setattr(log, field, value)
    
    await log.save()
    return log



@router.get("/{log_id}", response_model=Log)
async def get_log(log_id: str):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="log not found")
    return log

@router.get("/", response_model=List[Log])
async def list_logs(project_id=None):
    if not project_id in (None, ""):
        # print("!!!!!!!!!!")
        try:
            project_id = ObjectId(project_id)
            project_id = PydanticObjectId(project_id)
            return await Log.find(Log.project_id == project_id).to_list()
            
        except exception:
            raise HTTPException(status_code=422, detail="Invalid project_id")
    # print("??????????")
    return await Log.find_all().to_list()

@router.post("/filter", response_model=List[Log])
async def filter_logs(filters: Dict[str, Any], project_id=None):

    if not project_id in (None, ""):
        # print("!!!!!!!!!!")
        try:
            project_id = ObjectId(project_id)
            project_id = PydanticObjectId(project_id)
            filters["project_id"]=project_id
            
        except exception:
            raise HTTPException(status_code=422, detail="Invalid project_id")

    for key in filters:
        
        if key not in Log.model_fields:
            raise HTTPException(status_code=400, detail=f"Invalid filter field: {key}")
        if "project_id" in filters:
            try:
                filters["project_id"] = PydanticObjectId._validate(filters["project_id"])
            except ValidationError:
                raise HTTPException(status_code=400, detail="Invalid project_id format")
            

    return await Log.find(filters).to_list()






@router.delete("/{log_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_log(log_id: str):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    await log.delete()
    return None  # 204 No Content usually returns no body