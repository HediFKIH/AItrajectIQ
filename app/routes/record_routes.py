
# app/api/records.py

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
from typing import List,Literal
from models.record import Record  # Beanie Document
from typing import Optional
from beanie import PydanticObjectId
from security.api_keys import INGEST, READ, authenticate


router = APIRouter(
    prefix="/records",
    tags=["records"]
)


# Pydantic request schema
class RegisterRecordRequest(BaseModel):
    project_id : PydanticObjectId
    model_name: Literal["bert-base-uncased-cola"]
    tokens: list[str]
    Output: str
    relevances: list[float]



@router.post("/", response_model=Record)
async def create_record(record_data: RegisterRecordRequest, request: Request):
    auth = await authenticate(
        request, INGEST, project_id=str(record_data.project_id), allow_legacy=True
    )
    project = auth.project
    if project.type != "LLMExplain":
        raise HTTPException(status_code=400, detail="Records require an LLMExplain project")


    record = Record(**record_data.model_dump())
    await record.insert()
    return record


class RequestAnalysisData(BaseModel):
    blocked: bool
    blocked_reason:Optional[str] =None
    contains_pii: bool
    anonymized: bool


async def _owned_record(request: Request, record_id: str, scope: str) -> Record:
    try:
        record = await Record.get(PydanticObjectId(record_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid record_id")
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    await authenticate(request, scope, project_id=str(record.project_id), allow_legacy=True)
    return record


@router.get("/{record_id}", response_model=Record)
async def get_record(record_id: str, request: Request):
    return await _owned_record(request, record_id, READ)


@router.get("/", response_model=List[Record])
async def list_records(request: Request, project_id: Optional[str] = None):
    """Records of the calling key's project (`project_id` must match it if given)."""
    auth = await authenticate(request, READ, project_id=project_id or None, allow_legacy=True)
    return await Record.find(Record.project_id == auth.project.id).to_list()


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_record(record_id: str, request: Request):
    record = await _owned_record(request, record_id, INGEST)
    await record.delete()
    return None  # 204 No Content usually returns no body
