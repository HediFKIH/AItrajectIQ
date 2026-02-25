
# app/api/records.py

from fastapi import APIRouter, HTTPException, status, Header
from pydantic import BaseModel
from typing import List,Literal
from models.record import Record  # Beanie Document
from models.project import Project
from typing import Optional
from beanie import PydanticObjectId

from bson import ObjectId
from sys import exception


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
async def create_record(
    record_data: RegisterRecordRequest,
    api_key: str = Header(...),  # Extract from headers
):
    
    project = await Project.get(record_data.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.type != "LLMExplain":
        print(project.type)
        raise ValueError("Incompatible project type")
    if project.api_key.key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


    record = Record(**record_data.model_dump())
    await record.insert()
    return record


class RequestAnalysisData(BaseModel):
    blocked: bool
    blocked_reason:Optional[str] =None
    contains_pii: bool
    anonymized: bool



@router.get("/{record_id}", response_model=Record)
async def get_record(record_id: str):
    record = await Record.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record not found")
    return record

@router.get("/", response_model=List[Record])
async def list_records(project_id=None):
    if not project_id in (None, ""):
        # print("!!!!!!!!!!")
        try:
            project_id = ObjectId(project_id)
            project_id = PydanticObjectId(project_id)
            return await Record.find(Record.project_id == project_id).to_list()
            
        except exception:
            raise HTTPException(status_code=422, detail="Invalid project_id")
    # print("??????????")
    return await Record.find_all().to_list()






@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_record(record_id: str):
    record = await Record.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    await record.delete()
    return None  # 204 No Content usually returns no body