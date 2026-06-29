
# app/api/projects.py

from sys import exception
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Literal
from models.project import Project, ProjectCategory, ProjectTask
from models.metric import MetricName
from models.user import User
from models.apikey import APIKey
from beanie import Link, PydanticObjectId
from typing import Optional
from datetime import datetime
from utils.time import now_utc_plus_one
from fastapi import Query


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
    #api_key: Optional[APIKey] = None
    created_at: datetime = now_utc_plus_one()




@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: CreateProjectRequest):
    # If the user is in MongoDB, keep this. 
    # If users are ONLY in PostgreSQL, you should remove this check or use a Postgres client check.
    # user = await User.get(project_data.user_id)
    # if not user:
    #     raise HTTPException(status_code=404, detail="User not found")
    
    data = project_data.model_dump()
    project = Project(**data)
    await project.insert()
    return project

""" @router.get("/", response_model=List[Project])
async def list_projects(user_id=None):
    if not user_id in (None, ""):
        try:
            user_id = ObjectId(user_id)
            user_id = PydanticObjectId(user_id)
            return await Project.find(Project.user_id == user_id).to_list()
            
        except exception:
            raise HTTPException(status_code=422, detail="Invalid user_id")
    return await Project.find_all().to_list()
 """

@router.get("/", response_model=List[Project])
async def list_projects(client_id: Optional[str] = Query(None)):
    if client_id:
        return await Project.find(Project.client_id == client_id).to_list()
    return await Project.find_all().to_list()

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project



class UpdateProjectRequest(BaseModel):
    name: Optional[str] =None
    client_id: Optional[PydanticObjectId] =None
    type:Optional[ProjectCategory] = None
    task: Optional[ProjectTask] = None
    enabled_metrics:Optional[List[MetricName]] = None

@router.patch("/{project_id}", response_model=Project)
async def update_project(project_id: str, project_update: UpdateProjectRequest):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    update_data = project_update.model_dump(exclude_unset=True)  # only fields sent in request
    for field, value in update_data.items():
        setattr(project, field, value)
    
    await project.save()
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    await project.delete()
    return None  # 204 No Content usually returns no body