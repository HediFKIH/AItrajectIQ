from beanie import Delete, Document, Indexed, before_event
from pydantic import EmailStr
from typing import Annotated
from datetime import datetime
from models.project import Project



class User(Document):
    email: Annotated[EmailStr, Indexed()]
    name: str
    password: str
    created_at: datetime 
    @before_event(Delete)
    async def delete_projects(self):
        await Project.find(Project.user_id == self.id).delete()