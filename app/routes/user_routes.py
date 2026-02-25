
# app/api/users.py

from datetime import datetime,timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List
from models.user import User  # Beanie Document
from typing import Optional
from utils.time import now_utc_plus_one

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

# Pydantic request schema
class CreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    created_at: datetime = now_utc_plus_one()

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: CreateUserRequest):
    user = User(**user_data.model_dump())
    await user.insert()
    return user

@router.get("/", response_model=List[User])
async def list_users():
    return await User.find_all().to_list()

@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user






class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None

@router.patch("/{user_id}", response_model=User)
async def update_user(user_id: str, user_update: UpdateUserRequest):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = user_update.model_dump(exclude_unset=True)  # only fields sent in request
    for field, value in update_data.items():
        setattr(user, field, value)
    
    await user.save()
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await user.delete()
    return None  # 204 No Content usually returns no body