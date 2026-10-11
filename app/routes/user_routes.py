
# app/api/users.py

import hashlib
import secrets
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List
from beanie import PydanticObjectId
from models.user import User  # Beanie Document
from typing import Optional
from security.api_keys import require_api_key
from utils.time import now_utc_plus_one

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

PBKDF2_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime


def _to_out(user: User) -> UserOut:
    return UserOut(id=str(user.id), email=user.email, name=user.name, created_at=user.created_at)


async def _user_or_404(user_id: str) -> User:
    try:
        user = await User.get(PydanticObjectId(user_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# Pydantic request schema
class CreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    created_at: datetime = now_utc_plus_one()

@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: CreateUserRequest):
    data = user_data.model_dump()
    data["password"] = hash_password(data["password"])
    user = User(**data)
    await user.insert()
    return _to_out(user)

@router.get("/", response_model=List[UserOut], dependencies=[Depends(require_api_key())])
async def list_users():
    return [_to_out(u) for u in await User.find_all().to_list()]

@router.get("/{user_id}", response_model=UserOut, dependencies=[Depends(require_api_key())])
async def get_user(user_id: str):
    return _to_out(await _user_or_404(user_id))


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None

@router.patch("/{user_id}", response_model=UserOut, dependencies=[Depends(require_api_key())])
async def update_user(user_id: str, user_update: UpdateUserRequest):
    user = await _user_or_404(user_id)
    update_data = user_update.model_dump(exclude_unset=True)  # only fields sent in request
    if update_data.get("password"):
        update_data["password"] = hash_password(update_data["password"])
    for field, value in update_data.items():
        setattr(user, field, value)
    await user.save()
    return _to_out(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_api_key())])
async def delete_user(user_id: str):
    user = await _user_or_404(user_id)
    await user.delete()
    return None  # 204 No Content usually returns no body
