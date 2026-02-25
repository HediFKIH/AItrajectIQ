from typing import List, Literal,Optional
from pydantic import BaseModel
from beanie import PydanticObjectId
from utils.time import now_utc_plus_one
from datetime import datetime

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str]=None
    messages: list[Message]
    temperature: Optional[float]=None
    max_tokens: Optional[int] =None
    hidden_names: Optional[list] =None
    contexts: Optional[List[str]]=None
    project_id: PydanticObjectId
    provider_endpoint:str
    provider_key:str


class RegisterLogRequest(BaseModel):
    project_id : PydanticObjectId
    prompt: str
    history: list[Message]
    retrieved_contexts: Optional[List[str]] =None
    temperature: Optional[float]=1
    max_tokens: Optional[int] =None
    timestamp: datetime = now_utc_plus_one()
    blocked: Optional[bool]
    blocked_reason:Optional[str] =None
    contains_pii: Optional[bool]
    anonymized: Optional[bool]


class ResponseData(BaseModel):
    Output: Optional[str] =None
    Model: Optional[str]=None
    latency: Optional[float]=None
    n_tokens: Optional[int]=None


