import json
from beanie import Document, PydanticObjectId
from .metric import Metric
from typing import Optional, List
from datetime import datetime
from pydantic import model_validator




class Log(Document):
    project_id : PydanticObjectId
    prompt: str
    history: list
    retrieved_contexts: Optional[List[str]] =None
    Output: Optional[str]= None
    Model: Optional[str]= None
    timestamp: datetime 
    temperature: Optional[float]=1
    latency: Optional[int]=None
    n_tokens: Optional[int]=None
    blocked: Optional[bool]=None
    blocked_reason:Optional[str]=None
    contains_pii: Optional[bool]=None
    anonymized: Optional[bool]= None

    metrics: Optional[List[Metric]]=None
    
    final_score: Optional[float] = None
    #reception_endpoint= str

    #base_url: str
    #i probably shouldn't be storing the API keys of my clients to their providers


    @model_validator(mode='after')
    def check_blocked(cls,model):
        if model.blocked and not model.blocked_reason:
            raise ValueError("Block reason must be specified")
        
        if model.contains_pii and not model.anonymized: 
            raise ValueError("Must be anonymized")

        return model
    class Settings:
        indexes = [
            [
                ("project_id"),
                ("blocked"),
            ],
        ]





