from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

from .metric import Metric


class TraceStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    timeout = "timeout"


class Trace(Document):
    project_id: PydanticObjectId
    name: Optional[str] = None
    status: TraceStatus = TraceStatus.running
    metadata: Optional[dict] = None

    total_spans: int = 0
    total_llm_spans: int = 0
    total_tool_spans: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0

    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    metrics: List[Metric] = []

    class Settings:
        name = "traces"
        indexes = [
            "project_id",
            "status",
        ]
