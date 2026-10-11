from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import ASCENDING, IndexModel
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

    source: str = "sdk"                         # sdk | otel
    external_id: Optional[str] = None           # OpenTelemetry trace id (hex)
    last_activity_at: Optional[datetime] = None

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
            # Partial: SDK traces store external_id = null and must not collide.
            IndexModel(
                [("project_id", ASCENDING), ("external_id", ASCENDING)],
                unique=True,
                partialFilterExpression={"external_id": {"$type": "string"}},
            ),
            IndexModel([("project_id", ASCENDING), ("started_at", ASCENDING)]),
        ]
