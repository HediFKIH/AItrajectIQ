from beanie import Document, PydanticObjectId
from pydantic import Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

from .metric import Metric


class SpanType(str, Enum):
    llm = "llm"
    tool = "tool"


class Span(Document):
    trace_id: PydanticObjectId
    project_id: PydanticObjectId
    step_index: int = 0
    span_type: SpanType

    # --- Hierarchy fields (set by SDK) ---
    sdk_span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    caller_chain: Optional[List[str]] = None

    # --- LLM fields ---
    provider: Optional[str] = None
    model: Optional[str] = None
    messages_in: Optional[list] = None
    response: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # --- Tool fields ---
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None

    latency_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # --- Generic agent execution feedback ---
    iteration: Optional[int] = None         # agent loop iteration number (any framework)
    action_success: Optional[bool] = None   # did the action achieve its goal?
    similarity_score: Optional[float] = None  # optional: any similarity/confidence metric
    feedback: Optional[str] = None          # optional: reflection/eval result (any string)

    metrics: List[Metric] = []

    class Settings:
        name = "spans"
        indexes = [
            "trace_id",
            "project_id",
            "span_type",
        ]
