import json
from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel


class OtelEvent(Document):
    """GenAI content event received as an OTLP log record, waiting to be merged into its span."""

    project_id: PydanticObjectId
    otel_trace_id: str
    otel_span_id: str
    name: str
    time_ns: int = 0
    # JSON {"attributes": {...}, "body": ...}: attribute names contain dots, which
    # older MongoDB servers reject as field names.
    data: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def as_event(self) -> dict:
        data = json.loads(self.data or "{}")
        return {
            "name": self.name,
            "time_ns": self.time_ns,
            "attributes": data.get("attributes") or {},
            "body": data.get("body"),
        }

    class Settings:
        name = "otel_events"
        indexes = [
            IndexModel([("project_id", ASCENDING), ("otel_trace_id", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)], expireAfterSeconds=24 * 3600),
        ]
