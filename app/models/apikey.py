from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field


class ApiKeyScope(str, Enum):
    ingest = "ingest"   # SDK /ingest/* routes, OTLP receiver, records
    read = "read"       # MCP endpoint


class ApiKey(Document):
    project_id: PydanticObjectId
    name: str = "default"
    # SHA-256 of the raw key. The raw key is only returned once, at creation.
    key_hash: Annotated[str, Indexed(unique=True)]
    prefix: str
    scopes: List[ApiKeyScope] = Field(
        default_factory=lambda: [ApiKeyScope.ingest, ApiKeyScope.read]
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    class Settings:
        name = "api_keys"
        indexes = ["project_id"]
