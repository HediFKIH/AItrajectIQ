
from datetime import datetime,timezone,timedelta

from pydantic import model_validator, BaseModel
from typing import Optional


utc_plus_one = timezone(timedelta(hours=1))

class APIKey(BaseModel):
    key: str
    created_at: datetime= datetime.now(utc_plus_one)
    expires_at: Optional[datetime]
    active: bool
