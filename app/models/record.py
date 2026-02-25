from beanie import Document, PydanticObjectId
from typing import Optional, Literal





class Record(Document):
    project_id : PydanticObjectId
    model_name: Literal["bert-base-uncased-cola"]
    tokens: list[str]
    Output: str
    relevances: Optional[list[float]]


    class Settings:
        indexes = [
            [
                ("project_id"),
                ("model_name"),
            ],
        ]





