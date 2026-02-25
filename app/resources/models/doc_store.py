from beanie import Document
from typing import List

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    page_title: str
    page_id : int
    lead_paragraph: str
    section: str
    categories:List[str]


class Chunk(Document):
    chunk_id: int =Field(...)
    text: str
    metadata: Metadata
    class Settings:
        indexes = [
            [
                ("chunk_id"),
            ],
        ]




# record = {
#     "id": i,
#     "text": full_text,
#     "metadata":{
#         "page title":title,
#         "page id":page_id,
#         "lead paragraph":lead_paragraph,
#         "section":section[0],
#         "categories":categories

#     }
# }