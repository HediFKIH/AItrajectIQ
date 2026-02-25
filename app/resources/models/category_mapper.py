from beanie import Document
from typing import List,Optional



class Category(Document):
    category_id: int
    category:str
    chunk_ids:Optional[List[int]]=[]
    class Settings:
        indexes = [
            [
                ("category"),
                ("category_id"),
            ],
        ]