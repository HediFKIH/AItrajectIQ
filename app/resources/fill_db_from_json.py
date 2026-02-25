

# from .models.category_mapper import Category
from pydantic import BaseModel
from typing import List

from fastapi import HTTPException




# class CreateCategoryRequest(BaseModel):
#     category_id: int
#     category:str


# async def create_category(category_data: CreateCategoryRequest):
#     category = Category(**category_data.model_dump())
#     await category.insert()
#     return category


# class UpdateCategoryRequest(BaseModel):
#     chunk_ids: List[int]
 


# async def update_category(category_id: int, category_update: UpdateCategoryRequest):
#     category = await Category.get(category_id)
#     if not category:
#         raise HTTPException(status_code=404, detail="Category not found")
    
#     update_data = category_update.model_dump(exclude_unset=True)  # only fields sent in request
#     for field, value in update_data.items():
#         setattr(category, field, value)
    
#     await category.save()
#     return category






from pathlib import Path
from app.resources.models.category_mapper import Category











def map_json_to_category(json_data: dict) :
    return Category(
        category_id=json_data["id"],
        category=json_data["category"],
    )


import asyncio
from app.config.db import init_db  
import json

# file_path="app/resources/recent_category_store.jsonl"

# async def main():
#     await init_db()
#     print("Beanie initialized. Now you can use your models.")
#     file = Path(file_path)
#     if not file.exists():
#         raise FileNotFoundError(f"{file_path} not found.")
#     success_count=0

#     fail_count=0
#     i=0


#     with file.open("r", encoding="utf-8") as f:
#         for i, line in enumerate(f, 1):
#             try:
#                 raw_data = json.loads(line)

#                 category = map_json_to_category(raw_data)
#                 await category.insert()
#                 success_count += 1
#             except Exception as e:
#                 fail_count += 1
#                 print(f"[{i}] Error inserting line: {e}")

#     print(f"\n Done. Inserted: {success_count}, Failed: {fail_count}")









# file_path="app/resources/recent_category_mapper.json"

# async def main():
#     await init_db()
#     print("Beanie initialized. Now you can use your models.")
#     file = Path(file_path)
#     if not file.exists():
#         raise FileNotFoundError(f"{file_path} not found.")
#     success_count=0

#     fail_count=0
#     i=0


#     with file.open("r", encoding="utf-8") as f:
#         data = json.load(f) 

       
#         categories = await Category.find_all().to_list()

#         for category in categories:

#             i+=1
#             if success_count%1000==0:
#                 print(success_count, " categories updated successfully")
#             try:
#                 chunks=data.get(category.category)
#                 if not chunks: 
#                     print(category.category, " was not found")
#                     await category.delete()
#                     continue
#                 category.chunk_ids = chunks 
#                 await category.save()   
        
#                 success_count += 1

#             except Exception as e:
#                 fail_count += 1
#                 print(f"[{i}] Error updating line: {e}")

#     print(f"\n Done. Updated: {success_count}, Failed: {fail_count}")


# asyncio.run(main())














from pathlib import Path
from app.resources.models.doc_store import Chunk, Metadata




def map_json_to_chunk(json_data: dict) -> Chunk:
    return Chunk(
        chunk_id=json_data["id"],
        text=json_data["text"],
        metadata=Metadata(
            page_title=json_data["metadata"]["page title"],
            page_id=json_data["metadata"]["page id"],
            lead_paragraph=json_data["metadata"]["lead paragraph"],
            section=json_data["metadata"]["section"],
            categories=json_data["metadata"]["categories"],
        )
    )


import asyncio
from app.config.db import init_db  
import json

file_path="app/resources/recent_doc_store.jsonl"

async def main():
    await init_db()
    print("Beanie initialized. Now you can use your models.")
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    success_count=0

    fail_count=0

    with file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                raw_data = json.loads(line)

                chunk = map_json_to_chunk(raw_data)
                await chunk.insert()
                success_count += 1
            except Exception as e:
                fail_count += 1
                print(f"[{i}] Error inserting line: {e}")
        if success_count% 10000==0:
            print(f"Inserted {success_count} chunk")

    print(f"\n Done. Inserted: {success_count}, Failed: {fail_count}")



asyncio.run(main())


