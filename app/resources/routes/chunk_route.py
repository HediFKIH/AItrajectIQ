from fastapi import APIRouter, HTTPException
from typing import List
from ..models.doc_store import Chunk # Beanie Document


router = APIRouter(
    prefix="/chunk",
    tags=["Chunks"]
)


@router.get("/count", response_model=int)
async def count_chunks():
    return await Chunk.count()


@router.get("/", response_model=List[Chunk])
async def list_categories(ids=None):
    if ids: 
        return await Chunk.find({"chunk_id": {"$in": ids}}).to_list()
    return await Chunk.find_all().to_list()




@router.get("/{chunk_id}", response_model=Chunk)
async def get_chunk_by_id(chunk_id: int):
    chunk = await Chunk.find_one(Chunk.chunk_id == chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk



