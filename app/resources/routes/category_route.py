
from fastapi import APIRouter, HTTPException
from typing import List
from ..models.category_mapper import Category # Beanie Document


router = APIRouter(
    prefix="/category",
    tags=["Categories"]
)

@router.get("/count", response_model=int)
async def count_categories():
    return await Category.count()



@router.get("/", response_model=List[Category])
async def list_categories():
    return await Category.find_all().to_list()

@router.get("/{category_id}", response_model=Category)
async def get_category_by_id(category_id: int):
    category = await Category.find_one(Category.category_id == category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category



