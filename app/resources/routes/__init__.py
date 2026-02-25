from fastapi import APIRouter

from .category_route import router as category_router
from .chunk_route import router as chunk_router



router= APIRouter(prefix="/resources")
router.include_router(category_router)
router.include_router(chunk_router)
