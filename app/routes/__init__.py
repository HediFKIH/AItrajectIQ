from fastapi import APIRouter

from .user_routes import router as user_router
from .project_routes import router as project_router
from .log_routes import router as log_router
from .record_routes import router as record_router


router= APIRouter()
router.include_router(user_router)
router.include_router(project_router)
router.include_router(log_router)
router.include_router(record_router)