# src/subgen/api/routes/__init__.py
from __future__ import annotations

from fastapi import APIRouter

# Root router to be included by app.py
api_router = APIRouter()

# Import sub-routers (even if placeholder) to avoid later refactor.
from subgen.api.routes.health import router as health_router  # noqa: E402
from subgen.api.routes.subtitles import router as subtitles_router  # noqa: E402
from subgen.api.routes.jobs import router as jobs_router  # noqa: E402

api_router.include_router(health_router)
api_router.include_router(subtitles_router)
api_router.include_router(jobs_router)