# src/subgen/api/routes/__init__.py
from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter()

from subgen.api.routes.health import router as health_router  # noqa: E402
from subgen.api.routes.subtitles import router as subtitles_router  # noqa: E402
from subgen.api.routes.jobs import router as jobs_router  # noqa: E402
from subgen.api.routes.metrics import router as metrics_router  # noqa: E402

api_router.include_router(health_router)
api_router.include_router(subtitles_router)
api_router.include_router(jobs_router)
api_router.include_router(metrics_router)