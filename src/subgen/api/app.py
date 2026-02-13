# src/subgen/api/app.py
from __future__ import annotations

import logging

from fastapi import FastAPI

from subgen.api.config import load_config
from subgen.api.middlewares.error_handler import install_error_handlers
from subgen.api.middlewares.request_context import RequestContextMiddleware
from subgen.api.routes import api_router
from subgen.utils.logger import configure_logging, get_logger

logger = get_logger("subgen")


def create_app() -> FastAPI:
    cfg = load_config()

    app = FastAPI(
        title="SubGen API",
        version="0.1.0",
    )

    # Request context first (request_id, optional trace_id)
    app.add_middleware(RequestContextMiddleware, header_name="X-Request-Id")

    # Error handlers (stable error JSON, includes request_id)
    install_error_handlers(app)

    # Routers
    app.include_router(api_router)

    @app.on_event("startup")
    async def _startup() -> None:
        # Process-level logging baseline:
        # - console -> stderr, INFO
        # - optional process log file -> cfg.log_path (if set)
        configure_logging(
            logger_name="subgen",
            console_level=logging.INFO,
            file_level=logging.DEBUG,
            log_path=(cfg.log_path or None),
        )
        logger.info("API_STARTUP")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("API_SHUTDOWN")

    return app


app = create_app()
