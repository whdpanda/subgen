# src/subgen/api/middlewares/access_log.py
from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from subgen.api.metrics import inc_http_request
from subgen.utils.logger import get_logger, get_trace_id

logger = get_logger("subgen.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Access log + lightweight HTTP metrics.
    Assumes RequestContextMiddleware already sets trace_id into ContextVar.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            inc_http_request(method=method, path=path, status=status)

            # Keep it concise, but include trace id for correlation
            logger.info(
                "ACCESS method=%s path=%s status=%s dur_ms=%.2f trace=%s",
                method,
                path,
                status,
                dur_ms,
                get_trace_id(),
            )