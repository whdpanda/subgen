from __future__ import annotations

import uuid
from typing import Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


def _maybe_set_trace_id(trace_id: str) -> Callable[[], None]:
    """
    Optional hook into subgen.utils.logger's trace context (if available).
    This avoids refactor later: Step 4 can just implement set_trace_id/clear_trace_id
    in logger module, and API automatically starts injecting trace_id.
    """
    try:
        from subgen.utils import logger as logger_mod
    except Exception:
        return lambda: None

    setter = getattr(logger_mod, "set_trace_id", None)
    clearer = getattr(logger_mod, "clear_trace_id", None)
    if callable(setter) and callable(clearer):
        setter(trace_id)
        return lambda: clearer()
    return lambda: None


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    - Generates request_id
    - Stores to request.state.request_id
    - Adds X-Request-Id response header
    - Optionally sets trace_id to logger context (if logger supports)
    """

    def __init__(self, app, header_name: str = "X-Request-Id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(self.header_name)
        if not rid:
            rid = uuid.uuid4().hex

        request.state.request_id = rid

        cleanup = _maybe_set_trace_id(rid)
        try:
            resp: Response = await call_next(request)
        finally:
            cleanup()

        resp.headers[self.header_name] = rid
        return resp


def get_request_id(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)
