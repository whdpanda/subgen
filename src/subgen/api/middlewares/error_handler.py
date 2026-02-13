from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from subgen.api.errors import SubgenApiError
from subgen.api.middlewares.request_context import get_request_id
from subgen.utils.logger import get_logger

logger = get_logger("subgen")


def _err_payload(
    *,
    code: str,
    message: str,
    request_id: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id or "",
        }
    }


def install_error_handlers(app: FastAPI) -> None:
    """
    Centralized error handling.

    Constraints:
    - response JSON must not be polluted: never print, only JSONResponse
    - logs go to logger (stderr/file)
    - include request_id for debugging
    """

    @app.exception_handler(SubgenApiError)
    async def _handle_subgen_api_error(request: Request, exc: SubgenApiError) -> JSONResponse:
        rid = get_request_id(request)
        logger.info(f"API_ERROR rid={rid} code={exc.code} status={exc.status_code} msg={exc.message}")
        logger.debug(f"API_ERROR rid={rid} details={exc.details}")
        return JSONResponse(
            status_code=exc.status_code,
            content=_err_payload(code=exc.code, message=exc.message, request_id=rid, details=exc.details),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        rid = get_request_id(request)
        logger.exception(f"API_UNHANDLED_ERROR rid={rid}")
        return JSONResponse(
            status_code=500,
            content=_err_payload(code="internal_error", message=str(exc), request_id=rid),
        )
