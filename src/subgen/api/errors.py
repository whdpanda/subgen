from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SubgenApiError(Exception):
    """
    Typed API error carrying a stable machine-readable code.
    """

    code: str
    message: str
    status_code: int = 400
    details: Optional[Dict[str, Any]] = None


class InvalidPathError(SubgenApiError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code="invalid_path", message=message, status_code=400, details=details)


class NotFoundError(SubgenApiError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code="not_found", message=message, status_code=404, details=details)


class ToolInvokeError(SubgenApiError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code="tool_invoke_error", message=message, status_code=500, details=details)


class NotImplementedApiError(SubgenApiError):
    def __init__(self, message: str = "endpoint not implemented yet", *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code="not_implemented", message=message, status_code=501, details=details)
