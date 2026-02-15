# src/subgen/utils/logger.py
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import List, Optional

# Per-request trace id (request_id) for correlating logs across modules/tools.
_TRACE_ID: ContextVar[str] = ContextVar("subgen_trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    _TRACE_ID.set(trace_id or "-")


def clear_trace_id() -> None:
    """
    IMPORTANT for FastAPI: clear request-scoped trace id to avoid leaking to next request.
    """
    _TRACE_ID.set("-")


def get_trace_id() -> str:
    return _TRACE_ID.get()


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        # Inject trace_id into every log record.
        record.trace_id = get_trace_id()
        return True


# Request-scoped file handlers stack (for API request-level debug.log)
# Each request can push its own handler; we pop/close on request end.
_FILE_HANDLER_STACK: ContextVar[List[logging.Handler]] = ContextVar("subgen_file_handler_stack", default=[])


def _is_console_handler(h: logging.Handler) -> bool:
    # FileHandler is a StreamHandler subclass — must exclude it explicitly.
    if isinstance(h, logging.FileHandler):
        return False
    if isinstance(h, logging.StreamHandler):
        return getattr(h, "stream", None) in (sys.stderr, sys.stdout)
    return False


def _is_same_filehandler(h: logging.Handler, log_path_abs: str) -> bool:
    return isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_path_abs


def _detailed_file_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s "
        "trace=%(trace_id)s %(module)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str = "subgen") -> logging.Logger:
    """
    Returns a process-wide logger with idempotent handler setup.
    If not configured yet, attaches a minimal stderr console handler at INFO.

    NOTE:
    - This is a convenience wrapper; for the main "subgen" logger,
      prefer configure_logging() on startup.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(logging.INFO)
    ch.addFilter(TraceIdFilter())
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def configure_logging(
    *,
    logger_name: str = "subgen",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_path: Optional[str] = None,
) -> logging.Logger:
    """
    Configure:
    - console handler -> stderr (INFO by default), concise formatter
    - optional process-level file handler -> log_path (DEBUG by default), detailed formatter

    Idempotent:
    - no duplicate console handler
    - no duplicate file handler for same path
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    trace_filter = TraceIdFilter()

    # --- Console (stderr) handler ---
    console_handler: Optional[logging.Handler] = None
    for h in logger.handlers:
        if _is_console_handler(h):
            console_handler = h
            break

    if console_handler is None:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(console_level)
        ch.addFilter(trace_filter)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    else:
        console_handler.setLevel(console_level)
        if not any(isinstance(f, TraceIdFilter) for f in getattr(console_handler, "filters", [])):
            console_handler.addFilter(trace_filter)
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # --- Process-level file handler (optional) ---
    if log_path:
        log_path_abs = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(log_path_abs), exist_ok=True)

        file_handler: Optional[logging.Handler] = None
        for h in logger.handlers:
            if _is_same_filehandler(h, log_path_abs):
                file_handler = h
                break

        if file_handler is None:
            fh = logging.FileHandler(log_path_abs, encoding="utf-8")
            fh.setLevel(file_level)
            fh.addFilter(trace_filter)
            fh.setFormatter(_detailed_file_formatter())
            logger.addHandler(fh)
        else:
            file_handler.setLevel(file_level)
            if not any(isinstance(f, TraceIdFilter) for f in getattr(file_handler, "filters", [])):
                file_handler.addFilter(trace_filter)
            if file_handler.formatter is None:
                file_handler.setFormatter(_detailed_file_formatter())

    return logger


def push_file_log(*, log_path: str, logger_name: str = "subgen", file_level: int = logging.DEBUG) -> None:
    """
    Request-scoped file logging:
    - Add a FileHandler to logger_name
    - Push it into a ContextVar stack so we can pop/close safely

    This prevents handler accumulation across requests.
    """
    if not log_path:
        return

    logger = logging.getLogger(logger_name)
    logger.propagate = False

    log_path_abs = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path_abs), exist_ok=True)

    fh = logging.FileHandler(log_path_abs, encoding="utf-8")
    fh.setLevel(file_level)
    fh.addFilter(TraceIdFilter())
    fh.setFormatter(_detailed_file_formatter())

    logger.addHandler(fh)

    stack = list(_FILE_HANDLER_STACK.get())
    stack.append(fh)
    _FILE_HANDLER_STACK.set(stack)


def pop_file_log(*, logger_name: str = "subgen") -> None:
    """
    Pop the last pushed request-scoped FileHandler and close it.
    """
    stack = list(_FILE_HANDLER_STACK.get())
    if not stack:
        return

    h = stack.pop()
    _FILE_HANDLER_STACK.set(stack)

    logger = logging.getLogger(logger_name)
    try:
        logger.removeHandler(h)
    finally:
        try:
            h.close()
        except Exception:
            # Never throw from logging cleanup in web server context
            pass