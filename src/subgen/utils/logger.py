# src/subgen/utils/logger.py
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "subgen") -> logging.Logger:
    """
    Returns a process-wide logger with idempotent handler setup.
    Handlers are configured by configure_logging(). If not configured yet,
    we attach a minimal stderr StreamHandler at INFO level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Minimal default: stderr console handler
    h = logging.StreamHandler(stream=sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

    # Avoid double logging if root logger is configured elsewhere
    logger.propagate = False
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
    - console handler -> stderr (INFO by default)
    - optional file handler -> log_path (DEBUG by default)

    Idempotent:
    - does not add duplicate handlers of the same type/path.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(console_level, file_level))

    # Ensure no propagation to root (prevents duplicates in some envs)
    logger.propagate = False

    # --- Console (stderr) handler ---
    has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    else:
        # upgrade existing console handler config if needed
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(console_level)
                if h.formatter is None:
                    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                break

    # --- File handler ---
    if log_path:
        log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        def _is_same_filehandler(h: logging.Handler) -> bool:
            return isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_path

        if not any(_is_same_filehandler(h) for h in logger.handlers):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

    return logger
