# src/subgen/api/utils/logging_context.py
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Optional

from subgen.utils.logger import get_logger, push_file_log, pop_file_log

logger = get_logger("subgen.api")


@dataclass(frozen=True)
class RequestLogHandle:
    out_dir: Path
    debug_log_path: Path


def _ensure_debug_log_path(out_dir: str) -> RequestLogHandle:
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    debug_path = (p / "debug.log").resolve()
    return RequestLogHandle(out_dir=p, debug_log_path=debug_path)


@contextlib.contextmanager
def request_debug_logging(
    *,
    out_dir: Optional[str],
    capture_stdout: bool = True,
    logger_name: str = "subgen",
) -> Iterator[Optional[RequestLogHandle]]:
    """
    Request-scoped debug.log.

    - Push a request file handler (so logs go into <out_dir>/debug.log)
    - Optionally redirect stdout (print) into the same debug.log
    - Pop/close handler on exit to avoid accumulation/leaks
    """
    if not out_dir:
        yield None
        return

    h = _ensure_debug_log_path(out_dir)
    debug_path_str = str(h.debug_log_path)

    # attach request-scoped handler
    push_file_log(log_path=debug_path_str, logger_name=logger_name)

    f: Optional[IO[str]] = None
    try:
        if capture_stdout:
            f = open(h.debug_log_path, "a", encoding="utf-8", buffering=1)
            with contextlib.redirect_stdout(f):
                yield h
        else:
            yield h
    finally:
        try:
            if f is not None:
                f.flush()
                f.close()
        except Exception:
            logger.exception("Failed to close stdout capture file handle (best-effort).")

        # detach request-scoped handler
        try:
            pop_file_log(logger_name=logger_name)
        except Exception:
            logger.exception("Failed to pop request debug log handler (best-effort).")
