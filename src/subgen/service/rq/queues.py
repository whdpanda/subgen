from __future__ import annotations

from typing import Any


def get_queue(conn: Any, name: str, default_timeout_sec: int) -> Any:
    """
    Return an RQ Queue instance.
    """
    if not name or not name.strip():
        raise ValueError("RQ queue name is empty")
    if default_timeout_sec <= 0:
        raise ValueError("default_timeout_sec must be > 0")

    try:
        from rq import Queue
    except ModuleNotFoundError as e:
        raise RuntimeError("rq package is required for queue runtime; install subgen runtime dependencies") from e

    return Queue(
        name=name,
        connection=conn,
        default_timeout=default_timeout_sec,
    )
