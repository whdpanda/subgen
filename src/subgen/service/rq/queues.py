from __future__ import annotations

from typing import Optional

from rq import Queue
import redis


def get_queue(conn: redis.Redis, name: str, default_timeout_sec: int) -> Queue:
    """
    Return an RQ Queue instance.
    """
    if not name or not name.strip():
        raise ValueError("RQ queue name is empty")
    if default_timeout_sec <= 0:
        raise ValueError("default_timeout_sec must be > 0")

    return Queue(
        name=name,
        connection=conn,
        default_timeout=default_timeout_sec,
    )