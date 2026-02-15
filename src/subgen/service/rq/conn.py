from __future__ import annotations

from typing import Optional

import redis


def get_redis_connection(redis_url: str) -> redis.Redis:
    """
    Create a Redis connection from REDIS_URL.
    Keep it simple and deterministic for K8s.
    """
    if not redis_url or not redis_url.strip():
        raise ValueError("redis_url is empty")

    # decode_responses=False: we store bytes/JSON text as needed; RQ works with bytes.
    conn = redis.from_url(redis_url, decode_responses=False)

    # Fail fast if Redis is unreachable (helps during deployment).
    try:
        conn.ping()
    except Exception as e:
        raise RuntimeError(f"Redis ping failed for url={redis_url!r}: {e}") from e

    return conn