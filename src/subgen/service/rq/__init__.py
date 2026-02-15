"""RQ integration helpers.

Avoid importing heavy optional deps at package import time so tests that
monkeypatch task internals can import modules without redis installed.
"""

__all__ = ["get_redis_connection", "get_queue"]


def get_redis_connection(*args, **kwargs):
    from subgen.service.rq.conn import get_redis_connection as _impl
    return _impl(*args, **kwargs)


def get_queue(*args, **kwargs):
    from subgen.service.rq.queues import get_queue as _impl
    return _impl(*args, **kwargs)
