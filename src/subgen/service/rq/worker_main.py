from __future__ import annotations

import os

from rq import Worker

from subgen.api.config import load_config
from subgen.service.rq.conn import get_redis_connection
from subgen.service.rq.queues import get_queue
from subgen.utils.logger import get_logger

logger = get_logger("subgen.worker_main")


def main() -> None:
    cfg = load_config()

    conn = get_redis_connection(cfg.redis_url)
    queue = get_queue(conn, cfg.rq_queue_name, cfg.rq_job_timeout_sec)

    worker_name = os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME") or "subgen-worker"
    logger.info("Starting worker: name=%s queue=%s redis=%s", worker_name, cfg.rq_queue_name, cfg.redis_url)

    w = Worker([queue], connection=conn, name=worker_name)
    # with_scheduler=False keeps it minimal and deterministic
    w.work(with_scheduler=False)


if __name__ == "__main__":
    main()