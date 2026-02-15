from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _split_csv(v: str) -> List[str]:
    parts = [p.strip() for p in (v or "").split(",")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class ApiConfig:
    """
    API runtime config (env-driven).
    Keep it dependency-light to avoid extra deps in Docker.

    Step 2 adds:
      - Redis/RQ config
    Step 3 adds:
      - Job root (on PVC)
    """

    # Default mount root inside container (e.g. /data).
    data_root: str = os.getenv("SUBGEN_DATA_ROOT", "/data")

    # Optional allowlist override. If empty -> [data_root].
    # Example: SUBGEN_ALLOWED_ROOTS="/data,/mnt/share"
    allowed_roots: List[str] = None  # type: ignore[assignment]

    # Jobs root directory (must be under allowed roots in K8s manifests)
    # Backward compatible env names:
    # - new: SUBGEN_JOB_ROOT / SUBGEN_REDIS_URL / SUBGEN_RQ_* (used by k8s manifests)
    # - legacy: JOB_ROOT / REDIS_URL / RQ_*
    job_root: str = os.getenv("SUBGEN_JOB_ROOT", os.getenv("JOB_ROOT", "/data/jobs"))

    # Redis/RQ
    redis_url: str = os.getenv("SUBGEN_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    rq_queue_name: str = os.getenv("SUBGEN_RQ_QUEUE_NAME", os.getenv("RQ_QUEUE_NAME", "subgen"))
    rq_job_timeout_sec: int = int(os.getenv("SUBGEN_RQ_JOB_TIMEOUT_SEC", os.getenv("RQ_JOB_TIMEOUT", "5400")))

    # Logging knobs (Step 4 will fully wire debug.log per request)
    log_level: str = os.getenv("SUBGEN_API_LOG_LEVEL", "INFO")
    log_path: str = os.getenv("SUBGEN_LOG_PATH", "")

    # Quality loop knobs (future)
    quality_max_passes_default: int = int(os.getenv("SUBGEN_QUALITY_MAX_PASSES", "3"))

    # Output directory policy (Step 3 will rely on this)
    allow_create_output_dir: bool = os.getenv("SUBGEN_ALLOW_CREATE_OUTPUT_DIR", "1") not in ("0", "false", "False")

    def __post_init__(self) -> None:
        # dataclass(frozen=True) + default None fields: use object.__setattr__
        roots_env = os.getenv("SUBGEN_ALLOWED_ROOTS", "").strip()
        roots = _split_csv(roots_env) if roots_env else [self.data_root]
        object.__setattr__(self, "allowed_roots", roots)


def load_config() -> ApiConfig:
    return ApiConfig()
