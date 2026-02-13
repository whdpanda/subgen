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
      - allowed_roots allowlist
      - output dir creation policy
    """

    # Default mount root inside container (e.g. /data).
    data_root: str = os.getenv("SUBGEN_DATA_ROOT", "/data")

    # Optional allowlist override. If empty -> [data_root].
    # Example: SUBGEN_ALLOWED_ROOTS="/data,/mnt/share"
    allowed_roots: List[str] = None  # type: ignore[assignment]

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
