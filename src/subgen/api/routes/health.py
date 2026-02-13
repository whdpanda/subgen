from __future__ import annotations

from fastapi import APIRouter

from subgen.api.config import load_config
from subgen.api import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    cfg = load_config()
    return {
        "ok": True,
        "service": "subgen-api",
        "version": __version__,
        "data_root": cfg.data_root,
        "allowed_roots": cfg.allowed_roots,
        "allow_create_output_dir": cfg.allow_create_output_dir,
    }
