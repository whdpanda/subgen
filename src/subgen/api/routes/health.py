from __future__ import annotations

from fastapi import APIRouter

from subgen.api.config import load_config
from subgen.api import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """
    Human/debug-friendly health: includes config surface that is safe to expose.
    """
    cfg = load_config()
    return {
        "ok": True,
        "service": "subgen-api",
        "version": __version__,
        "data_root": cfg.data_root,
        "allowed_roots": cfg.allowed_roots,
        "allow_create_output_dir": cfg.allow_create_output_dir,
    }


@router.get("/healthz")
def healthz() -> dict:
    """
    K8s liveness: must be fast and never block on external deps.
    """
    return {"ok": True}


@router.get("/readyz")
def readyz() -> dict:
    """
    K8s readiness: keep it minimal.
    If later you want Redis readiness, add a short timeout ping here.
    """
    return {"ok": True}