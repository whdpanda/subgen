# src/subgen/api/routes/metrics.py
from __future__ import annotations

from fastapi import APIRouter, Response

from subgen.api.metrics import metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def prom_metrics() -> Response:
    """
    Prometheus scrape endpoint.
    """
    body = metrics().to_prometheus_text()
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")