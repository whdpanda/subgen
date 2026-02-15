# src/subgen/api/metrics.py
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class MetricKey:
    name: str
    labels: Tuple[Tuple[str, str], ...] = ()

    def render_prom(self) -> str:
        if not self.labels:
            return self.name
        inner = ",".join([f'{k}="{v}"' for k, v in self.labels])
        return f"{self.name}{{{inner}}}"


class Metrics:
    """
    Ultra-light metrics registry (no external deps).
    - Counter only (enough for Step9 acceptance)
    - Prometheus text exposition format (subset)

    NOTE:
    - This is process-local. In K8s, scrape per-pod.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[MetricKey] = Counter()

    def inc(self, name: str, *, labels: Dict[str, str] | None = None, value: int = 1) -> None:
        key = MetricKey(name=name, labels=tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += int(value)

    def snapshot(self) -> Dict[MetricKey, int]:
        with self._lock:
            return dict(self._counters)

    def to_prometheus_text(self) -> str:
        snap = self.snapshot()
        # Minimal exposition: one line per metric series
        lines: list[str] = []
        for k in sorted(snap.keys(), key=lambda x: (x.name, x.labels)):
            lines.append(f"{k.render_prom()} {snap[k]}")
        return "\n".join(lines) + ("\n" if lines else "")


# Singleton registry for API process
_registry = Metrics()


def metrics() -> Metrics:
    return _registry


def inc_http_request(method: str, path: str, status: int) -> None:
    metrics().inc(
        "subgen_http_requests_total",
        labels={"method": method, "path": path, "status": str(status)},
        value=1,
    )


def inc_job_created(kind: str) -> None:
    metrics().inc("subgen_jobs_created_total", labels={"kind": kind}, value=1)


def inc_job_status_read() -> None:
    metrics().inc("subgen_jobs_status_read_total", value=1)


def inc_job_result_read() -> None:
    metrics().inc("subgen_jobs_result_read_total", value=1)