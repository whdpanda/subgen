# tests/_helpers.py
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator


def make_cfg(*, job_root: str) -> Any:
    # Minimal config object for unit tests (monkeypatched into load_config)
    return SimpleNamespace(job_root=job_root)


@contextmanager
def nullctx() -> Iterator[None]:
    yield