from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from subgen.core.jobs.layout import DEFAULT_JOB_LAYOUT, JobLayout, ensure_dir, norm_path

T = TypeVar("T")


def _model_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Support both Pydantic v2 (model_dump) and v1 (dict).
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")  # v2
    if hasattr(obj, "dict"):
        return obj.dict()  # v1
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported model type for serialization: {type(obj)!r}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomic write: write temp file then replace.
    Prevents partially-written JSON when pod is killed.
    """
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JobSpecStore:
    """
    Unified read/write access to on-disk job artifacts.
    This class is shared by API and worker.

    It does NOT enforce security policies like allowed roots.
    Policy validation should happen in API layer before creating the spec.
    """

    def __init__(self, job_root: str | Path, layout: JobLayout = DEFAULT_JOB_LAYOUT) -> None:
        self.job_root = Path(norm_path(job_root))
        self.layout = layout

    # -------------------------
    # Paths
    # -------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.layout.job_dir(self.job_root, job_id)

    def spec_path(self, job_id: str) -> Path:
        return self.layout.spec_path(self.job_root, job_id)

    def status_path(self, job_id: str) -> Path:
        return self.layout.status_path(self.job_root, job_id)

    def result_path(self, job_id: str) -> Path:
        return self.layout.result_path(self.job_root, job_id)

    def debug_log_path(self, job_id: str) -> Path:
        return self.layout.debug_log_path(self.job_root, job_id)

    def report_path(self, job_id: str) -> Path:
        return self.layout.report_path(self.job_root, job_id)

    def out_dir(self, job_id: str) -> Path:
        return self.layout.out_dir(self.job_root, job_id)

    def tmp_dir(self, job_id: str) -> Path:
        return self.layout.tmp_dir(self.job_root, job_id)

    # -------------------------
    # Lifecycle
    # -------------------------

    def ensure_job_dir(self, job_id: str, *, create_out: bool = True, create_tmp: bool = True) -> Path:
        jd = self.job_dir(job_id)
        ensure_dir(jd)
        if create_out:
            ensure_dir(self.out_dir(job_id))
        if create_tmp:
            ensure_dir(self.tmp_dir(job_id))
        return jd

    # -------------------------
    # Write
    # -------------------------

    def write_spec(self, job_id: str, spec: Any) -> Path:
        path = self.spec_path(job_id)
        self.ensure_job_dir(job_id)
        _atomic_write_json(path, _model_to_dict(spec))
        return path

    def write_status(self, job_id: str, status: Any) -> Path:
        path = self.status_path(job_id)
        self.ensure_job_dir(job_id)
        _atomic_write_json(path, _model_to_dict(status))
        return path

    def write_result(self, job_id: str, result: Any) -> Path:
        path = self.result_path(job_id)
        self.ensure_job_dir(job_id)
        _atomic_write_json(path, _model_to_dict(result))
        return path

    # -------------------------
    # Read
    # -------------------------

    def read_spec_dict(self, job_id: str) -> Dict[str, Any]:
        return _read_json(self.spec_path(job_id))

    def read_status_dict(self, job_id: str) -> Dict[str, Any]:
        return _read_json(self.status_path(job_id))

    def read_result_dict(self, job_id: str) -> Dict[str, Any]:
        return _read_json(self.result_path(job_id))

    def read_as_model(self, job_id: str, model_cls: Type[T], which: str) -> T:
        """
        which: 'spec' | 'status' | 'result'
        """
        if which == "spec":
            data = self.read_spec_dict(job_id)
        elif which == "status":
            data = self.read_status_dict(job_id)
        elif which == "result":
            data = self.read_result_dict(job_id)
        else:
            raise ValueError("which must be 'spec' | 'status' | 'result'")

        # Pydantic v2: model_validate; v1: parse_obj
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(data)  # type: ignore[attr-defined]
        if hasattr(model_cls, "parse_obj"):
            return model_cls.parse_obj(data)  # type: ignore[attr-defined]
        raise TypeError(f"Unsupported model class: {model_cls!r}")

    # -------------------------
    # Existence checks (safe)
    # -------------------------

    def has_spec(self, job_id: str) -> bool:
        return self.spec_path(job_id).exists()

    def has_status(self, job_id: str) -> bool:
        return self.status_path(job_id).exists()

    def has_result(self, job_id: str) -> bool:
        return self.result_path(job_id).exists()