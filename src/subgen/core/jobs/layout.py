from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobLayout:
    """
    Single source of truth for on-disk job layout.
    Changing filenames here is a breaking change. Do not change for PR#6+.

    Root:
      <job_root>/
        <job_id>/
          spec.json
          status.json
          result.json
          debug.log
          report.json
          out/                 (recommended; optional)
          tmp/                 (optional)
    """

    spec_filename: str = "spec.json"
    status_filename: str = "status.json"
    result_filename: str = "result.json"
    debug_log_filename: str = "debug.log"
    report_filename: str = "report.json"

    out_dirname: str = "out"
    tmp_dirname: str = "tmp"

    def job_dir(self, job_root: str | Path, job_id: str) -> Path:
        root = Path(job_root)
        return root / job_id

    def spec_path(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.spec_filename

    def status_path(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.status_filename

    def result_path(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.result_filename

    def debug_log_path(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.debug_log_filename

    def report_path(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.report_filename

    def out_dir(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.out_dirname

    def tmp_dir(self, job_root: str | Path, job_id: str) -> Path:
        return self.job_dir(job_root, job_id) / self.tmp_dirname


DEFAULT_JOB_LAYOUT = JobLayout()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_abs_path(p: str | Path) -> bool:
    return Path(p).is_absolute()


def norm_path(p: str | Path) -> str:
    # Normalize (no symlink resolution to avoid surprises)
    return os.path.normpath(str(p))