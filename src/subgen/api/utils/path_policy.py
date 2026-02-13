#src/subgen/api/utils/path_policy.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from subgen.api.config import load_config
from subgen.api.errors import InvalidPathError, NotFoundError


def _has_dot_segments(p: Path) -> bool:
    return any(part in (".", "..") for part in p.parts)


def _norm(p: str) -> str:
    # Keep it conservative; reject obvious badness.
    if "\x00" in p:
        raise InvalidPathError("path contains null byte", details={"path": p})
    return p.strip().strip('"').strip("'")


def _commonpath_ok(resolved: Path, root: Path) -> bool:
    """
    Robust containment check:
      commonpath([resolved, root]) == root
    Works across platforms better than naive prefix.
    """
    try:
        cp = os.path.commonpath([str(resolved), str(root)])
        return Path(cp) == root
    except Exception:
        return False


@dataclass(frozen=True)
class PathPolicy:
    allowed_roots: Tuple[Path, ...]
    allow_create_output_dir: bool

    def _validate_string(self, p: str) -> str:
        if not isinstance(p, str) or not p.strip():
            raise InvalidPathError("path is empty")
        return _norm(p)

    def _validate_absolute(self, pp: Path, original: str) -> None:
        if not pp.is_absolute():
            raise InvalidPathError("path must be absolute (container path)", details={"path": original})

    def _validate_segments(self, pp: Path, original: str) -> None:
        if _has_dot_segments(pp):
            raise InvalidPathError("path contains '.' or '..' segments", details={"path": original})

    def _resolve_non_strict(self, pp: Path) -> Path:
        # strict=False so we can validate output paths that don't exist yet.
        try:
            return pp.resolve(strict=False)
        except Exception:
            # If resolve fails (rare), fall back to absolute normalization.
            return pp.absolute()

    def _ensure_under_allowed_roots(self, resolved: Path, original: str) -> Path:
        for root in self.allowed_roots:
            if _commonpath_ok(resolved, root):
                return resolved
        raise InvalidPathError(
            "path must be under allowed_roots",
            details={"path": original, "allowed_roots": [str(r) for r in self.allowed_roots]},
        )

    def ensure_allowed(self, p: str) -> str:
        """
        Validate path string and ensure it is contained within allowed_roots.
        No filesystem existence requirement.
        """
        original = self._validate_string(p)
        pp = Path(original)

        self._validate_absolute(pp, original)
        self._validate_segments(pp, original)

        resolved = self._resolve_non_strict(pp)
        resolved = self._ensure_under_allowed_roots(resolved, original)

        return str(resolved)

    def ensure_file_exists(self, p: str) -> str:
        """
        Ensure path is allowed and exists as a file.
        """
        ap = self.ensure_allowed(p)
        path = Path(ap)
        if not path.exists():
            raise NotFoundError("file does not exist", details={"path": ap})
        if not path.is_file():
            raise InvalidPathError("path is not a file", details={"path": ap})
        return ap

    def ensure_dir(self, p: str, *, create: bool = False) -> str:
        """
        Ensure path is allowed and is a directory.
        If create=True, create it (controlled by allow_create_output_dir).
        """
        ap = self.ensure_allowed(p)
        path = Path(ap)

        if path.exists():
            if not path.is_dir():
                raise InvalidPathError("path is not a directory", details={"path": ap})
            return ap

        if not create:
            raise NotFoundError("directory does not exist", details={"path": ap})

        if not self.allow_create_output_dir:
            raise InvalidPathError("creating output directory is disabled", details={"path": ap})

        path.mkdir(parents=True, exist_ok=True)
        return ap


def get_path_policy(allowed_roots: Optional[Iterable[str]] = None) -> PathPolicy:
    cfg = load_config()

    roots = list(allowed_roots) if allowed_roots else list(cfg.allowed_roots)
    if not roots:
        roots = [cfg.data_root]

    # Normalize roots to resolved absolute Paths once.
    root_paths = []
    for r in roots:
        r0 = _norm(r)
        rp = Path(r0)
        if not rp.is_absolute():
            raise InvalidPathError("allowed root must be absolute", details={"root": r0})
        rp_resolved = rp.resolve(strict=False)
        root_paths.append(rp_resolved)

    return PathPolicy(allowed_roots=tuple(root_paths), allow_create_output_dir=cfg.allow_create_output_dir)
