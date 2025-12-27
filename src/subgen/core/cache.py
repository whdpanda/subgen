from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from subgen.utils.io import write_json, read_json


def file_sha1(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_asr_cache_key(
    audio_path: Path,
    asr_model: str,
    language: str,
) -> str:
    audio_hash = file_sha1(audio_path)
    raw = f"{audio_hash}|{asr_model}|{language}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def asr_cache_path(out_dir: Path, stem: str, key: str) -> Path:
    return out_dir / f"{stem}.transcript.{key}.json"


def save_transcript_json(path: Path, transcript_dict: Dict[str, Any]) -> None:
    write_json(path, transcript_dict)


def load_transcript_json(path: Path) -> Dict[str, Any]:
    return read_json(path)
