from __future__ import annotations

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
    *,
    asr_model: str,
    language: str,
    preprocess: Optional[str] = None,
    asr_device: str = "auto",
    asr_compute_type: Optional[str] = None,
    asr_beam_size: int = 1,
    asr_best_of: int = 1,
    asr_vad_filter: bool = False,
) -> str:
    """
    Cache key must change when ANY factor that impacts ASR output changes.
    """
    audio_hash = file_sha1(audio_path)
    raw = "|".join(
        [
            audio_hash,
            f"model={asr_model}",
            f"lang={language}",
            f"pre={preprocess or 'none'}",
            f"dev={asr_device}",
            f"ct={asr_compute_type or 'auto'}",
            f"beam={asr_beam_size}",
            f"best_of={asr_best_of}",
            f"vad={int(bool(asr_vad_filter))}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def asr_cache_path(out_dir: Path, stem: str, key: str) -> Path:
    return out_dir / f"{stem}.transcript.{key}.json"


def save_transcript_json(path: Path, transcript_dict: Dict[str, Any]) -> None:
    write_json(path, transcript_dict)


def load_transcript_json(path: Path) -> Dict[str, Any]:
    return read_json(path)
