from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KBConfig:
    persist_dir: Path
    collection: str


def kb_config_from_env() -> KBConfig:
    import os

    persist = Path(os.getenv("SUBGEN_KB_DIR", "./.subgen_kb"))
    collection = os.getenv("SUBGEN_KB_COLLECTION", "subgen-kb")
    return KBConfig(persist_dir=persist, collection=collection)
