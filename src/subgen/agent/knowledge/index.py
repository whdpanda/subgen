from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from .loader import KBConfig


@dataclass(frozen=True)
class KBHit:
    source: str
    score: float
    text: str


class KnowledgeBase:
    def __init__(self, cfg: KBConfig) -> None:
        self.cfg = cfg
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self._vs = Chroma(
            collection_name=cfg.collection,
            embedding_function=self._embeddings,
            persist_directory=str(cfg.persist_dir),
        )

    def search(self, query: str, *, k: int = 6) -> List[KBHit]:
        # langchain-chroma returns (Document, score) for similarity_search_with_score
        pairs = self._vs.similarity_search_with_score(query, k=k)
        hits: List[KBHit] = []
        for doc, score in pairs:
            hits.append(
                KBHit(
                    source=str(doc.metadata.get("source", "")),
                    score=float(score),
                    text=doc.page_content,
                )
            )
        return hits

    def info(self) -> Dict[str, Any]:
        return {
            "persist_dir": str(self.cfg.persist_dir),
            "collection": self.cfg.collection,
        }
