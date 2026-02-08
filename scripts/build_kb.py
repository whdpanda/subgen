from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings: offline by default
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass(frozen=True)
class BuildArgs:
    src: Path
    persist: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    reset: bool
    embedding_model: str


def iter_markdown_files(src: Path) -> Iterable[Path]:
    # Deterministic ordering for stable chunking/ids
    files = [p for p in src.rglob("*.md") if p.is_file()]
    for p in sorted(files, key=lambda x: x.as_posix()):
        yield p


def file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    h.update(p.read_bytes())
    return h.hexdigest()


def load_docs(src: Path) -> List[Document]:
    docs: List[Document] = []
    for md in iter_markdown_files(src):
        text = md.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(md.as_posix()),
                    "sha1": file_sha1(md),
                    "kind": "knowledge_md",
                    "title": md.stem,
                },
            )
        )
    return docs


def split_docs(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    out: List[Document] = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            out.append(
                Document(
                    page_content=c,
                    metadata={
                        **d.metadata,
                        "chunk_id": i,
                    },
                )
            )
    return out


def doc_id(doc: Document) -> str:
    # stable ID: source + sha1 + chunk_id
    src = doc.metadata.get("source", "")
    sha1 = doc.metadata.get("sha1", "")
    cid = str(doc.metadata.get("chunk_id", "0"))
    return f"{src}::{sha1}::{cid}"


def _reset_collection_if_needed(vs: Chroma, *, collection: str, reset: bool) -> None:
    """
    Drop the whole collection when reset=True.

    Why:
    - Deterministic rebuild semantics (no accidental duplicates)
    - Fast and simple during development (PR#4 repeatability)
    """
    if not reset:
        return

    try:
        # Newer wrappers may implement this directly.
        vs.delete_collection()
        return
    except Exception:
        pass

    # Fallback: underlying chroma client, name may differ by version.
    for attr in ("_client", "client"):
        client = getattr(vs, attr, None)
        if client is None:
            continue
        try:
            client.delete_collection(collection)
            return
        except Exception:
            pass

    # Best-effort: do not crash build on reset failure.


def build_index(args: BuildArgs) -> Tuple[int, Path]:
    args.persist.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.src)
    chunks = split_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    # Create VS handle
    vs = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(args.persist),
    )

    # reset (drop) collection if requested
    _reset_collection_if_needed(vs, collection=args.collection, reset=args.reset)

    # IMPORTANT: if collection was dropped, recreate a fresh handle
    if args.reset:
        vs = Chroma(
            collection_name=args.collection,
            embedding_function=embeddings,
            persist_directory=str(args.persist),
        )

    ids = [doc_id(d) for d in chunks]
    vs.add_documents(documents=chunks, ids=ids)

    return len(chunks), args.persist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Knowledge markdown root, e.g. docs/knowledge")
    ap.add_argument("--persist", type=str, default="./.subgen_kb", help="Chroma persist dir")
    ap.add_argument("--collection", type=str, default="subgen-kb", help="Chroma collection name")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--reset", action="store_true", help="Reset (drop) collection before indexing")
    ns = ap.parse_args()

    args = BuildArgs(
        src=Path(ns.src),
        persist=Path(ns.persist),
        collection=ns.collection,
        chunk_size=ns.chunk_size,
        chunk_overlap=ns.chunk_overlap,
        reset=bool(ns.reset),
        embedding_model=str(ns.embedding_model),
    )

    if not args.src.exists():
        raise SystemExit(f"--src not found: {args.src}")

    n, persist = build_index(args)

    extra = " (reset)" if args.reset else ""
    print(
        "[OK] indexed chunks=%d, persist=%s, collection=%s, embed=%s%s"
        % (n, persist.as_posix(), args.collection, args.embedding_model, extra)
    )


if __name__ == "__main__":
    main()
