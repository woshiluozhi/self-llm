from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import sqlite3
from typing import Iterable

import numpy as np


# Current stage: application integration.
# This module implements a small local vector store for RAG learning. It uses
# deterministic hashed character n-gram embeddings and stores vectors in SQLite.
DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "rag_docs"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "rag_store" / "rag_index.sqlite"
EMBEDDING_DIM = 768


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: int
    chunk_index: int
    score: float
    text: str


class HashingEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            index = value % self.dim
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def _tokens(self, text: str) -> Iterable[str]:
        compact = "".join(text.lower().split())
        for size in (2, 3, 4):
            if len(compact) >= size:
                for index in range(len(compact) - size + 1):
                    yield compact[index : index + size]


def chunk_text(text: str, max_chars: int = 640, overlap: int = 90) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = paragraph

        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max(0, max_chars - overlap) :]

    if current:
        chunks.append(current)
    return chunks


def iter_documents(docs_dir: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            yield path, path.read_text(encoding="utf-8")


def init_db(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_path)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL
        )
        """
    )
    return conn


def build_index(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
    dim: int = EMBEDDING_DIM,
) -> int:
    embedder = HashingEmbedder(dim=dim)
    conn = init_db(index_path)
    total = 0

    for path, text in iter_documents(docs_dir):
        relative = str(path.relative_to(docs_dir))
        for chunk_index, chunk in enumerate(chunk_text(text)):
            vector = embedder.embed(chunk)
            conn.execute(
                """
                INSERT INTO chunks (source, chunk_index, text, embedding, dim)
                VALUES (?, ?, ?, ?, ?)
                """,
                (relative, chunk_index, chunk, vector.tobytes(), dim),
            )
            total += 1

    conn.commit()
    conn.close()
    return total


def index_stats(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, int | bool]:
    if not index_path.exists():
        return {"exists": False, "chunks": 0, "sources": 0}

    conn = sqlite3.connect(index_path)
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    sources = conn.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0]
    conn.close()
    return {"exists": True, "chunks": chunks, "sources": sources}


def search(
    query: str,
    index_path: Path = DEFAULT_INDEX_PATH,
    top_k: int = 4,
    dim: int = EMBEDDING_DIM,
) -> list[RetrievedChunk]:
    if not index_path.exists():
        raise FileNotFoundError(f"RAG index does not exist: {index_path}")

    embedder = HashingEmbedder(dim=dim)
    query_vector = embedder.embed(query)
    conn = sqlite3.connect(index_path)
    rows = conn.execute(
        "SELECT id, source, chunk_index, text, embedding, dim FROM chunks"
    ).fetchall()
    conn.close()

    scored: list[RetrievedChunk] = []
    for chunk_id, source, chunk_index, text, blob, stored_dim in rows:
        if stored_dim != dim:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        score = float(np.dot(query_vector, vector))
        if math.isfinite(score):
            scored.append(
                RetrievedChunk(
                    source=source,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    score=score,
                    text=text,
                )
            )

    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored[:top_k]


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for item in chunks:
        parts.append(
            f"[source: {item.source}#{item.chunk_index}, score: {item.score:.3f}]\n"
            f"{item.text}"
        )
    return "\n\n".join(parts)
