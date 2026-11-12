"""FAISS vector store with simple on-disk persistence (index.faiss + chunks.json)."""
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np

from .chunker import Chunk
from .config import INDEX_DIR


class Hit(NamedTuple):
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, path: Path = INDEX_DIR):
        self.path = Path(path)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    # --- write ---
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if self.index is None:
            # Inner product on normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        shutil.rmtree(self.path, ignore_errors=True)

    # --- read ---
    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vector, k)
        return [Hit(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def documents(self) -> dict[str, int]:
        """Document name -> number of chunks."""
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.doc] = counts.get(c.doc, 0) + 1
        return counts

    def has_doc(self, name: str) -> bool:
        return name in self.documents()

    # --- persistence ---
    def save(self) -> None:
        if self.index is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        (self.path / "chunks.json").write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path = INDEX_DIR) -> "VectorStore":
        store = cls(path)
        index_file, chunks_file = store.path / "index.faiss", store.path / "chunks.json"
        if index_file.exists() and chunks_file.exists():
            store.index = faiss.read_index(str(index_file))
            store.chunks = [Chunk(**d) for d in json.loads(chunks_file.read_text(encoding="utf-8"))]
        return store
