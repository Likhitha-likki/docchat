"""Ties everything together: ingest PDFs, retrieve chunks, generate cited answers."""
import re
from typing import BinaryIO, Callable, Iterator

from .chunker import chunk_pages
from .config import TOP_K, MIN_SCORE
from .embeddings import Embedder
from .llm import NOT_FOUND, stream_answer
from .loader import load_pdf
from .store import Hit, VectorStore


class DocChat:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm_stream: Callable[[str, list[Hit]], Iterator[str]] = stream_answer,
    ):
        # Everything is injectable so tests (and your Project 2 eval code) can swap parts out
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore.load()
        self._llm_stream = llm_stream

    def ingest(self, source: str | bytes | BinaryIO, doc_name: str) -> int:
        """Index one PDF. Returns number of chunks added (0 if it was already indexed)."""
        if self.store.has_doc(doc_name):
            return 0
        pages = load_pdf(source, doc_name)
        if not pages:
            raise ValueError(
                f"No text found in '{doc_name}'. It may be a scanned PDF (needs OCR) or empty."
            )
        chunks = chunk_pages(pages)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.store.save()
        return len(chunks)

    def retrieve(self, question: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> list[Hit]:
        hits = self.store.search(self.embedder.embed_query(question), k)
        return [h for h in hits if h.score >= min_score]

    def ask(self, question: str, k: int = TOP_K) -> tuple[list[Hit], Iterator[str]]:
        """Returns (retrieved hits, stream of answer text)."""
        hits = self.retrieve(question, k)
        if not hits:
            return [], iter([NOT_FOUND])  # nothing relevant: skip the LLM call entirely
        return hits, self._llm_stream(question, hits)


def cited_hits(answer: str, hits: list[Hit]) -> list[tuple[int, Hit]]:
    """Return only the sources the answer actually cites, as (number, hit)."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [(n, hits[n - 1]) for n in numbers if 1 <= n <= len(hits)]
