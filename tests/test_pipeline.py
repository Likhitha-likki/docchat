import hashlib

import numpy as np
import pytest

from docchat import pipeline
from docchat.llm import NOT_FOUND
from docchat.loader import Page
from docchat.pipeline import DocChat, cited_hits
from docchat.store import VectorStore


class FakeEmbedder:
    """Bag-of-words hashing embedder: no model download needed."""

    DIM = 256

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype="float32")
        for w in text.lower().split():
            w = w.strip(".,?!")
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, q):
        return self._vec(q)[None, :]


def fake_llm(question, hits):
    yield "The answer is in the source "
    yield "[1]."


PAGES = [
    Page("pets.pdf", 1, "Cats are small carnivorous mammals. Cats sleep for up to sixteen hours a day."),
    Page("space.pdf", 4, "The Moon orbits the Earth roughly every twenty seven days. Tides depend on the Moon."),
]


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_pdf", lambda src, name: [p for p in PAGES if p.doc == name])
    return DocChat(embedder=FakeEmbedder(), store=VectorStore(tmp_path / "idx"), llm_stream=fake_llm)


def test_ingest_and_dedupe(chat):
    assert chat.ingest(b"x", "pets.pdf") > 0
    assert chat.ingest(b"x", "pets.pdf") == 0  # already indexed
    assert chat.store.documents().keys() == {"pets.pdf"}


def test_empty_pdf_raises(chat):
    with pytest.raises(ValueError):
        chat.ingest(b"x", "nonexistent.pdf")


def test_retrieval_finds_right_page(chat):
    chat.ingest(b"x", "pets.pdf")
    chat.ingest(b"x", "space.pdf")
    hits = chat.retrieve("How long do cats sleep each day?", min_score=0.05)
    assert hits[0].chunk.doc == "pets.pdf" and hits[0].chunk.page == 1
    hits = chat.retrieve("How often does the Moon orbit the Earth?", min_score=0.05)
    assert hits[0].chunk.doc == "space.pdf" and hits[0].chunk.page == 4


def test_irrelevant_question_skips_llm(chat):
    chat.ingest(b"x", "pets.pdf")
    hits, stream = chat.ask("quantum chromodynamics lagrangian")
    assert hits == []
    assert "".join(stream) == NOT_FOUND


def test_ask_returns_cited_sources(chat):
    chat.ingest(b"x", "pets.pdf")
    chat.retrieve  # noqa
    hits, stream = chat.ask("How long do cats sleep?")
    answer = "".join(stream)
    cited = cited_hits(answer, hits)
    assert [n for n, _ in cited] == [1]
    assert cited[0][1].chunk.doc == "pets.pdf"


def test_store_persists_to_disk(chat, tmp_path):
    chat.ingest(b"x", "pets.pdf")
    reloaded = VectorStore.load(tmp_path / "idx")
    assert reloaded.documents() == chat.store.documents()
    q = FakeEmbedder().embed_query("cats sleep")
    assert reloaded.search(q, 1)[0].chunk.doc == "pets.pdf"


def test_reset(chat):
    chat.ingest(b"x", "pets.pdf")
    chat.store.reset()
    assert chat.store.documents() == {}
    assert chat.retrieve("cats") == []
