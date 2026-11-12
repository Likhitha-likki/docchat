from docchat.chunker import chunk_pages, split_text
from docchat.loader import Page

TEXT = " ".join(f"This is sentence number {i} about topic {i % 5}." for i in range(80))


def test_chunks_respect_size_limit():
    chunks = split_text(TEXT, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap():
    chunks = split_text(TEXT, chunk_size=300, overlap=80)
    # the tail of one chunk should reappear at the start of the next
    tail = chunks[0].split(". ")[-1][:20]
    assert tail in chunks[1]


def test_no_text_lost():
    chunks = split_text(TEXT, chunk_size=300, overlap=60)
    joined = " ".join(chunks)
    for i in range(80):
        assert f"sentence number {i} " in joined


def test_very_long_word_run_is_split():
    chunks = split_text("word " * 1000, chunk_size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)


def test_page_numbers_preserved():
    pages = [Page("a.pdf", 3, TEXT), Page("a.pdf", 7, "Short page.")]
    chunks = chunk_pages(pages, chunk_size=300, overlap=60)
    assert {c.page for c in chunks} == {3, 7}
    assert [c for c in chunks if c.page == 7][0].text == "Short page."


def test_empty_text():
    assert split_text("   ") == []
