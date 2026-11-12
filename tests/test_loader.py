import io

import pytest

pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas  # noqa: E402

from docchat.loader import load_pdf  # noqa: E402


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_load_pdf_page_numbers():
    pages = load_pdf(make_pdf(["First page text", "", "Third page text"]), "t.pdf")
    assert [p.page for p in pages] == [1, 3]  # blank page 2 skipped, numbering preserved
    assert "Third page" in pages[1].text
