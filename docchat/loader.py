"""PDF -> list of pages with text."""
import io
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader


@dataclass
class Page:
    doc: str
    page: int  # 1-based, matches what you see in a PDF viewer
    text: str


def load_pdf(source: str | bytes | BinaryIO, doc_name: str) -> list[Page]:
    """Extract text page by page. Pages with no extractable text (e.g. scans) are skipped."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    reader = PdfReader(source)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(doc=doc_name, page=number, text=text))
    return pages
