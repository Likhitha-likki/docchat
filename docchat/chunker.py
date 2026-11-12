"""Split page text into overlapping chunks, preserving page numbers for citations."""
import re
from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .loader import Page

# Sentence ends, or blank lines (paragraph breaks)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class Chunk:
    doc: str
    page: int
    text: str


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _hard_split(piece: str, size: int) -> list[str]:
    """Break an over-long piece at the last space before `size`."""
    parts = []
    while len(piece) > size:
        cut = piece.rfind(" ", 0, size)
        cut = cut if cut > 0 else size
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    pieces: list[str] = []
    for unit in _BOUNDARY.split(_clean(text)):
        unit = unit.strip()
        if unit:
            pieces.extend(_hard_split(unit, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Start the next chunk with the tail of this one (the overlap)
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            if carry_len + len(piece) + 1 > chunk_size:
                carry, carry_len = [], 0
            current, length = carry, carry_len
        current.append(piece)
        length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page number."""
    out = []
    for p in pages:
        for text in split_text(p.text, chunk_size, overlap):
            out.append(Chunk(doc=p.doc, page=p.page, text=text))
    return out
