"""Recursive character text splitter with overlap (implemented from scratch).

Strategy: try to split on the largest natural boundary first (paragraph, then line,
then sentence, then word, then character) so chunks stay semantically coherent, and
re-merge small pieces up to `chunk_size`, carrying `overlap` characters into the next chunk.
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MIN_CHUNK_CHARS = 30


def _merge(pieces: list[str], sep: str, size: int, overlap: int) -> list[str]:
    docs: list[str] = []
    current: list[str] = []
    total = 0
    sep_len = len(sep)
    for piece in pieces:
        extra = len(piece) + (sep_len if current else 0)
        if total + extra > size and current:
            docs.append(sep.join(current).strip())
            # drop pieces from the front until only `overlap` characters remain
            while current and (
                total > overlap
                or (total + len(piece) + (sep_len if current else 0) > size and total > 0)
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)
        current.append(piece)
        total += len(piece) + (sep_len if len(current) > 1 else 0)
    if current:
        docs.append(sep.join(current).strip())
    return [d for d in docs if d]


def split_text(text: str, size: int, overlap: int, separators: list[str] | None = None) -> list[str]:
    separators = separators or SEPARATORS
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    sep = separators[-1]
    rest: list[str] = []
    for i, candidate in enumerate(separators):
        if candidate == "":
            sep = ""
            break
        if candidate in text:
            sep, rest = candidate, separators[i + 1:]
            break

    pieces = text.split(sep) if sep else list(text)
    out: list[str] = []
    good: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            good.append(piece)
            continue
        if good:
            out.extend(_merge(good, sep, size, overlap))
            good = []
        out.extend(split_text(piece, size, overlap, rest) if rest else [piece])
    if good:
        out.extend(_merge(good, sep, size, overlap))
    return out


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[dict]:
    """Chunk each page separately so a chunk never straddles two pages (accurate citations)."""
    chunks: list[dict] = []
    for page_number, text in pages:
        for piece in split_text(text, size, overlap):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append({"page": page_number, "text": piece})
    return chunks
