"""Document loading: turn uploaded bytes into clean, page-numbered text.

PDFs keep their real page numbers. TXT / MD / DOCX have no pages, so they are cut into
"virtual pages" of ~3000 characters so every citation can still point to a location.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
VIRTUAL_PAGE_CHARS = 3000


def clean_text(text: str) -> str:
    """Normalise whitespace and repair hyphenated line breaks produced by PDF extraction."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "infor-\nmation" -> "information"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _virtual_pages(text: str, size: int = VIRTUAL_PAGE_CHARS) -> list[tuple[int, str]]:
    pages: list[str] = []
    current: list[str] = []
    length = 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and length + len(para) > size:
            pages.append("\n\n".join(current))
            current, length = [], 0
        current.append(para)
        length += len(para)
    if current:
        pages.append("\n\n".join(current))
    return list(enumerate(pages, start=1))


def _read_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise ValueError("This PDF is password-protected.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append((number, text))
    except ValueError:
        raise
    except Exception as exc:  # corrupt file etc.
        raise ValueError("Could not read this PDF file.") from exc
    if not pages:
        raise ValueError("No extractable text found (scanned PDF?). OCR is not supported.")
    return pages


def _read_docx(data: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("Could not read this DOCX file.") from exc
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a supported file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(data)
    if ext == ".docx":
        pages = _virtual_pages(clean_text(_read_docx(data)))
    elif ext in {".txt", ".md"}:
        pages = _virtual_pages(clean_text(data.decode("utf-8", errors="replace")))
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use PDF, TXT, MD or DOCX.")
    if not pages:
        raise ValueError("The file contains no text.")
    return pages
