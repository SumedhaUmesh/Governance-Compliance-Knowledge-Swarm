"""Extract plain text from large regulatory PDFs using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageSpan:
    """1-based page number and [start, end) character offsets in `full_text`."""

    page_number: int
    start: int
    end: int


def extract_pdf(path: str | Path) -> tuple[str, list[PageSpan]]:
    """
    Read all pages with PyMuPDF and return concatenated text plus per-page character spans.

    Non-empty pages are joined with ``\\n\\n``; spans refer only to page body text (not separators).
    """
    pdf_path = Path(path)
    doc = fitz.open(pdf_path)
    sep = "\n\n"
    chunks: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            start = offset
            chunks.append(text)
            end = start + len(text)
            spans.append(PageSpan(page_number=i + 1, start=start, end=end))
            chunks.append(sep)
            offset = end + len(sep)
        full_text = "".join(chunks)
        if full_text.endswith(sep):
            full_text = full_text[: -len(sep)]
        return full_text, spans
    finally:
        doc.close()


def page_for_offset(offset: int, spans: list[PageSpan]) -> int | None:
    """Return 1-based page number containing the character offset, if known."""
    if not spans:
        return None
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = spans[mid]
        if s.start <= offset < s.end:
            return s.page_number
        if offset < s.start:
            hi = mid - 1
        else:
            lo = mid + 1
    if offset >= spans[-1].end:
        return spans[-1].page_number
    if offset < spans[0].start:
        return spans[0].page_number
    return None


def iter_lines_with_offsets(text: str) -> Iterator[tuple[int, str]]:
    """Yield (char_offset, line_without_newline) for each line."""
    pos = 0
    for raw in text.splitlines():
        yield pos, raw
        pos += len(raw) + 1
