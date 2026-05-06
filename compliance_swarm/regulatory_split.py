"""Detect Chapter / Article / Section structure and split text into LlamaIndex Documents."""

from __future__ import annotations

import re
from pathlib import Path

from llama_index.core import Document

from compliance_swarm.pdf_extract import PageSpan, iter_lines_with_offsets, page_for_offset

# Line-start patterns common in EU/US/code-style regulations (tune per corpus).
_CHAPTER = re.compile(
    r"^(?:Chapter|CHAPTER)\s+(.+)$",
    re.I,
)
_ARTICLE = re.compile(
    r"^(?:Article|ARTICLE|Art\.)\s+(\d+[a-z]?)(?:\s*[.:]\s*(.*))?$",
    re.I,
)
_SECTION = re.compile(
    r"^(?:Section|SECTION|Sec\.|§)\s*([\w.-]+)(?:\s*[.:]\s*(.*))?$",
    re.I,
)


def _match_heading(line: str) -> dict[str, str] | None:
    """If `line` is a structural heading, return metadata field updates."""
    if m := _CHAPTER.match(line):
        title = m.group(1).strip()
        return {"chapter": title, "article": "", "section": ""}
    if m := _ARTICLE.match(line):
        num = m.group(1).strip()
        return {"article": num, "section": ""}
    if m := _SECTION.match(line):
        num = m.group(1).strip()
        return {"section": num}
    return None


def split_into_documents(
    full_text: str,
    page_spans: list[PageSpan],
    source_file: str | Path,
    *,
    doc_id_prefix: str = "",
) -> list[Document]:
    """
    Split regulatory text so each segment inherits stable structural metadata.

    Each emitted Document is one continuous region under the same Chapter/Article/Section
    state; headings start a new Document so metadata stays accurate through hierarchical
    chunking.
    """
    source_str = str(source_file)
    state: dict[str, str] = {"chapter": "", "article": "", "section": ""}
    docs: list[Document] = []
    buffer: list[str] = []
    segment_start_char: int | None = None
    segment_last_char: int = 0

    def flush() -> None:
        nonlocal buffer, segment_start_char
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        start = segment_start_char if segment_start_char is not None else 0
        end = max(segment_last_char - 1, start)
        buffer = []
        segment_start_char = None
        if not body:
            return
        end_clamped = min(end, max(len(full_text) - 1, 0))
        meta = {
            "source_file": source_str,
            "chapter": state["chapter"],
            "article": state["article"],
            "section": state["section"],
            "page_start": page_for_offset(start, page_spans),
            "page_end": page_for_offset(end_clamped, page_spans),
        }
        doc_id = f"{doc_id_prefix}{len(docs)}"
        docs.append(Document(doc_id=doc_id, text=body, metadata=meta))

    for pos, line in iter_lines_with_offsets(full_text):
        stripped = line.strip()
        if stripped:
            heading = _match_heading(stripped)
            if heading:
                flush()
                state.update(heading)
                segment_start_char = pos
            elif segment_start_char is None:
                segment_start_char = pos
        buffer.append(line)
        segment_last_char = pos + len(line)

    flush()
    return docs
