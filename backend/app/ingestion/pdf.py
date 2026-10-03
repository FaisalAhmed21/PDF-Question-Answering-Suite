"""Multi-strategy PDF parser with automatic fallback.

Parser chain:
  1. pymupdf4llm  → Markdown with structure (primary)
  2. docling      → Complex / OCR-heavy layouts (if installed)
  3. PyMuPDF fitz → Plain text (last resort)
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import List

from app.config import settings
from app.types import ElementType, RawElement

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# Layout difficulty heuristic
# ────────────────────────────────────────────────────────

def layout_looks_hard(pdf_path: str, sample_pages: int = 6) -> bool:
    """Sample the first *sample_pages* pages to detect hard layouts."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        n_pages = min(len(doc), sample_pages)
        sparse = 0
        multi_col = 0
        table_heavy = 0

        for i in range(n_pages):
            page = doc[i]
            text = page.get_text("text")
            width = page.rect.width

            # Sparse / OCR-ish
            text_str = str(text).strip()
            if len(text_str) < 80:
                sparse += 1
                continue

            # Multi-column detection
            blocks = page.get_text("blocks")
            left_blocks = [b for b in blocks if b[0] < width * 0.35]
            right_blocks = [b for b in blocks if b[2] > width * 0.65]
            if left_blocks and right_blocks:
                multi_col += 1

            # Table-heavy
            pipe_count = text_str.count("|")
            tab_count = text_str.count("\t")
            if pipe_count > 10 or tab_count > 10:
                table_heavy += 1

        doc.close()
        hard_ratio = (sparse + multi_col + table_heavy) / max(n_pages, 1)
        return hard_ratio > 0.3
    except Exception:
        return False


# ────────────────────────────────────────────────────────
# Parser implementations
# ────────────────────────────────────────────────────────

def _parse_pymupdf4llm(pdf_path: str) -> List[RawElement]:
    """Primary parser: pymupdf4llm → Markdown pages."""
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    elements: List[RawElement] = []
    for page_data in pages:
        if not isinstance(page_data, dict):
            continue
        page_num = page_data.get("metadata", {}).get("page", 1)
        md = page_data.get("text", "")
        if not md.strip():
            continue
        elements.append(RawElement(
            type=ElementType.TEXT,
            content=f"<!-- page={page_num} -->\n{md}",
            page_number=page_num,
            source_id="pymupdf4llm",
        ))
    return elements


def _parse_docling(pdf_path: str) -> List[RawElement]:
    """Fallback 1: docling for complex layouts."""
    try:
        from docling.document_converter import DocumentConverter # type: ignore
    except ImportError:
        raise ImportError("docling is not installed")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()
    elements: List[RawElement] = []
    for i, part in enumerate(md.split("\n\n")):
        part = part.strip()
        if not part:
            continue
        elements.append(RawElement(
            type=ElementType.TEXT,
            content=part,
            page_number=i + 1,
            source_id="docling",
        ))
    return elements


def _parse_pymupdf_plain(pdf_path: str) -> List[RawElement]:
    """Fallback 2: plain text via PyMuPDF/fitz."""
    import fitz

    doc = fitz.open(pdf_path)
    elements: List[RawElement] = []
    for i in range(len(doc)):
        page = doc[i]
        text = str(page.get_text("text")).strip()
        if text:
            elements.append(RawElement(
                type=ElementType.TEXT,
                content=f"<!-- page={i + 1} -->\n{text}",
                page_number=i + 1,
                source_id="pymupdf",
            ))
    doc.close()
    return elements


# ────────────────────────────────────────────────────────
# Markdown block parser → structured RawElements
# ────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PAGE_RE = re.compile(r"<!--\s*page=(\d+)\s*-->")
_TABLE_LINE_RE = re.compile(r"^\|.*\|$")


def parse_markdown_blocks(elements: List[RawElement]) -> List[RawElement]:
    """Parse raw markdown text into structured RawElement objects."""
    structured: List[RawElement] = []
    current_section = ""
    current_page = 1

    for elem in elements:
        if getattr(elem, "page_number", None):
            current_page = elem.page_number

        lines = elem.content.split("\n")
        text_buffer: List[str] = []
        table_buffer: List[str] = []

        def flush_text(page_to_use):
            nonlocal text_buffer
            if text_buffer:
                content = "\n".join(text_buffer).strip()
                if content:
                    structured.append(RawElement(
                        type=ElementType.TEXT,
                        content=content,
                        page_number=page_to_use,
                        source_id=elem.source_id,
                        section_title=current_section,
                    ))
                text_buffer = []

        def flush_table(page_to_use):
            nonlocal table_buffer
            if table_buffer:
                content = "\n".join(table_buffer).strip()
                if content:
                    structured.append(RawElement(
                        type=ElementType.TABLE,
                        content=content,
                        page_number=page_to_use,
                        source_id=elem.source_id,
                        section_title=current_section,
                    ))
                table_buffer = []

        for line in lines:
            # Page boundary
            page_match = _PAGE_RE.search(line)
            if page_match:
                flush_text(current_page)
                flush_table(current_page)
                current_page = int(page_match.group(1))
                continue

            # Heading
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush_text(current_page)
                flush_table(current_page)
                current_section = heading_match.group(2).strip()
                continue

            # Table line
            if _TABLE_LINE_RE.match(line.strip()):
                flush_text(current_page)
                table_buffer.append(line)
                continue

            # If we were in a table and hit non-table, flush table
            if table_buffer:
                flush_table(current_page)

            text_buffer.append(line)

        flush_text(current_page)
        flush_table(current_page)

    return structured


# ────────────────────────────────────────────────────────
# Main parse entry point
# ────────────────────────────────────────────────────────

async def parse_pdf(pdf_path: str) -> List[RawElement]:
    """Parse a PDF using the configured strategy with fallbacks."""
    strategy = settings.PDF_PARSER.lower()
    hard_layout = False

    if strategy == "auto":
        hard_layout = await asyncio.to_thread(layout_looks_hard, pdf_path)

    parsers = []
    if strategy == "pymupdf4llm" or strategy == "auto":
        parsers.append(("pymupdf4llm", _parse_pymupdf4llm))
    if strategy == "docling" or (strategy == "auto" and hard_layout):
        parsers.insert(0 if hard_layout else 1, ("docling", _parse_docling))
    if strategy == "pymupdf":
        parsers.append(("pymupdf", _parse_pymupdf_plain))

    # Always add plain pymupdf as last resort
    if not any(n == "pymupdf" for n, _ in parsers):
        parsers.append(("pymupdf", _parse_pymupdf_plain))

    for name, parser_fn in parsers:
        try:
            logger.info("Trying PDF parser: %s", name)
            elements = await asyncio.to_thread(parser_fn, pdf_path)
            if elements:
                logger.info("Parser %s succeeded: %d elements", name, len(elements))
                return parse_markdown_blocks(elements)
        except ImportError:
            logger.warning("Parser %s not installed, skipping", name)
        except Exception as exc:
            logger.warning("Parser %s failed: %s", name, exc)

    return []


def count_pdf_pages(pdf_path: str) -> int:
    """Count pages using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0
