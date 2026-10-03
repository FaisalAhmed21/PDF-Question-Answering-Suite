"""Structure-aware and fixed-size chunking strategies."""

from __future__ import annotations

import logging
import re
from typing import List

from app.types import ChunkType, ElementType, RawElement, Chunk
from app.ingestion.nlp_enhancements import semantic_chunking
from app.ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


# ── Smart text splitter ────────────────────────────────

def _smart_split(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split text preferring paragraph, sentence, then line boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try paragraph break
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start:
            # Try sentence break
            cut = text.rfind(". ", start, end)
            if cut != -1:
                cut += 2  # include the period and space
        if cut == -1 or cut <= start:
            # Try line break
            cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            # Hard cut
            cut = end

        chunks.append(text[start:cut])
        start = max(cut - overlap, start + 1)

    return chunks


# ── Structure-aware chunking ───────────────────────────

def _chunk_type_from_element(elem: RawElement) -> ChunkType:
    mapping = {
        ElementType.TABLE: ChunkType.TABLE,
        ElementType.IMAGE: ChunkType.IMAGE,
        ElementType.TRANSCRIPT: ChunkType.TRANSCRIPT,
    }
    return mapping.get(elem.type, ChunkType.TEXT)


def structure_aware_chunk(
    elements: List[RawElement],
    document_id: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> List[Chunk]:
    """Group consecutive text elements by section/page, tables as standalone chunks."""
    chunks: List[Chunk] = []
    idx = 0
    text_group: List[str] = []
    group_page = 1
    group_section = ""
    group_offset = 0

    def flush_group():
        nonlocal text_group, idx
        if not text_group:
            return
        merged = "\n\n".join(text_group)
        for part in _smart_split(merged, max_chars, overlap):
            chunks.append(Chunk(
                document_id=document_id,
                content=part,
                page_number=group_page,
                section_title=group_section,
                chunk_type=ChunkType.TEXT,
                char_offset=group_offset,
                chunk_index=idx,
            ))
            idx += 1
        text_group = []

    for elem in elements:
        # Tables, images, transcripts → standalone
        if elem.type in (ElementType.TABLE, ElementType.IMAGE, ElementType.TRANSCRIPT):
            flush_group()
            chunks.append(Chunk(
                document_id=document_id,
                content=elem.content,
                page_number=elem.page_number,
                section_title=elem.section_title,
                chunk_type=_chunk_type_from_element(elem),
                char_offset=0,
                chunk_index=idx,
            ))
            idx += 1
            continue

        # Group text by section + page
        if (elem.section_title != group_section or elem.page_number != group_page) and text_group:
            flush_group()

        if not text_group:
            group_page = elem.page_number
            group_section = elem.section_title
            group_offset = 0

        text_group.append(elem.content)

    flush_group()
    return chunks


# ── Fixed-size chunking (fallback) ─────────────────────

async def semantic_fallback_chunk(
    elements: List[RawElement],
    document_id: str,
) -> List[Chunk]:
    """Fallback chunking using NLP semantic boundaries instead of fixed sliding windows."""
    chunks: List[Chunk] = []
    idx = 0

    for elem in elements:
        if elem.type == ElementType.TABLE:
            chunks.append(Chunk(
                document_id=document_id,
                content=elem.content,
                page_number=elem.page_number,
                section_title=elem.section_title,
                chunk_type=ChunkType.TABLE,
                char_offset=0,
                chunk_index=idx,
            ))
            idx += 1
            continue

        # Semantic chunking for text
        semantic_parts = await semantic_chunking(elem.content, embed_texts)
        start_offset = 0
        for part in semantic_parts:
            chunks.append(Chunk(
                document_id=document_id,
                content=part,
                page_number=elem.page_number,
                section_title=elem.section_title,
                chunk_type=ChunkType.TEXT,
                char_offset=start_offset,
                chunk_index=idx,
            ))
            idx += 1
            start_offset += len(part) + 1  # approximate tracking

    return chunks


# ── Main chunking entry point ──────────────────────────

async def chunk_elements(
    elements: List[RawElement],
    document_id: str,
) -> List[Chunk]:
    """Primary: structure-aware; fallback: semantic chunking."""
    chunks = structure_aware_chunk(elements, document_id)
    if not chunks:
        logger.warning("Structure-aware chunking produced 0 chunks, falling back to semantic chunking")
        chunks = await semantic_fallback_chunk(elements, document_id)
    return chunks
