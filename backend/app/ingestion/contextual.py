"""Contextual retrieval prefix: prepend short context to each chunk."""

from __future__ import annotations

import asyncio
import logging
from typing import List

from app.config import settings
from app.types import Chunk

logger = logging.getLogger(__name__)


def _heuristic_prefix(chunk: Chunk, filename: str) -> str:
    """Build a prefix from metadata (no LLM)."""
    parts = [f"This chunk is from '{filename}'"]
    if chunk.section_title:
        parts.append(f"section '{chunk.section_title}'")
    parts.append(f"page {chunk.page_number}")
    return ", ".join(parts) + ". "


async def _llm_prefixes(chunks: List[Chunk], filename: str) -> List[str]:
    """Ask the LLM to generate context prefixes in batches of 5."""
    from app.generation.llm_client import get_llm_client

    client = get_llm_client()
    prefixes: List[str] = []

    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        prompt_parts = []
        for j, c in enumerate(batch):
            prompt_parts.append(
                f"Chunk {j + 1} (page {c.page_number}, section: {c.section_title or 'N/A'}):\n"
                f"{c.content[:500]}"
            )

        prompt = (
            f"Document: {filename}\n\n"
            + "\n\n---\n\n".join(prompt_parts)
            + "\n\n---\n\n"
            "Write a 1-sentence description of each chunk's context to prepend for better "
            "search retrieval. Output ONLY the prefixes separated by `|||`."
        )

        try:
            resp = await client.generate(
                prompt=prompt,
                system="You generate concise context prefixes for document chunks.",
                max_tokens=300,
            )
            parts = [p.strip() for p in resp.split("|||")]
            # Pad or trim to match batch
            while len(parts) < len(batch):
                parts.append(_heuristic_prefix(batch[len(parts)], filename))
            prefixes.extend(parts[: len(batch)])
        except Exception as exc:
            logger.warning("LLM prefix generation failed: %s, using heuristic", exc)
            for c in batch:
                prefixes.append(_heuristic_prefix(c, filename))

    return prefixes


async def apply_contextual_prefixes(
    chunks: List[Chunk],
    filename: str,
) -> List[Chunk]:
    """Prepend contextual prefix to each chunk's contextualized_content."""
    use_llm = settings.CONTEXTUAL_RETRIEVAL_LLM and (
        settings.GROQ_API_KEY or settings.GEMINI_API_KEY
    )

    if use_llm:
        try:
            prefixes = await _llm_prefixes(chunks, filename)
        except Exception:
            prefixes = [_heuristic_prefix(c, filename) for c in chunks]
    else:
        prefixes = [_heuristic_prefix(c, filename) for c in chunks]

    for chunk, prefix in zip(chunks, prefixes):
        if not prefix.endswith(" "):
            prefix += " "
        chunk.contextualized_content = prefix + chunk.content

    return chunks
