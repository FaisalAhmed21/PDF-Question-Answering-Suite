"""Cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

import asyncio
import logging
from typing import List

from app.config import settings
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            from fastembed import TextCrossEncoder # type: ignore

        _reranker = TextCrossEncoder(model_name=settings.RERANK_MODEL)
    return _reranker


async def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: int = 15,
) -> List[RetrievedChunk]:
    """Rerank chunks using a cross-encoder model."""
    if not settings.RERANK_ENABLED or not chunks:
        return chunks[:top_k]

    try:
        reranker = _get_reranker()

        def _do():
            return list(reranker.rerank(query, [c.content for c in chunks]))

        results = await asyncio.to_thread(_do)

        # Results are often just an iterable of floats (scores) matching the input index
        scored = []
        for idx, res in enumerate(results):
            if isinstance(res, float) or isinstance(res, int):
                score = float(res)
            elif isinstance(res, dict):
                idx = res.get("index", idx)
                score = res["score"]
            else:
                idx = getattr(res, "index", idx)
                score = getattr(res, "score", 0.0)
                
            chunk = chunks[idx]
            chunk.rerank_score = score
            scored.append(chunk)

        # Sort by rerank score
        scored.sort(key=lambda c: c.rerank_score or 0, reverse=True)
        return scored[:top_k]

    except Exception as exc:
        logger.warning("Reranker failed, falling back to fusion order: %s", exc)
        return chunks[:top_k]
