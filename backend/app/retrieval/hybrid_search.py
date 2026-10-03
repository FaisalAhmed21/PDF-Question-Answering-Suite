"""Hybrid search: dense + sparse (BM25) with Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from app.config import settings
from app.ingestion.embedder import embed_query
from app.retrieval.bm25 import bm25_search
from app.retrieval.vector_store import get_vector_store
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    result_lists: List[List[RetrievedChunk]],
    k: int = 60,
) -> List[RetrievedChunk]:
    """Fuse multiple ranked lists using RRF: score(d) = Σ 1/(k + rank).

    Preserves the best dense cosine score on each chunk so the relevance gate
    can still use real similarity (RRF values are tiny ~0.01–0.03).
    """
    scores: Dict[str, float] = {}
    chunks_map: Dict[str, RetrievedChunk] = {}
    best_dense: Dict[str, float] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            cid = chunk.chunk_id
            rrf_score = 1.0 / (k + rank + 1)
            # Only cosine similarities count as dense_score — never BM25 raw scores
            dense = chunk.dense_score if chunk.dense_score is not None else None
            if cid in scores:
                scores[cid] += rrf_score
                if dense is not None:
                    best_dense[cid] = max(best_dense.get(cid, 0.0), dense)
                if dense is not None and dense >= (chunks_map[cid].dense_score or 0.0):
                    chunks_map[cid] = chunk
            else:
                scores[cid] = rrf_score
                if dense is not None:
                    best_dense[cid] = dense
                chunks_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, fused_score in ranked:
        c = chunks_map[cid]
        c.score = fused_score
        if cid in best_dense:
            c.dense_score = best_dense[cid]
        result.append(c)

    return result


async def hybrid_search(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """Run dense + sparse search in parallel and fuse with RRF."""
    # Dense search
    query_vec = await embed_query(query)
    vs = get_vector_store()

    if settings.HYBRID_ENABLED:
        dense_task = vs.search_dense(query_vec, top_k=top_k, document_id=document_id)
        sparse_task = bm25_search(query, document_id=document_id, top_k=top_k)

        try:
            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task
            )
        except Exception as exc:
            logger.warning("Sparse search failed, falling back to dense-only: %s", exc)
            dense_results = await vs.search_dense(
                query_vec, top_k=top_k, document_id=document_id
            )
            sparse_results = []

        if sparse_results:
            fused = _reciprocal_rank_fusion([dense_results, sparse_results])
        else:
            fused = dense_results
    else:
        fused = await vs.search_dense(query_vec, top_k=top_k, document_id=document_id)

    return fused[:top_k]
