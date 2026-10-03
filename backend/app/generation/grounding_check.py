"""Groundedness verification: relevance gate + post-generation check."""

from __future__ import annotations

import logging
import re
from typing import List

from app.config import settings
from app.db.models import EvalLog
from app.db.session import async_session
from app.generation.llm_client import get_llm_client
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)

_STOP = {
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "are", "with", "from", "that", "this", "have", "has",
    "been", "about", "into", "your", "their", "there", "does", "did", "can",
    "could", "would", "should", "please", "tell", "give", "explain",
}


async def _log_gate_failure(question: str, gate: str, top_score: float, draft: str = ""):
    """Log gate failures to EvalLog table."""
    try:
        async with async_session() as session:
            log = EvalLog(
                question=question,
                gate=gate,
                top_score=top_score,
                answer_draft=draft[:500] if draft else None,
            )
            session.add(log)
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to log gate failure: %s", exc)


def _query_terms(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if w not in _STOP
    }


def _best_dense(chunks: List[RetrievedChunk]) -> float:
    """Best cosine similarity, preferring dense_score over RRF fusion score."""
    best = 0.0
    for c in chunks:
        if c.dense_score is not None:
            best = max(best, c.dense_score)
        else:
            # Dense-only path: score is cosine. Skip tiny RRF-looking values.
            if c.score >= 0.05:
                best = max(best, c.score)
    return best


def query_chunk_overlap(query: str, chunks: List[RetrievedChunk], top_n: int = 8) -> float:
    """Fraction of query terms that appear in the top retrieved chunks."""
    q_words = _query_terms(query)
    if not q_words or not chunks:
        return 0.0

    corpus = " ".join(c.content for c in chunks[:top_n]).lower()
    hits = sum(1 for w in q_words if w in corpus)
    return hits / len(q_words)


def relevance_gate(
    chunks: List[RetrievedChunk],
    query: str = "",
) -> tuple[bool, float]:
    """Check if retrieval results are relevant enough to answer.

    Dense cosine is the primary signal. Cross-encoder logits are often
    negative even for relevant pairs, so rerank must not hard-block.
    Lexical overlap only vetoes clear out-of-domain misses.
    """
    if not chunks:
        return False, 0.0

    best_dense = _best_dense(chunks)
    best_rerank = max(
        (c.rerank_score for c in chunks if c.rerank_score is not None),
        default=None,
    )
    overlap = query_chunk_overlap(query, chunks) if query else 1.0

    dense_ok = best_dense >= settings.RELEVANCE_THRESHOLD
    # Soft rerank rescue for slightly weaker dense matches (logits, not 0-1)
    rerank_rescue = (
        best_rerank is not None
        and best_rerank >= max(settings.RERANK_THRESHOLD, 0.5)
        and best_dense >= 0.22
    )
    passes = dense_ok or rerank_rescue

    # Lexical veto — only for clearly unsupported questions
    if query and passes:
        if overlap <= 0.0:
            logger.info(
                "Relevance gate blocked by zero lexical overlap "
                "(dense=%.3f, rerank=%s)",
                best_dense,
                f"{best_rerank:.3f}" if best_rerank is not None else "n/a",
            )
            passes = False
        elif overlap < 0.34 and best_dense < 0.50:
            logger.info(
                "Relevance gate blocked by weak overlap+dense "
                "(overlap=%.2f, dense=%.3f)",
                overlap,
                best_dense,
            )
            passes = False

    logger.info(
        "Relevance gate: passes=%s dense=%.3f rerank=%s overlap=%.2f",
        passes,
        best_dense,
        f"{best_rerank:.3f}" if best_rerank is not None else "n/a",
        overlap,
    )
    return passes, best_dense


async def groundedness_check(
    answer: str,
    context: str,
) -> bool:
    """Post-generation check: verify answer is supported by context."""
    if not settings.GROUNDEDNESS_ENABLED:
        return True

    try:
        client = get_llm_client()
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Check if the ANSWER is completely supported by the CONTEXT. "
            "If the answer contains any information not present in the context, "
            "output UNGROUNDED. Otherwise output GROUNDED."
        )
        resp = await client.generate(
            prompt=prompt,
            system=(
                "You verify if answers are grounded in provided context. "
                "If it is grounded, your final word MUST be GROUNDED. "
                "If it is not, your final word MUST be UNGROUNDED."
            ),
            max_tokens=1024,
        )
        result = resp.strip().upper()
        last_word = result.split()[-1] if result.split() else ""
        if last_word == "GROUNDED":
            is_grounded = True
        elif last_word == "UNGROUNDED":
            is_grounded = False
        else:
            is_grounded = "GROUNDED" in result and "UNGROUNDED" not in result
        logger.info("Groundedness check: %s", "GROUNDED" if is_grounded else "UNGROUNDED")
        return is_grounded
    except Exception as exc:
        logger.warning("Groundedness check failed: %s", exc)
        return False
