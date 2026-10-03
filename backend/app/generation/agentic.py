"""Agentic RAG pipeline: self-correcting retrieval + generation loop."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, List, Optional

from app.config import settings
from app.generation.grounding_check import (
    groundedness_check,
    relevance_gate,
    _log_gate_failure,
)
from app.generation.llm_client import get_llm_client
from app.generation.prompt import (
    RATE_LIMIT_TEXT,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    build_prompt,
    extract_citations,
    finalize_answer,
    format_context,
    is_refusal_answer,
    is_transient_error_answer,
)
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.query_transform import (
    adaptive_second_hop,
    chat_aware_rewrite,
    generate_hyde,
    is_multihop,
)
from app.retrieval.reranker import rerank
from app.types import GenerationResult, RetrievedChunk

logger = logging.getLogger(__name__)


def _merge_ranked(
    existing: List[RetrievedChunk],
    new: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """Merge two ranked lists, dedup by chunk_id, keep highest score."""
    by_id: Dict[str, RetrievedChunk] = {}
    for c in existing:
        if c.chunk_id not in by_id or c.score > by_id[c.chunk_id].score:
            by_id[c.chunk_id] = c
    for c in new:
        if c.chunk_id not in by_id or c.score > by_id[c.chunk_id].score:
            by_id[c.chunk_id] = c
    merged = list(by_id.values())
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def _refusal_result(best_score: float, reason: str) -> GenerationResult:
    return GenerationResult(
        answer=REFUSAL_TEXT,
        citations=[],
        refused=True,
        refusal_reason=reason,
        grounded=True,
        retrieval_scores={"best_dense": best_score},
    )


def _error_result(best_score: float, reason: str) -> GenerationResult:
    return GenerationResult(
        answer=RATE_LIMIT_TEXT,
        citations=[],
        refused=True,
        refusal_reason=reason,
        grounded=True,
        retrieval_scores={"best_dense": best_score},
    )


async def _yield_text_chunks(text: str):
    """Yield fake stream chunks for a fully-buffered answer."""
    chunk_size = 24
    for i in range(0, len(text), chunk_size):
        yield json.dumps({"type": "token", "content": text[i : i + chunk_size]}) + "\n"


async def _stream_done(
    text: str,
    *,
    refused: bool,
    reason: str,
    citations: Optional[list] = None,
    grounded: bool = True,
):
    async for line in _yield_text_chunks(text):
        yield line
    yield json.dumps({
        "type": "done",
        "citations": citations or [],
        "refused": refused,
        "refusal_reason": reason,
        "grounded": grounded,
    }) + "\n"


async def agentic_rag(
    query: str,
    document_ids: List[str],
    history: Optional[List[dict]] = None,
    doc_names: Optional[Dict[str, str]] = None,
    stream: bool = False,
) -> GenerationResult | AsyncIterator[str]:
    """Full agentic RAG pipeline."""
    history = history or []
    doc_names = doc_names or {}

    has_llm = bool(settings.GROQ_API_KEY or settings.GEMINI_API_KEY)

    # 1. Rewrite query (chat-aware)
    effective_query = query
    if has_llm and history:
        effective_query = await chat_aware_rewrite(query, history)

    # Optional HyDE
    search_query = effective_query
    if has_llm and settings.HYDE_ENABLED:
        search_query = await generate_hyde(effective_query)

    # 2. Retrieve (hybrid search)
    all_chunks: List[RetrievedChunk] = []
    for doc_id in document_ids:
        chunks = await hybrid_search(search_query, document_id=doc_id, top_k=20)
        all_chunks.extend(chunks)

    all_chunks.sort(key=lambda x: x.score, reverse=True)

    # 3. Rerank
    all_chunks = await rerank(effective_query, all_chunks, top_k=15)

    # 4. Relevance gate (before any LLM answer generation)
    passes, best_score = relevance_gate(all_chunks, query=effective_query)
    if not passes and has_llm:
        await _log_gate_failure(query, "relevance", best_score)

    # 5. Multi-hop / borderline second hop (agentic only)
    # Only when retrieval is close — never burn LLM calls on clear misses.
    multihop = is_multihop(effective_query)
    borderline = 0.25 <= best_score < settings.RELEVANCE_THRESHOLD
    if (
        settings.AGENTIC_ENABLED
        and has_llm
        and (multihop or borderline)
        and all_chunks
        and not passes
    ):
        second_query = await adaptive_second_hop(effective_query, all_chunks)
        if second_query:
            for doc_id in document_ids:
                extra = await hybrid_search(second_query, document_id=doc_id, top_k=10)
                all_chunks = _merge_ranked(all_chunks, extra)
            all_chunks = await rerank(effective_query, all_chunks, top_k=15)
            passes2, best_score2 = relevance_gate(all_chunks, query=effective_query)
            if passes2:
                passes = True
                best_score = max(best_score, best_score2)

    if not has_llm:
        return _refusal_result(best_score, "No LLM key configured")

    # Out-of-document → refuse immediately (no LLM, no citations)
    if not passes:
        return _refusal_result(best_score, "Question not covered by document context")

    # 6. Generate answer
    top_chunks = all_chunks[:10]
    prompt = build_prompt(effective_query, top_chunks)
    context_str = format_context(top_chunks)

    if stream:
        async def _stream():
            logger.info("[STREAM] Starting generation (buffered for grounding)")
            client = get_llm_client()
            answer_text = ""

            try:
                parts: List[str] = []
                async for token in client.generate_stream(
                    prompt=prompt, system=SYSTEM_PROMPT, max_tokens=2048
                ):
                    parts.append(token)
                answer_text = "".join(parts).strip()
            except Exception as exc:
                logger.warning("[STREAM] Streaming exception: %s", exc)

            if not answer_text:
                logger.warning("[STREAM] Empty stream, trying non-stream generate")
                try:
                    answer_text = (await client.generate(
                        prompt=prompt, system=SYSTEM_PROMPT, max_tokens=2048
                    )).strip()
                except Exception as exc:
                    logger.error("[STREAM] Generate failed: %s", exc, exc_info=True)
                    async for line in _stream_done(
                        RATE_LIMIT_TEXT,
                        refused=True,
                        reason="LLM temporarily unavailable",
                    ):
                        yield line
                    return

            answer_text = finalize_answer(answer_text)

            if is_transient_error_answer(answer_text):
                async for line in _stream_done(
                    RATE_LIMIT_TEXT,
                    refused=True,
                    reason="LLM temporarily unavailable",
                ):
                    yield line
                return

            if is_refusal_answer(answer_text):
                async for line in _stream_done(
                    REFUSAL_TEXT,
                    refused=True,
                    reason="Not found in document context",
                ):
                    yield line
                return

            grounded = True
            try:
                grounded = await groundedness_check(answer_text, context_str)
            except Exception as exc:
                logger.warning("[STREAM] Groundedness check failed: %s", exc)
                grounded = False

            if not grounded:
                await _log_gate_failure(query, "groundedness", best_score, answer_text)
                async for line in _stream_done(
                    REFUSAL_TEXT,
                    refused=True,
                    reason="Answer not supported by document context",
                    grounded=False,
                ):
                    yield line
                return

            citations = extract_citations(
                top_chunks, doc_names=doc_names, query=query, answer=answer_text
            )
            async for line in _stream_done(
                answer_text,
                refused=False,
                reason="",
                citations=[c.model_dump() for c in citations],
                grounded=True,
            ):
                yield line

        return _stream()

    # Non-streaming
    client = get_llm_client()
    try:
        answer = await client.generate(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=2048)
    except Exception as exc:
        logger.error("Generate failed: %s", exc, exc_info=True)
        return _error_result(best_score, "LLM temporarily unavailable")

    answer = finalize_answer(answer)

    if is_transient_error_answer(answer):
        return _error_result(best_score, "LLM temporarily unavailable")

    if is_refusal_answer(answer):
        return _refusal_result(best_score, "Not found in document context")

    grounded = await groundedness_check(answer, context_str)

    if not grounded and settings.AGENTIC_ENABLED:
        await _log_gate_failure(query, "groundedness", best_score, answer)
        logger.warning("Answer ungrounded, attempting self-correction")

        second_query = await adaptive_second_hop(effective_query, top_chunks)
        if second_query:
            for doc_id in document_ids:
                extra = await hybrid_search(second_query, document_id=doc_id, top_k=10)
                all_chunks = _merge_ranked(all_chunks, extra)
            all_chunks = await rerank(effective_query, all_chunks, top_k=15)
            top_chunks = all_chunks[:10]

        prompt = build_prompt(effective_query, top_chunks)
        context_str = format_context(top_chunks)
        try:
            answer = await client.generate(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=2048)
        except Exception:
            return _error_result(best_score, "LLM temporarily unavailable")

        answer = finalize_answer(answer)

        if is_transient_error_answer(answer):
            return _error_result(best_score, "LLM temporarily unavailable")

        if is_refusal_answer(answer):
            return _refusal_result(best_score, "Not found in document context")

        grounded = await groundedness_check(answer, context_str)
        if not grounded:
            await _log_gate_failure(query, "groundedness_retry", best_score, answer)
            return _refusal_result(best_score, "Answer could not be grounded after retry")
    elif not grounded:
        await _log_gate_failure(query, "groundedness", best_score, answer)
        return _refusal_result(best_score, "Answer not supported by document context")

    answer = finalize_answer(answer)
    citations = extract_citations(
        top_chunks, doc_names=doc_names, query=query, answer=answer
    )
    return GenerationResult(
        answer=answer,
        citations=citations,
        refused=False,
        grounded=True,
        retrieval_scores={"best_dense": best_score},
    )
