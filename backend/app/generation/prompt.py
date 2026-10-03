"""Prompt engineering for grounded Q&A."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.types import Citation, RetrievedChunk

REFUSAL_TEXT = (
    "This question is outside the scope of the uploaded document(s). "
    "I can only answer questions that are covered in the PDF content."
)

RATE_LIMIT_TEXT = (
    "The answer service is temporarily unavailable. Please try again in a moment."
)

SYSTEM_PROMPT = (
    "You are a strictly grounded answering assistant. "
    "Answer the question using ONLY the provided context from uploaded PDF documents. "
    "If the context does not contain enough information to answer, reply exactly with:\n"
    f'"{REFUSAL_TEXT}"\n'
    "Do not use outside knowledge. Do not guess. "
    "When you can answer from the context, write ONLY the final answer in 1-3 sentences "
    "and include inline citations like [1] or [2] matching the numbered context blocks. "
    "Do not include reasoning, instructions, or meta commentary."
)


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered context."""
    parts = []
    for i, chunk in enumerate(chunks):
        header = f"[{i + 1}] page {chunk.page_number}"
        if chunk.section_title:
            header += f" — {chunk.section_title}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """Build the user prompt with formatted context."""
    context = format_context(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using ONLY the context above. "
        "If the answer is not in the context, reply exactly with the refusal sentence "
        "from your instructions — do not invent facts. "
        "If you can answer, write one short factual answer and include inline "
        "citations like [1], [2] for the chunks you used. "
        "Output the answer only — no reasoning steps."
    )


def is_refusal_answer(text: str) -> bool:
    """Detect model/system refusal wording."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if REFUSAL_TEXT.lower() in lowered:
        return True
    markers = (
        "don't have enough context",
        "do not have enough context",
        "outside the scope of the uploaded",
        "not covered in the",
        "no relevant information in the document",
        "cannot find this in the document",
        "isn't in the document",
        "is not in the document",
        "not mentioned in the",
        "not found in the provided context",
        "not present in the context",
        "i don't know based on",
        "i do not know based on",
    )
    return any(m in lowered for m in markers)


def is_transient_error_answer(text: str) -> bool:
    """Detect rate-limit / capacity style messages that must not be cited."""
    lowered = (text or "").strip().lower()
    markers = (
        "high demand",
        "experiencing high",
        "try again in a moment",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "capacity",
    )
    return any(m in lowered for m in markers)


def finalize_answer(text: str) -> str:
    """Final pass to strip reasoning leaks before showing the user."""
    from app.generation.llm_client import LLMClient

    return LLMClient._clean_cot(text or "")


_OCR_NOISE = re.compile(
    r"(?:[$_%&()]{2,}|&amp;|_\$|\\_|\\\(|\\\))"
)


def _clean_snippet(text: str) -> str:
    """Remove PDF OCR garbage while keeping the readable sentence."""
    cleaned = _OCR_NOISE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-;:")
    return cleaned


def _content_words(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "are", "with", "from", "that", "this", "have",
        "has", "been", "what", "which", "where", "when", "who",
    }
    return {
        w
        for w in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
        if w not in stop
    }


def _best_supporting_sentence(chunk_text: str, query: str, answer: str) -> str:
    """Pick the sentence in the chunk that best supports the answer."""
    # Prefer sentence boundaries; fall back to chunk text
    raw = chunk_text or ""
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    sentences = [_clean_snippet(s) for s in parts if _clean_snippet(s)]
    if not sentences:
        return _clean_snippet(raw)[:200]

    answer_words = _content_words(re.sub(r"\[\d+\]", "", answer))
    query_words = _content_words(query)

    best = sentences[0]
    best_score = -1.0
    for s in sentences:
        s_words = _content_words(s)
        if not s_words:
            continue
        # Weight answer overlap higher — highlight what was actually used
        ans_hit = len(answer_words & s_words)
        qry_hit = len(query_words & s_words)
        score = (ans_hit * 3.0) + (qry_hit * 1.0)
        # Prefer denser support relative to sentence length
        score += ans_hit / max(len(s_words), 1)
        if score > best_score:
            best_score = score
            best = s

    # If nothing overlaps the answer, keep best query match but require signal
    if best_score <= 0 and answer_words:
        # Try finding a contiguous phrase from the answer inside the chunk
        answer_plain = _clean_snippet(re.sub(r"\[\d+\]", "", answer))
        for s in sentences:
            if any(
                w in s.lower()
                for w in answer_words
                if len(w) > 4
            ):
                return s
    return best


def extract_citations(
    chunks: List[RetrievedChunk],
    doc_names: Optional[Dict[str, str]] = None,
    query: str = "",
    answer: str = "",
) -> List[Citation]:
    """Build page-level citations for a grounded answer.

    If the answer contains [n] markers, only those chunks are cited.
    Snippets are the sentences that actually support the answer text.
    Never cite refusals or transient error messages.
    """
    if not chunks or is_refusal_answer(answer) or is_transient_error_answer(answer):
        return []

    names = doc_names or {}

    cited_indices: set[int] = set()
    if answer:
        for match in re.findall(r"\[(\d+)\]", answer):
            idx = int(match) - 1
            if 0 <= idx < len(chunks):
                cited_indices.add(idx)

    if cited_indices:
        selected = [(i, chunks[i]) for i in sorted(cited_indices)]
    else:
        # No markers — do not guess multiple pages; use top chunk only
        selected = [(0, chunks[0])]

    seen: set[tuple[str, int]] = set()
    citations: List[Citation] = []

    for i, chunk in selected:
        key = (chunk.document_id, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)

        snippet = _best_supporting_sentence(chunk.content, query, answer)
        if not snippet:
            continue

        citations.append(
            Citation(
                citation_index=i + 1,
                document_id=chunk.document_id,
                document_name=names.get(chunk.document_id, ""),
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                snippet=snippet,
            )
        )
    return citations
