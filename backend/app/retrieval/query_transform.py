"""Query transformation: rewrite, multi-hop detection, HyDE."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.config import settings
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Multi-hop detection patterns
_MULTIHOP_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference|both|across|relationship|"
    r"similarities|contrast|between)\b",
    re.IGNORECASE,
)

_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her)\b",
    re.IGNORECASE,
)

_REWRITE_JUNK = re.compile(
    r"(we need to|rewrite|standalone|search query|pronouns|instruction|"
    r"chat history|latest (user )?question|as a standalone|resolve pronouns|"
    r"final answer|here is|so (the )?answer)",
    re.IGNORECASE,
)


def is_multihop(query: str) -> bool:
    """Detect if a query requires multi-hop reasoning."""
    if _MULTIHOP_PATTERNS.search(query):
        return True
    words = query.split()
    if len(words) > 8 and " and " in query.lower():
        return True
    return False


def _needs_rewrite(query: str) -> bool:
    """Only rewrite when the question depends on prior turns."""
    return bool(_PRONOUNS.search(query))


def _sanitize_rewrite(original: str, rewritten: str) -> str:
    """Accept a rewrite only if it looks like a short search query."""
    text = (rewritten or "").strip().strip('"').strip("'")
    if not text:
        return original

    # Reasoning models often dump CoT — take last non-empty line if needed
    if _REWRITE_JUNK.search(text) or len(text) > 160 or text.count("\n") > 2:
        lines = [ln.strip(" \"'`") for ln in text.splitlines() if ln.strip()]
        # Prefer a short trailing candidate
        for candidate in reversed(lines):
            if (
                3 < len(candidate) <= 120
                and not _REWRITE_JUNK.search(candidate)
                and "?" not in candidate[:20]  # avoid echoing instructions
            ):
                text = candidate
                break
        else:
            return original

    if _REWRITE_JUNK.search(text) or len(text) > 160:
        return original

    # Must still share some signal with the original question
    orig_terms = set(re.findall(r"[a-zA-Z]{3,}", original.lower()))
    new_terms = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
    if orig_terms and not (orig_terms & new_terms):
        return original

    return text


async def chat_aware_rewrite(
    query: str,
    history: List[dict],
) -> str:
    """Rewrite query resolving pronouns using chat history.

    Skips rewrite when the question is already standalone, and rejects
    reasoning-model dumps that would poison retrieval.
    """
    if not history or not _needs_rewrite(query):
        return query

    try:
        from app.generation.llm_client import get_llm_client

        client = get_llm_client()
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in history[-6:]
        )
        prompt = (
            f"Chat history:\n{history_text}\n\n"
            f"Latest question: {query}\n\n"
            "Rewrite the latest question as a short standalone search query. "
            "Resolve pronouns from history. "
            "Output ONLY the search query text — no explanation."
        )
        rewritten = await client.generate(
            prompt=prompt,
            system=(
                "You rewrite follow-up questions into short standalone search queries. "
                "Reply with ONLY the query, nothing else."
            ),
            max_tokens=48,
        )
        cleaned = _sanitize_rewrite(query, rewritten)
        if cleaned != query:
            logger.info("Query rewritten: '%s' → '%s'", query, cleaned)
        else:
            logger.info("Rewrite discarded; keeping original query")
        return cleaned
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)

    return query


async def adaptive_second_hop(
    query: str,
    retrieved: List[RetrievedChunk],
) -> Optional[str]:
    """Ask LLM for a second-hop query if info is missing."""
    try:
        from app.generation.llm_client import get_llm_client

        client = get_llm_client()
        context = "\n\n".join(
            f"[{i + 1}] {c.content[:300]}" for i, c in enumerate(retrieved[:5])
        )
        prompt = (
            f"User question: {query}\n\n"
            f"Already retrieved:\n{context}\n\n"
            "Output ONE short search query that would find missing facts. "
            "If nothing is missing, reply NONE."
        )
        resp = await client.generate(
            prompt=prompt,
            system=(
                "You help identify missing information for answering questions. "
                "Reply with ONLY a short query or NONE."
            ),
            max_tokens=48,
        )
        raw = (resp or "").strip().strip('"').strip("'")
        if not raw or raw.upper() == "NONE" or "NONE" == raw.split()[0].upper():
            return None

        cleaned = _sanitize_rewrite(query, raw)
        # Discard if sanitize fell back to original or still looks like junk
        if cleaned == query or len(cleaned) < 4 or len(cleaned) > 120:
            return None
        if _REWRITE_JUNK.search(cleaned):
            return None

        logger.info("Second-hop query: %s", cleaned)
        return cleaned
    except Exception as exc:
        logger.warning("Second-hop query failed: %s", exc)
        return None


async def generate_hyde(query: str) -> str:
    """Generate a hypothetical answer passage for HyDE embedding."""
    if not settings.HYDE_ENABLED:
        return query

    try:
        from app.generation.llm_client import get_llm_client

        client = get_llm_client()
        prompt = (
            f"Question: {query}\n\n"
            "Write a 3-5 sentence passage that would be a good answer to this question. "
            "Write it as if you are quoting a reference document."
        )
        resp = await client.generate(
            prompt=prompt,
            system="You generate hypothetical document passages.",
            max_tokens=250,
        )
        if resp and len(resp) > 20:
            return resp
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)

    return query
