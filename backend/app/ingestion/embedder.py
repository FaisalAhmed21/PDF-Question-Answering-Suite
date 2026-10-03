"""Text embedding with fastembed (local) or Gemini (cloud)."""

from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_fastembed_model = None
BATCH_SIZE = 32


def _get_fastembed():
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding

        _fastembed_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
    return _fastembed_model


async def _embed_fastembed(texts: List[str]) -> List[List[float]]:
    """Embed using local fastembed model."""
    model = _get_fastembed()

    def _do():
        return [list(v) for v in model.embed(texts)]

    return await asyncio.to_thread(_do)


async def _embed_gemini(texts: List[str]) -> List[List[float]]:
    """Embed using Gemini API."""
    model_name = settings.EMBEDDING_MODEL
    if model_name.startswith("BAAI/") or "/" in model_name:
        model_name = "gemini-embedding-2"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"
        f"?key={settings.GEMINI_API_KEY}"
    )
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            resp = await client.post(url, json={"content": {"parts": [{"text": text}]}})
            resp.raise_for_status()
            data = resp.json()
            results.append(data["embedding"]["values"])

    return results


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in batches of BATCH_SIZE."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        if provider == "gemini":
            embs = await _embed_gemini(batch)
        else:
            embs = await _embed_fastembed(batch)
        all_embeddings.extend(embs)

    return all_embeddings


async def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    results = await embed_texts([text])
    return results[0]
