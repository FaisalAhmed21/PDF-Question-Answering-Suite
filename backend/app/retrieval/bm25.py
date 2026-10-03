"""From-scratch BM25 sparse search scorer."""

from __future__ import annotations

import math
import re
import logging
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChunkRecord
from app.db.session import async_session
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, regex-based, skip tokens <= 1 char."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class BM25:
    """BM25 scorer with k1=1.5, b=0.75."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.N: int = 0
        self.corpus_tokens: List[List[str]] = []
        self.doc_indices: List[int] = []

    def fit(self, documents: List[str]):
        """Index a corpus of documents."""
        self.N = len(documents)
        self.corpus_tokens = []
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = _tokenize(doc)
            self.corpus_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            seen = set()
            for t in tokens:
                if t not in seen:
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
                    seen.add(t)

        total = sum(self.doc_lengths)
        self.avg_dl = total / self.N if self.N > 0 else 1.0

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """Score a single document against a query."""
        query_tokens = _tokenize(query)
        doc_tokens = self.corpus_tokens[doc_idx]
        dl = self.doc_lengths[doc_idx]

        # Count term frequencies
        tf_map: Dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        total = 0.0
        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            idf = self._idf(qt)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_dl)
            total += idf * numerator / denominator

        return total

    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """Return top_k (doc_index, score) pairs."""
        scores = []
        for i in range(self.N):
            s = self.score(query, i)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


async def bm25_search(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """Load chunk records from DB, build BM25 index, search."""
    async with async_session() as session:
        stmt = select(ChunkRecord)
        if document_id:
            stmt = stmt.where(ChunkRecord.document_id == document_id)
        result = await session.execute(stmt)
        records = result.scalars().all()

    if not records:
        return []

    # Use contextualized_content if available
    texts = [
        r.contextualized_content or r.content
        for r in records
    ]

    bm25 = BM25()
    bm25.fit(texts)
    results = bm25.search(query, top_k=top_k)

    chunks = []
    for idx, score in results:
        rec = records[idx]
        chunks.append(RetrievedChunk(
            chunk_id=rec.id,
            document_id=rec.document_id,
            content=rec.content,
            contextualized_content=rec.contextualized_content or "",
            page_number=rec.page_number,
            section_title=rec.section_title or "",
            chunk_type=rec.chunk_type,
            chunk_index=rec.chunk_index,
            char_offset=rec.char_offset,
            score=score,
        ))

    return chunks
