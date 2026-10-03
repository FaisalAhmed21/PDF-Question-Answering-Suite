"""Qdrant vector store for chunk storage and dense retrieval."""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.db.models import ChunkRecord
from app.types import RetrievedChunk

logger = logging.getLogger(__name__)

_vector_store: Optional["VectorStore"] = None


class VectorStore:
    """Qdrant-backed vector store with cosine similarity search."""

    def __init__(self):
        if settings.QDRANT_URL:
            self.client = QdrantClient(
                url=settings.QDRANT_URL.strip(), 
                api_key=(settings.QDRANT_API_KEY or "").strip() or None,
                timeout=120.0
            )
        else:
            self.client = QdrantClient(path=settings.QDRANT_PATH, timeout=120.0)
        self.collection = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        """Create or recreate collection if dimensions mismatch."""
        dim = settings.EMBEDDING_DIM
        
        exists = self.client.collection_exists(self.collection)
        if exists:
            info = self.client.get_collection(self.collection)
            vectors_config = getattr(info.config.params, "vectors", None)
            existing_dim = getattr(vectors_config, "size", None)
            if existing_dim is not None and existing_dim != dim:
                logger.warning(
                    "Collection dim %d != configured %d, recreating",
                    existing_dim, dim,
                )
                self.client.delete_collection(self.collection)
                exists = False
                
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            # Payload index for filtered search
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created collection '%s' (dim=%d)", self.collection, dim)

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    async def upsert_chunks(
        self,
        chunk_records: List[ChunkRecord],
        embeddings: List[List[float]],
    ) -> None:
        """Insert chunk vectors with full payload."""
        points = []
        for rec, emb in zip(chunk_records, embeddings):
            points.append(models.PointStruct(
                id=rec.qdrant_point_id or str(uuid.uuid4()),
                vector=emb,
                payload={
                    "chunk_id": rec.id,
                    "document_id": rec.document_id,
                    "content": rec.content,
                    "contextualized_content": rec.contextualized_content or "",
                    "page_number": rec.page_number,
                    "section_title": rec.section_title or "",
                    "chunk_type": rec.chunk_type,
                    "chunk_index": rec.chunk_index,
                    "char_offset": rec.char_offset,
                },
            ))

        import asyncio
        batch_size = 20
        for i in range(0, len(points), batch_size):
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )

    async def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 20,
        document_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Vector similarity search with optional document filtering."""
        import asyncio
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        # use query_points instead of deprecated search
        response = await asyncio.to_thread(
            self.client.query_points, # type: ignore
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        results = response.points

        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                content=payload.get("content", ""),
                contextualized_content=payload.get("contextualized_content", ""),
                page_number=payload.get("page_number", 1),
                section_title=payload.get("section_title", ""),
                chunk_type=payload.get("chunk_type", "text"),
                chunk_index=payload.get("chunk_index", 0),
                char_offset=payload.get("char_offset", 0),
                score=hit.score,
                dense_score=hit.score,
            ))

        return chunks

    async def delete_by_document(self, document_id: str) -> None:
        """Remove all vectors for a given document."""
        import asyncio
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
