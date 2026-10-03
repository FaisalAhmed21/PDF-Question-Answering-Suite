"""Full async ingestion pipeline: parse → chunk → embed → store."""

from __future__ import annotations

import logging
import uuid
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ChunkRecord, Document
from app.db.session import async_session
from app.ingestion.chunker import chunk_elements
from app.ingestion.contextual import apply_contextual_prefixes
from app.ingestion.embedder import embed_texts
from app.ingestion.nlp_enhancements import generate_doc2query, summarize_table
from app.ingestion.pdf import count_pdf_pages, parse_pdf
from app.retrieval.vector_store import get_vector_store
from app.types import Chunk, ChunkType

logger = logging.getLogger(__name__)


async def _update_status(session: AsyncSession, doc_id: str, status: str, **kwargs):
    """Update document status in DB."""
    result = await session.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = status
        for k, v in kwargs.items():
            setattr(doc, k, v)
        await session.commit()


async def run_ingestion(document_id: str) -> None:
    """Execute the full ingestion pipeline for a document."""
    async with async_session() as session:
        try:
            # 1. Load document
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if not doc:
                logger.error("Document %s not found", document_id)
                return

            pdf_path = doc.storage_path

            # 2. Parse
            await _update_status(session, document_id, "parsing")
            elements = await parse_pdf(pdf_path)
            if not elements:
                await _update_status(
                    session, document_id, "failed",
                    error_message="No content extracted from PDF"
                )
                return

            page_count = count_pdf_pages(pdf_path)
            await _update_status(session, document_id, "parsing", page_count=page_count)

            # 3. Chunk
            await _update_status(session, document_id, "chunking")
            chunks = await chunk_elements(elements, document_id)
            if not chunks:
                await _update_status(
                    session, document_id, "failed",
                    error_message="Chunking produced zero chunks"
                )
                return

            # 4. Contextual prefixes + Advanced NLP (with timeout protection)
            try:
                import asyncio
                chunks = await asyncio.wait_for(
                    apply_contextual_prefixes(chunks, doc.filename),
                    timeout=60.0,  # 60s max for all contextual prefixes
                )
            except asyncio.TimeoutError:
                logger.warning("Contextual prefix generation timed out for %s, using raw chunks", document_id)
            except Exception as exc:
                logger.warning("Contextual prefix generation failed for %s: %s", document_id, exc)

            for c in chunks:
                try:
                    import asyncio
                    if c.chunk_type == ChunkType.TABLE:
                        summary = await asyncio.wait_for(
                            summarize_table(c.content),
                            timeout=15.0,
                        )
                        c.contextualized_content = summary
                    elif c.chunk_type == ChunkType.TEXT:
                        questions = await asyncio.wait_for(
                            generate_doc2query(c.content),
                            timeout=15.0,
                        )
                        if questions:
                            base = c.contextualized_content or c.content
                            c.contextualized_content = f"{base}\n\nQuestions this answers:\n{questions}"
                except asyncio.TimeoutError:
                    logger.warning("NLP enhancement timed out for chunk %s", c.chunk_index)
                except Exception as exc:
                    logger.warning("NLP enhancement failed for chunk %s: %s", c.chunk_index, exc)

            # 5. Embed
            await _update_status(session, document_id, "embedding")
            texts_to_embed = [
                c.contextualized_content or c.content for c in chunks
            ]
            embeddings = await embed_texts(texts_to_embed)

            # 6. Clean old data
            vs = get_vector_store()
            await vs.delete_by_document(document_id)
            # Delete old chunk records
            old_chunks = await session.execute(
                select(ChunkRecord).where(ChunkRecord.document_id == document_id)
            )
            for old in old_chunks.scalars().all():
                await session.delete(old)
            await session.commit()

            # 7. Store chunk records + upsert vectors
            chunk_records: List[ChunkRecord] = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                rec = ChunkRecord(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk.content,
                    contextualized_content=chunk.contextualized_content,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    chunk_type=chunk.chunk_type.value,
                    char_offset=chunk.char_offset,
                    chunk_index=chunk.chunk_index,
                    qdrant_point_id=point_id,
                )
                chunk_records.append(rec)
                session.add(rec)

            await session.commit()

            # Upsert to Qdrant
            await vs.upsert_chunks(chunk_records, embeddings)

            # 8. Done
            await _update_status(
                session, document_id, "ready",
                chunk_count=len(chunk_records),
                page_count=page_count,
            )
            logger.info(
                "Ingestion complete for %s: %d chunks, %d pages",
                document_id, len(chunk_records), page_count,
            )

        except Exception as exc:
            logger.exception("Ingestion failed for %s", document_id)
            try:
                vs = get_vector_store()
                await vs.delete_by_document(document_id)
            except Exception:
                pass
            await _update_status(
                session, document_id, "failed",
                error_message=str(exc)[:1000],
            )
