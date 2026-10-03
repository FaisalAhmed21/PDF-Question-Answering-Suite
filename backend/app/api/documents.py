"""Document management API routes."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from sqlalchemy import select

from app.config import settings
from app.db.models import Document
from app.db.session import async_session
from app.ingestion.pipeline import run_ingestion
from app.retrieval.vector_store import get_vector_store
from app.types import DocumentInfo, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF file and kick off async ingestion."""
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Check size
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")

    # Save to disk
    doc_id = str(uuid.uuid4())
    upload_dir = settings.upload_path / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    file_path.write_bytes(content)

    # Create DB record
    async with async_session() as session:
        doc = Document(
            id=doc_id,
            filename=file.filename,
            storage_path=str(file_path),
            content_type=file.content_type or "application/pdf",
            status="pending",
        )
        session.add(doc)
        await session.commit()

    # Start background ingestion
    background_tasks.add_task(run_ingestion, doc_id)

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status="pending",
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    async with async_session() as session:
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()
        return [
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                status=d.status,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                error_message=d.error_message,
                created_at=d.created_at,
            )
            for d in docs
        ]


@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """Poll ingestion status."""
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(404, "Document not found")
        return {
            "id": doc.id,
            "status": doc.status,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "error_message": doc.error_message,
        }


@router.get("/{document_id}/file")
async def get_document_file(document_id: str):
    """Serve the original uploaded file."""
    from fastapi.responses import FileResponse

    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(404, "Document not found")

        if not os.path.exists(doc.storage_path):
            raise HTTPException(404, "File not found on disk")

        return FileResponse(
            doc.storage_path,
            media_type="application/pdf",
        )


@router.post("/{document_id}/reingest")
async def reingest_document(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    """Re-process a document without re-uploading."""
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(404, "Document not found")

        doc.status = "pending"
        doc.error_message = None
        await session.commit()

    background_tasks.add_task(run_ingestion, document_id)
    return {"status": "pending", "message": "Re-ingestion started"}


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete document, vectors, chunks, and file."""
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(404, "Document not found")

        # Delete vectors
        try:
            vs = get_vector_store()
            await vs.delete_by_document(document_id)
        except Exception:
            pass

        # Delete file from disk
        try:
            upload_dir = settings.upload_path / document_id
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
        except Exception:
            pass

        # Delete DB record (cascades to chunks)
        await session.delete(doc)
        await session.commit()

    return {"status": "deleted"}
