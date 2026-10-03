"""Shared Pydantic models and types used across modules."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Element types from PDF parsing ─────────────────────
class ElementType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    TRANSCRIPT = "transcript"


class RawElement(BaseModel):
    """A parsed element from a PDF page."""
    type: ElementType
    content: str
    page_number: int = 1
    source_id: str = ""
    section_title: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Chunk ──────────────────────────────────────────────
class ChunkType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    TRANSCRIPT = "transcript"


class Chunk(BaseModel):
    """A chunked piece of document content."""
    document_id: str
    content: str
    contextualized_content: str = ""
    page_number: int = 1
    section_title: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    char_offset: int = 0
    chunk_index: int = 0


# ── Retrieved chunks ──────────────────────────────────
class RetrievedChunk(BaseModel):
    """A chunk retrieved from search with relevance score."""
    chunk_id: str
    document_id: str
    content: str
    contextualized_content: str = ""
    page_number: int = 1
    section_title: str = ""
    chunk_type: str = "text"
    chunk_index: int = 0
    char_offset: int = 0
    score: float = 0.0
    dense_score: float | None = None  # cosine similarity before RRF fusion
    rerank_score: float | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Citation ──────────────────────────────────────────
class Citation(BaseModel):
    """A page-level citation for an answer."""
    citation_index: int = 0
    document_id: str
    document_name: str = ""
    page_number: int
    section_title: str = ""
    snippet: str = ""


# ── Generation result ─────────────────────────────────
class GenerationResult(BaseModel):
    """Result from the answer generation pipeline."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    grounded: bool = True
    retrieval_scores: Dict[str, float] = Field(default_factory=dict)


# ── Document status ───────────────────────────────────
class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


# ── API request/response models ──────────────────────
class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None


class ChatSessionCreate(BaseModel):
    title: str = "New Chat"
    document_ids: List[str] = Field(default_factory=list)


class ChatSessionInfo(BaseModel):
    id: str
    title: str
    document_ids: List[str] = Field(default_factory=list)
    created_at: datetime | None = None


class ChatMessageRequest(BaseModel):
    content: str
    stream: bool = False


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: List[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    created_at: datetime | None = None
