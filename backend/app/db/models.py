"""SQLAlchemy 2.0 async ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), default="default")
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), default="application/pdf")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    chunks: Mapped[list[ChunkRecord]] = relationship(
        "ChunkRecord", back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRecord(Base):
    __tablename__ = "chunk_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunk_type: Mapped[str] = mapped_column(String(32), default="text")
    char_offset: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), default="default")
    title: Mapped[str] = mapped_column(String(256), default="New Chat")
    document_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")


class EvalLog(Base):
    __tablename__ = "eval_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    gate: Mapped[str] = mapped_column(String(64), nullable=False)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    answer_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
