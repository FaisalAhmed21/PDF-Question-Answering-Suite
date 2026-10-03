"""Chat API routes with SSE streaming support."""

from __future__ import annotations

import json
import uuid
from typing import cast

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.db.models import ChatSession, Document, Message
from app.db.session import async_session
from app.generation.agentic import agentic_rag
from app.types import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionInfo,
    GenerationResult,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionInfo)
async def create_session(data: ChatSessionCreate):
    """Create a new chat session scoped to documents."""
    async with async_session() as session:
        # Validate document IDs exist
        for doc_id in data.document_ids:
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            if not result.scalar_one_or_none():
                raise HTTPException(404, f"Document {doc_id} not found")

        chat = ChatSession(
            id=str(uuid.uuid4()),
            title=data.title,
            document_ids=data.document_ids,
        )
        session.add(chat)
        await session.commit()

        return ChatSessionInfo(
            id=chat.id,
            title=chat.title,
            document_ids=chat.document_ids or [],
            created_at=chat.created_at,
        )


@router.get("/sessions", response_model=list[ChatSessionInfo])
async def list_sessions():
    """List all chat sessions."""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [
            ChatSessionInfo(
                id=s.id,
                title=s.title,
                document_ids=s.document_ids or [],
                created_at=s.created_at,
            )
            for s in sessions
        ]


@router.post("/{session_id}/message")
async def send_message(session_id: str, data: ChatMessageRequest):
    """Send a question, get an answer. Supports SSE streaming."""
    async with async_session() as session:
        # Load chat session
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(404, "Chat session not found")

        document_ids = chat.document_ids or []
        if not document_ids:
            raise HTTPException(400, "No documents associated with this chat session")

        # Get doc names for citations
        doc_names = {}
        for doc_id in document_ids:
            doc_result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                doc_names[doc_id] = doc.filename

        # Save user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=data.content,
        )
        session.add(user_msg)
        await session.commit()

        # Load history
        history_result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        history_msgs = history_result.scalars().all()
        history = [
            {"role": m.role, "content": m.content}
            for m in history_msgs[:-1]  # exclude current message
        ]

    # Run RAG pipeline
    if data.stream:
        result = await agentic_rag(
            query=data.content,
            document_ids=document_ids,
            history=history,
            doc_names=doc_names,
            stream=True,
        )

        async def event_generator():
            full_answer = []
            citations = []
            refused = False
            refusal_reason = ""

            if hasattr(result, "__aiter__"):
                async for line in result: # type: ignore
                    event_data = json.loads(line)
                    if event_data["type"] == "token":
                        full_answer.append(event_data["content"])
                        yield {"data": line}
                    elif event_data["type"] == "done":
                        citations = event_data.get("citations", [])
                        refused = event_data.get("refused", False)
                        refusal_reason = event_data.get("refusal_reason", "")
                        yield {"data": line}
            else:
                # agentic_rag returned a GenerationResult (e.g. due to relevance gate refusal)
                answer = result.answer
                citations = [c.model_dump() for c in result.citations]
                refused = result.refused
                refusal_reason = result.refusal_reason
                full_answer.append(answer)
                
                yield {"data": json.dumps({"type": "token", "content": answer}) + "\n"}
                yield {"data": json.dumps({
                    "type": "done",
                    "citations": citations,
                    "refused": refused,
                    "refusal_reason": refusal_reason,
                }) + "\n"}

            # Save assistant message
            answer_text = "".join(full_answer)
            async with async_session() as session:
                asst_msg = Message(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                    citations=citations,
                    refused=refused,
                    refusal_reason=refusal_reason,
                )
                session.add(asst_msg)
                await session.commit()

        return EventSourceResponse(event_generator())

    else:
        raw_result = await agentic_rag(
            query=data.content,
            document_ids=document_ids,
            history=history,
            doc_names=doc_names,
            stream=False,
        )
        
        if hasattr(raw_result, "__aiter__"):
            raise HTTPException(500, "Unexpected streaming response")
            
        gen_result = cast(GenerationResult, raw_result)

        # Save assistant message
        async with async_session() as session:
            asst_msg = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=gen_result.answer,
                citations=[c.model_dump() for c in gen_result.citations],
                refused=gen_result.refused,
                refusal_reason=gen_result.refusal_reason,
            )
            session.add(asst_msg)
            await session.commit()

            return ChatMessageResponse(
                id=asst_msg.id,
                role="assistant",
                content=gen_result.answer,
                citations=gen_result.citations,
                refused=gen_result.refused,
                refusal_reason=gen_result.refusal_reason,
                created_at=asst_msg.created_at,
            )


@router.get("/{session_id}/history", response_model=list[ChatMessageResponse])
async def get_history(session_id: str):
    """Get full conversation history for a session."""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(404, "Chat session not found")

        msg_result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        messages = msg_result.scalars().all()

        return [
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=m.citations or [],
                refused=m.refused,
                refusal_reason=m.refusal_reason or "",
                created_at=m.created_at,
            )
            for m in messages
        ]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its messages."""
    async with async_session() as session:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(404, "Chat session not found")

        await session.delete(chat)
        await session.commit()

    return {"status": "deleted"}
