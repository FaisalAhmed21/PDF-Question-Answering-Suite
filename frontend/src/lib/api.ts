// API client for the PDF Q&A backend

import type { ChatMessage, ChatSession, DocumentInfo, StreamEvent, UploadResponse } from './types';

const API_BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Documents ─────────────────────────────────────────

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${text}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return request<DocumentInfo[]>('/documents');
}

export async function getDocumentStatus(id: string): Promise<DocumentInfo> {
  return request<DocumentInfo>(`/documents/${id}/status`);
}

export function getDocumentFileUrl(id: string): string {
  return `${API_BASE}/documents/${id}/file`;
}

export async function deleteDocument(id: string): Promise<void> {
  await request(`/documents/${id}`, { method: 'DELETE' });
}

export async function reingestDocument(id: string): Promise<void> {
  await request(`/documents/${id}/reingest`, { method: 'POST' });
}

// ── Chat Sessions ─────────────────────────────────────

export async function createChatSession(
  title: string,
  documentIds: string[]
): Promise<ChatSession> {
  return request<ChatSession>('/chat/sessions', {
    method: 'POST',
    body: JSON.stringify({ title, document_ids: documentIds }),
  });
}

export async function listChatSessions(): Promise<ChatSession[]> {
  return request<ChatSession[]>('/chat/sessions');
}

export async function deleteChatSession(id: string): Promise<void> {
  await request(`/chat/${id}`, { method: 'DELETE' });
}

// ── Chat Messages ─────────────────────────────────────

export async function sendMessageStream(
  sessionId: string,
  content: string,
  onToken: (token: string) => void,
  onDone: (event: StreamEvent) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/${sessionId}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, stream: true }),
  });

  if (!res.ok) {
    throw new Error(`Stream failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No reader available');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // SSE format: "data: {...}"
      let jsonStr = trimmed;
      if (trimmed.startsWith('data:')) {
        jsonStr = trimmed.slice(5).trim();
      }

      if (!jsonStr) continue;

      try {
        const event: StreamEvent = JSON.parse(jsonStr);
        if (event.type === 'token' && event.content) {
          onToken(event.content);
        } else if (event.type === 'done') {
          onDone(event);
        }
      } catch {
        // Skip malformed lines
      }
    }
  }
}

export async function getChatHistory(sessionId: string): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/chat/${sessionId}/history`);
}
