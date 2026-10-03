// Shared TypeScript types for the PDF Q&A application

export interface DocumentInfo {
  id: string;
  filename: string;
  status: 'pending' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'failed';
  page_count: number | null;
  chunk_count: number | null;
  error_message: string | null;
  created_at: string | null;
}

export interface Citation {
  citation_index: number;
  document_id: string;
  document_name: string;
  page_number: number;
  section_title: string;
  snippet: string;
}

export interface ChatSession {
  id: string;
  title: string;
  document_ids: string[];
  created_at: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string;
  created_at: string | null;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
}

export interface StreamEvent {
  type: 'token' | 'done';
  content?: string;
  citations?: Citation[];
  refused?: boolean;
  refusal_reason?: string;
  grounded?: boolean;
}
