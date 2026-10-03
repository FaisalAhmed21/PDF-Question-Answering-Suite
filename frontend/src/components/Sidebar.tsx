'use client';

import React, { useEffect, useState } from 'react';
import type { DocumentInfo, ChatSession } from '@/lib/types';
import { deleteDocument, deleteChatSession, createChatSession, getDocumentStatus } from '@/lib/api';

interface SidebarProps {
  documents: DocumentInfo[];
  sessions: ChatSession[];
  activeSession: ChatSession | null;
  onSelectSession: (session: ChatSession) => void;
  onNewUpload: () => void;
  onRefreshDocuments: () => void;
  onRefreshSessions: () => void;
  onDocumentClick: (doc: DocumentInfo) => void;
}

const STATUS_LABELS: Record<string, string> = {
  pending: '⏳ Pending',
  parsing: '📄 Parsing',
  chunking: '✂️ Chunking',
  embedding: '🧠 Embedding',
  ready: '✅ Ready',
  failed: '❌ Failed',
};

export function Sidebar({
  documents,
  sessions,
  activeSession,
  onSelectSession,
  onNewUpload,
  onRefreshDocuments,
  onRefreshSessions,
  onDocumentClick,
}: SidebarProps) {
  const [tab, setTab] = useState<'docs' | 'chats'>('docs');

  // Poll for processing documents
  useEffect(() => {
    const processing = documents.filter(
      (d) => !['ready', 'failed'].includes(d.status)
    );

    if (processing.length === 0) return;

    const interval = setInterval(() => {
      processing.forEach(async (doc) => {
        try {
          const status = await getDocumentStatus(doc.id);
          if (status.status === 'ready' || status.status === 'failed') {
            onRefreshDocuments();
          }
        } catch {
          // ignore
        }
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [documents, onRefreshDocuments]);

  const handleDeleteDoc = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this document?')) return;
    try {
      await deleteDocument(id);
      onRefreshDocuments();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteChatSession(id);
      onRefreshSessions();
    } catch (err) {
      console.error(err);
    }
  };

  const handleNewChat = async () => {
    const readyDocs = documents.filter((d) => d.status === 'ready');
    if (readyDocs.length === 0) {
      alert('No ready documents. Upload and wait for processing to complete.');
      return;
    }
    try {
      const session = await createChatSession(
        'New Chat',
        readyDocs.map((d) => d.id)
      );
      onSelectSession(session);
      onRefreshSessions();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <aside className="w-72 h-screen flex flex-col glass-panel border-r border-surface-700/50 rounded-none">
      {/* Header */}
      <div className="p-4 border-b border-surface-700/40">
        <div className="flex items-center gap-2.5 mb-4">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-bold text-surface-50">PDF Q&A</h1>
            <p className="text-[10px] text-surface-300">RAG-Powered Assistant</p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex gap-1 bg-surface-800/60 rounded-lg p-0.5">
          <button
            onClick={() => setTab('docs')}
            className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-all ${
              tab === 'docs'
                ? 'bg-primary-600/80 text-white shadow-sm'
                : 'text-surface-300 hover:text-surface-100'
            }`}
          >
            Documents
          </button>
          <button
            onClick={() => setTab('chats')}
            className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-all ${
              tab === 'chats'
                ? 'bg-primary-600/80 text-white shadow-sm'
                : 'text-surface-300 hover:text-surface-100'
            }`}
          >
            Chats
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {tab === 'docs' ? (
          documents.length === 0 ? (
            <div className="text-center py-8 text-surface-300 text-sm">
              <p className="mb-1">No documents yet</p>
              <p className="text-xs text-surface-300/60">Upload a PDF to get started</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onDocumentClick(doc)}
                className="group p-3 rounded-xl bg-surface-800/30 hover:bg-surface-800/60 
                         border border-transparent hover:border-surface-700/40
                         cursor-pointer transition-all duration-200"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-surface-100 truncate">
                      {doc.filename}
                    </p>
                    <div className="flex items-center gap-2 mt-1.5">
                      <span className={`status-${doc.status}`}>
                        {STATUS_LABELS[doc.status] || doc.status}
                      </span>
                      {doc.page_count && (
                        <span className="text-[10px] text-surface-300">
                          {doc.page_count} pages
                        </span>
                      )}
                    </div>
                    {doc.error_message && (
                      <p className="text-[10px] text-red-400 mt-1 truncate">
                        {doc.error_message}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDeleteDoc(doc.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded-lg
                             text-surface-300 hover:text-red-400 hover:bg-red-500/10
                             transition-all duration-200"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))
          )
        ) : (
          sessions.length === 0 ? (
            <div className="text-center py-8 text-surface-300 text-sm">
              <p className="mb-1">No chats yet</p>
              <p className="text-xs text-surface-300/60">Start a new conversation</p>
            </div>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => onSelectSession(s)}
                className={`w-full text-left p-3 rounded-xl transition-all duration-200 group cursor-pointer
                          ${activeSession?.id === s.id
                            ? 'bg-primary-600/20 border border-primary-500/30'
                            : 'hover:bg-surface-800/50 border border-transparent'
                          }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 min-w-0">
                    <svg className="w-4 h-4 text-surface-300 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                    <span className="text-sm text-surface-200 truncate">{s.title}</span>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded-lg
                             text-surface-300 hover:text-red-400 transition-opacity"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))
          )
        )}
      </div>

      {/* Bottom buttons */}
      <div className="p-3 border-t border-surface-700/40 space-y-2">
        <button onClick={onNewUpload} className="btn-primary w-full flex items-center justify-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          Upload PDF
        </button>
        <button onClick={handleNewChat} className="btn-ghost w-full flex items-center justify-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          New Chat
        </button>
      </div>
    </aside>
  );
}
