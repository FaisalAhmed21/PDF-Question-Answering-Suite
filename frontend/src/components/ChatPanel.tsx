'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatSession, ChatMessage, Citation, DocumentInfo } from '@/lib/types';
import {
  createChatSession,
  getChatHistory,
  sendMessageStream,
} from '@/lib/api';

interface ChatPanelProps {
  session: ChatSession | null;
  documents: DocumentInfo[];
  onCitationClick: (citation: Citation) => void;
  onSessionCreated: (session: ChatSession) => void;
  /** Called when a grounded answer has citations — open PDF viewer. */
  onGroundedAnswer?: (citations: Citation[]) => void;
  /** Called when the answer is out-of-document — hide PDF highlights. */
  onOutOfContext?: () => void;
}

export function ChatPanel({
  session,
  documents,
  onCitationClick,
  onSessionCreated,
  onGroundedAnswer,
  onOutOfContext,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load history when session changes
  useEffect(() => {
    if (session) {
      getChatHistory(session.id)
        .then(setMessages)
        .catch(console.error);
    } else {
      setMessages([]);
    }
  }, [session?.id]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    let currentSession = session;

    // Auto-create session if none active
    if (!currentSession) {
      const readyDocs = documents.filter((d) => d.status === 'ready');
      if (readyDocs.length === 0) {
        alert('No ready documents. Upload and process a PDF first.');
        return;
      }
      currentSession = await createChatSession(
        input.slice(0, 50),
        readyDocs.map((d) => d.id)
      );
      onSessionCreated(currentSession);
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      citations: [],
      refused: false,
      refusal_reason: '',
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingContent('');

    try {
      let fullContent = '';
      let finalCitations: Citation[] = [];
      let wasRefused = false;
      let refusalReason = '';

      await sendMessageStream(
        currentSession.id,
        userMessage.content,
        (token) => {
          fullContent += token;
          setStreamingContent(fullContent);
        },
        (event) => {
          wasRefused = event.refused || false;
          refusalReason = event.refusal_reason || '';
          finalCitations = wasRefused ? [] : (event.citations || []);
        }
      );

      const contentLower = fullContent.toLowerCase();
      const looksOutOfScope =
        contentLower.includes('outside the scope') ||
        contentLower.includes('not covered in the') ||
        contentLower.includes('only answer questions that are covered');
      if (looksOutOfScope) {
        wasRefused = true;
        finalCitations = [];
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: fullContent,
        citations: wasRefused ? [] : finalCitations,
        refused: wasRefused,
        refusal_reason: refusalReason,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setStreamingContent('');

      if (wasRefused) {
        onOutOfContext?.();
      } else if (finalCitations.length > 0) {
        onGroundedAnswer?.(finalCitations);
      } else {
        onOutOfContext?.();
      }
    } catch (err) {
      console.error('Message send failed:', err);
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Sorry, an error occurred while processing your question. Please try again.',
        citations: [],
        refused: true,
        refusal_reason: String(err),
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      setStreamingContent('');
      onOutOfContext?.();
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, session, documents, onSessionCreated, onGroundedAnswer, onOutOfContext]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Empty state
  if (!session && messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="text-center max-w-md animate-fade-in">
          <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary-500/20 to-primary-700/20 border border-primary-500/20 flex items-center justify-center">
            <svg className="w-8 h-8 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-surface-100 mb-2">
            Ask Your Documents
          </h2>
          <p className="text-surface-300 mb-6 leading-relaxed">
            Upload a PDF, wait for it to be processed, then ask questions. 
            Your answers will be grounded in the document with page citations.
          </p>

          {/* Quick start input */}
          <div className="relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your PDFs..."
              rows={2}
              className="glass-input w-full px-4 py-3 pr-12 resize-none text-sm"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="absolute right-2 bottom-2 p-2 rounded-lg bg-primary-600 text-white 
                       hover:bg-primary-500 disabled:opacity-30 disabled:cursor-not-allowed
                       transition-all duration-200"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Header */}
      <div className="px-6 py-3 border-b border-surface-700/40 bg-surface-900/40 backdrop-blur-sm">
        <h2 className="text-sm font-semibold text-surface-100">
          {session?.title || 'Chat'}
        </h2>
        <p className="text-[10px] text-surface-300 mt-0.5">
          {session?.document_ids.length || 0} document(s) active
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onCitationClick={onCitationClick}
          />
        ))}

        {/* Streaming indicator */}
        {isLoading && streamingContent && (
          <div className="assistant-bubble">
            <div className="markdown-content text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {streamingContent}
              </ReactMarkdown>
              <span className="inline-block w-2 h-4 bg-primary-400 animate-pulse ml-0.5" />
            </div>
          </div>
        )}

        {/* Loading dots */}
        {isLoading && !streamingContent && (
          <div className="assistant-bubble">
            <div className="flex items-center gap-1.5 py-1">
              <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-surface-700/40 bg-surface-900/40 backdrop-blur-sm">
        <div className="relative max-w-3xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="glass-input w-full px-4 py-3 pr-14 resize-none text-sm"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-primary-600 text-white
                     hover:bg-primary-500 disabled:opacity-30 disabled:cursor-not-allowed
                     transition-all duration-200 shadow-lg shadow-primary-600/20"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Message Bubble Component ──────────────────────────

function MessageBubble({
  message,
  onCitationClick,
}: {
  message: ChatMessage;
  onCitationClick: (c: Citation) => void;
}) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="user-bubble">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  const isRefused = message.refused;

  return (
    <div className="flex justify-start">
      <div className={isRefused ? 'refused-bubble' : 'assistant-bubble'}>
        <div className="markdown-content text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && !isRefused && (
          <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-surface-700/30">
            {message.citations.map((cit: Citation, i: number) => (
              <button
                key={i}
                onClick={() => onCitationClick(cit)}
                className="citation-pill"
                title={cit.snippet}
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                Page {cit.page_number}
                {cit.section_title && (
                  <span className="opacity-60">· {cit.section_title}</span>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
