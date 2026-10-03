'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChatPanel } from '@/components/ChatPanel';
import { PdfViewer } from '@/components/PdfViewer';
import { UploadModal } from '@/components/UploadModal';
import type { DocumentInfo, ChatSession, Citation } from '@/lib/types';
import { listDocuments, listChatSessions, getDocumentFileUrl } from '@/lib/api';

export default function Home() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [showPdf, setShowPdf] = useState(false);

  const [pdfSearchQuery, setPdfSearchQuery] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (e) {
      console.error('Failed to fetch documents:', e);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const sess = await listChatSessions();
      setSessions(sess);
    } catch (e) {
      console.error('Failed to fetch sessions:', e);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    refreshSessions();
  }, [refreshDocuments, refreshSessions]);

  const handleCitationClick = useCallback((citation: Citation) => {
    const docUrl = getDocumentFileUrl(citation.document_id);
    setPdfUrl(docUrl);
    setPdfPage(citation.page_number);
    setPdfSearchQuery(citation.snippet);
    setShowPdf(true);
  }, []);

  const handleGroundedAnswer = useCallback((citations: Citation[]) => {
    if (!citations.length) return;
    handleCitationClick(citations[0]);
  }, [handleCitationClick]);

  const handleOutOfContext = useCallback(() => {
    setPdfSearchQuery(null);
    setShowPdf(false);
  }, []);

  const handleDocumentClick = (doc: DocumentInfo) => {
    if (doc.status === 'ready') {
      setPdfUrl(getDocumentFileUrl(doc.id));
      setPdfPage(1);
      setPdfSearchQuery(null);
      setShowPdf(true);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        documents={documents}
        sessions={sessions}
        activeSession={activeSession}
        onSelectSession={setActiveSession}
        onNewUpload={() => setShowUpload(true)}
        onRefreshDocuments={refreshDocuments}
        onRefreshSessions={refreshSessions}
        onDocumentClick={handleDocumentClick}
      />

      {/* Main content */}
      <div className="flex-1 flex min-w-0">
        {/* Chat panel */}
        <ChatPanel
          session={activeSession}
          documents={documents}
          onCitationClick={handleCitationClick}
          onSessionCreated={(session) => {
            setActiveSession(session);
            refreshSessions();
          }}
          onGroundedAnswer={handleGroundedAnswer}
          onOutOfContext={handleOutOfContext}
        />

        {/* PDF Viewer — opens on grounded answers / citation clicks */}
        {showPdf && pdfUrl && (
          <PdfViewer
            url={pdfUrl}
            page={pdfPage}
            searchQuery={pdfSearchQuery}
            onClose={() => {
              setShowPdf(false);
              setPdfSearchQuery(null);
            }}
            onPageChange={setPdfPage}
          />
        )}
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={() => {
            refreshDocuments();
            setShowUpload(false);
          }}
        />
      )}
    </div>
  );
}
