'use client';

import React, { useState, useRef, useCallback } from 'react';
import { uploadDocument } from '@/lib/api';

interface UploadModalProps {
  onClose: () => void;
  onUploaded: () => void;
}

export function UploadModal({ onClose, onUploaded }: UploadModalProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size exceeds 50MB limit');
      return;
    }

    setIsUploading(true);
    setError('');
    setProgress(30);

    try {
      await uploadDocument(file);
      setProgress(100);
      setTimeout(onUploaded, 500);
    } catch (err) {
      setError(String(err));
      setIsUploading(false);
      setProgress(0);
    }
  }, [onUploaded]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="glass-panel w-full max-w-md mx-4 p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-lg font-bold text-surface-50">Upload PDF</h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-surface-300 hover:text-surface-100
                     hover:bg-surface-700/50 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Drop zone */}
        <div
          className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          {isUploading ? (
            <>
              <div className="w-12 h-12 rounded-xl bg-primary-500/15 flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-400 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
              <p className="text-sm text-surface-200 font-medium">Uploading...</p>
              <div className="w-full max-w-xs h-1.5 bg-surface-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </>
          ) : (
            <>
              <div className="w-12 h-12 rounded-xl bg-primary-500/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-sm text-surface-200 font-medium">
                  Drop a PDF here or click to browse
                </p>
                <p className="text-xs text-surface-300/60 mt-1">
                  PDF files up to 50MB
                </p>
              </div>
            </>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleInputChange}
          className="hidden"
        />

        {error && (
          <div className="mt-3 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}

        <p className="mt-4 text-[10px] text-surface-300/50 text-center">
          The document will be automatically parsed, chunked, and embedded for Q&A.
        </p>
      </div>
    </div>
  );
}
