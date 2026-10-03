'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PdfViewerProps {
  url: string;
  page: number;
  searchQuery?: string | null;
  onClose: () => void;
  onPageChange: (page: number) => void;
}

const STOP = new Set([
  'the', 'and', 'for', 'are', 'with', 'from', 'that', 'this',
  'have', 'has', 'been', 'into', 'only', 'also', 'used',
]);

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function contentWords(text: string): string[] {
  return normalize(text)
    .split(' ')
    .filter((w) => w.length > 2 && !STOP.has(w));
}

/** Build distinctive phrases (bigrams/trigrams) from the citation snippet. */
function buildPhrases(snippet: string): string[] {
  const words = contentWords(snippet);
  const phrases: string[] = [];

  for (let n = 3; n >= 2; n--) {
    for (let i = 0; i <= words.length - n; i++) {
      phrases.push(words.slice(i, i + n).join(' '));
    }
  }

  // Also keep unusually specific single tokens (e.g. "taka", "bangladeshi")
  for (const w of words) {
    if (w.length >= 5) phrases.push(w);
  }

  // Prefer longer phrases first
  return Array.from(new Set(phrases)).sort((a, b) => b.length - a.length);
}

export function PdfViewer({ url, page, searchQuery, onClose, onPageChange }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }

  const snippetNorm = useMemo(
    () => (searchQuery ? normalize(searchQuery) : ''),
    [searchQuery]
  );

  const phrases = useMemo(
    () => (searchQuery ? buildPhrases(searchQuery) : []),
    [searchQuery]
  );

  const snippetWordSet = useMemo(
    () => new Set(contentWords(searchQuery || '')),
    [searchQuery]
  );

  const textRenderer = useCallback(
    (textItem: { str: string }) => {
      if (!snippetNorm || !textItem.str.trim()) return textItem.str;

      const itemNorm = normalize(textItem.str);
      if (!itemNorm || itemNorm.length < 3) return textItem.str;

      const itemWords = contentWords(textItem.str);

      // 1) Contiguous span: item text is a substring of the citation snippet
      //    (or vice-versa for longer PDF spans). Require enough content.
      if (itemNorm.length >= 12 && snippetNorm.includes(itemNorm)) {
        return `<mark class="pdf-cite-mark">${textItem.str}</mark>`;
      }
      if (
        snippetNorm.length >= 12 &&
        itemNorm.includes(snippetNorm) &&
        itemWords.length >= 3
      ) {
        return `<mark class="pdf-cite-mark">${textItem.str}</mark>`;
      }

      // 2) Multi-word phrase match from the citation (most precise for word tokens)
      for (const phrase of phrases) {
        const phraseWords = phrase.split(' ');
        if (phraseWords.length >= 2) {
          if (itemNorm.includes(phrase) || phrase.includes(itemNorm)) {
            // Avoid highlighting tiny fragments that only share one short word
            if (itemWords.length >= 2 || itemNorm.length >= 10) {
              return `<mark class="pdf-cite-mark">${textItem.str}</mark>`;
            }
          }
          // Word-level tokens: highlight only if this token belongs to a
          // multi-word phrase AND is a distinctive (>=5 char) content word
          // that appears in the snippet — but alone is too loose.
          // Skip single-token highlighting here; handled below more strictly.
        }
      }

      // 3) Single-token items: highlight only distinctive snippet words that
      //    are unlikely to appear broadly (length >= 6) AND appear in a
      //    multi-word phrase from the citation.
      if (itemWords.length === 1) {
        const w = itemWords[0];
        if (w.length >= 6 && snippetWordSet.has(w)) {
          const inPhrase = phrases.some(
            (p) => p.split(' ').length >= 2 && p.split(' ').includes(w)
          );
          if (inPhrase) {
            return `<mark class="pdf-cite-mark">${textItem.str}</mark>`;
          }
        }
        return textItem.str;
      }

      // 4) Short multi-word spans: require majority of content words in snippet
      //    AND at least one bigram/trigram overlap
      if (itemWords.length >= 2) {
        const hits = itemWords.filter((w) => snippetWordSet.has(w)).length;
        const ratio = hits / itemWords.length;
        const hasPhrase = phrases.some(
          (p) => p.split(' ').length >= 2 && itemNorm.includes(p)
        );
        if (ratio >= 0.75 && (hasPhrase || hits >= 3)) {
          return `<mark class="pdf-cite-mark">${textItem.str}</mark>`;
        }
      }

      return textItem.str;
    },
    [snippetNorm, phrases, snippetWordSet]
  );

  const canPrev = page > 1;
  const canNext = numPages !== null && page < numPages;

  return (
    <div className="w-[500px] h-full flex flex-col border-l border-surface-700/50 bg-surface-900/80 animate-fade-in relative z-50">
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-700/40 shrink-0">
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={!canPrev}
            onClick={() => onPageChange(page - 1)}
            className="p-1.5 rounded-lg text-surface-300 hover:text-surface-100 hover:bg-surface-700/50
                     disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Previous page"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <span className="text-sm font-medium text-surface-200 min-w-[100px] text-center">
            Page {page}{numPages ? ` / ${numPages}` : ''}
          </span>
          <button
            type="button"
            disabled={!canNext}
            onClick={() => onPageChange(page + 1)}
            className="p-1.5 rounded-lg text-surface-300 hover:text-surface-100 hover:bg-surface-700/50
                     disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Next page"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-surface-300 hover:text-surface-100 hover:bg-surface-700/50 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 bg-surface-950 overflow-auto flex justify-center py-4 relative">
        <Document
          file={url}
          onLoadSuccess={onDocumentLoadSuccess}
          className="max-w-full"
          loading={
            <div className="flex items-center justify-center h-full text-surface-400 absolute inset-0">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-surface-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Loading Document...
            </div>
          }
          error={
            <div className="flex flex-col items-center justify-center h-full text-red-400 absolute inset-0 gap-2">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Failed to load PDF.
            </div>
          }
        >
          <Page
            key={`${page}-${snippetNorm}`}
            pageNumber={page}
            renderTextLayer={true}
            renderAnnotationLayer={true}
            customTextRenderer={textRenderer}
            className="shadow-xl"
            width={450}
          />
        </Document>
      </div>
    </div>
  );
}
