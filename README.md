# RAG-Powered PDF Q&A Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents, ask natural-language questions, and get answers that are **strictly grounded in the PDF content** — with page citations and in-viewer highlighting of the exact supporting lines.

---

## What this project does

Typical “chat with PDF” demos often invent facts when the document does not contain the answer. This project is built around the opposite goal:

1. **Ingest** a PDF into searchable chunks (parse → chunk → enrich → embed → store).
2. **Retrieve** the most relevant chunks for a user question (hybrid dense + BM25 search, then rerank).
3. **Decide** whether the question is actually covered by the document (relevance gate + lexical overlap).
4. **Generate** an answer using only retrieved context (LLM must not use outside knowledge).
5. **Verify** the answer is grounded; otherwise refuse cleanly.
6. **Cite & highlight** the exact PDF sentences that supported the answer; open that page in a side PDF viewer.

If a question is **outside** the PDF (e.g. “national bird of Bangladesh” when the PDF never mentions a bird), the system replies that the question is out of scope and **does not** show fake citations or highlights.

If a question **is** in the PDF (e.g. “currency of Bangladesh” when the PDF states the Taka), you get a short grounded answer, citation pills, and the matching line highlighted in the PDF on the right.

---

## Key features

| Feature | What it means in practice |
|--------|----------------------------|
| PDF upload & processing | Sidebar upload; status progresses through parsing → chunking → embedding → **Ready** |
| Hybrid retrieval | Dense vector search (embeddings) + sparse BM25 keyword search, fused with RRF |
| Cross-encoder reranking | Reorders candidates so the most answer-relevant chunks rise to the top |
| Relevance gate | Blocks clearly out-of-document questions *before* wasting LLM calls |
| Grounded generation | System prompt + context-only answering; refuses when unsupported |
| Groundedness check | Optional post-check: if the draft answer isn’t supported by context, refuse / retry |
| Agentic second hop | On borderline or multi-hop questions, can run an extra retrieval pass |
| Streaming chat UI | Answers stream into the chat panel over SSE |
| Page citations | Answers include `[1]`, `[2]` style markers mapped to PDF pages |
| PDF viewer + highlight | Opens the cited page and highlights the supporting snippet (not the whole page) |
| Multi-provider LLM | Primary + fallback (e.g. Groq → Gemini) with CoT / reasoning text stripped from answers |

---

## High-level architecture

```
┌─────────────────────┐         ┌──────────────────────────────────────────────┐
│  Next.js Frontend   │  HTTP   │  FastAPI Backend                             │
│  - Chat panel       │◄───────►│  - /documents/*  (upload, status, file)      │
│  - Sidebar          │  /api/* │  - /chat/*       (sessions, SSE answers)     │
│  - PDF viewer       │  proxy  │  - SQLite metadata + chat history            │
└─────────────────────┘         │  - Qdrant vector store (local or cloud)      │
                                │  - LLM providers (Groq / Gemini)             │
                                └──────────────────────────────────────────────┘
```

**Data stores**

- **SQLite (`ragchatbot.db`)** — documents, chunk records, chat sessions, messages, eval logs.
- **Qdrant** — embedding vectors for semantic search (local folder `qdrant_data/` or Qdrant Cloud).
- **`uploads/`** — original PDF files on disk.

Paths resolve from the **project root** (parent of `backend/`), so the backend behaves the same whether you start it from `backend/` or elsewhere.

---

## How it works (end-to-end)

### A. Document ingestion pipeline

When you upload a PDF, the backend saves the file and runs an async ingestion job:

```
PDF upload
   │
   ▼
Parse PDF (PyMuPDF / pymupdf4llm; optional Docling)
   │  → text, headings, tables, page numbers
   ▼
Chunking (semantic-aware splitting where possible)
   │
   ▼
Contextual enrichment (optional, LLM-assisted)
   │  → short prefixes / Doc2Query-style questions / table summaries
   ▼
Embed chunks (fastembed: BAAI/bge-small-en-v1.5, 384-d)
   │
   ▼
Store
   ├── SQLite: ChunkRecord metadata (page, section, content, …)
   └── Qdrant: vectors + payload for dense retrieval
```

Document status in the UI roughly maps to: `pending` → `parsing` → `chunking` → `embedding` → `ready` (or `failed`).

Only **Ready** documents can be used in a chat session.

### B. Question-answering (agentic RAG) pipeline

When you ask a question in chat:

```
User question
   │
   ├─ (optional) Chat-aware rewrite
   │     Only if the question has pronouns / depends on prior turns.
   │     Standalone questions are left unchanged (avoids CoT poisoning retrieval).
   │
   ├─ Hybrid search per active document
   │     Dense (Qdrant) + BM25 (SQLite chunks) → Reciprocal Rank Fusion
   │     Dense cosine scores are preserved separately from tiny RRF fusion scores.
   │
   ├─ Rerank (cross-encoder)
   │
   ├─ Relevance gate
   │     Uses dense similarity + lexical overlap (and soft rerank signal).
   │     FAIL → refuse immediately: "outside the scope of the uploaded document(s)"
   │            with NO citations / NO PDF highlight
   │
   ├─ (optional) Agentic second-hop retrieval
   │     For multi-hop or borderline scores only
   │
   ├─ LLM generation (context-only prompt)
   │     Answer cleaned of reasoning leaks ("So final", "Provide citation", …)
   │
   ├─ Groundedness check (optional)
   │     FAIL → refuse or self-correct with another retrieval hop
   │
   └─ Citations + UI
         Extract supporting sentence(s) from cited chunks [n]
         Stream answer to chat
         Open PDF on the right and highlight those lines
```

### C. What you see in the UI

1. **Sidebar** — upload PDFs, watch processing status, open a document, manage chats.
2. **Chat** — ask questions; answers stream in; refused answers use a distinct style and have no citation pills.
3. **PDF viewer (right panel)** — opens automatically on a grounded answer (or when you click a citation). Highlights are driven by the **citation snippet** (the sentence that actually supported the answer), not by loose keywords across the whole page.

---

## Tech stack

### Backend

| Piece | Choice |
|-------|--------|
| API | FastAPI + Uvicorn |
| Streaming | SSE (`sse-starlette`) |
| ORM / DB | SQLAlchemy 2.0 async + SQLite (`aiosqlite`) |
| Vectors | Qdrant (`qdrant-client`) |
| Embeddings | `fastembed` (`BAAI/bge-small-en-v1.5`) |
| PDF parsing | PyMuPDF / `pymupdf4llm` (optional Docling) |
| LLMs | Groq and/or Google Gemini (primary + fallback) |
| Config | `pydantic-settings` via `backend/.env` |

### Frontend

| Piece | Choice |
|-------|--------|
| Framework | Next.js 14 (App Router) |
| UI | React 18 + Tailwind CSS |
| Markdown answers | `react-markdown` + `remark-gfm` |
| PDF rendering | `react-pdf` / PDF.js |
| API access | `/api/*` rewritten/proxied to `http://localhost:8000` |

---

## Project layout

```
NLP Assignment/
├── README.md
├── .gitignore
├── backend/
│   ├── .env.example          # copy → .env and fill keys (never commit .env)
│   ├── pyproject.toml
│   └── app/
│       ├── main.py           # FastAPI entry + CORS + routers
│       ├── config.py         # settings from .env
│       ├── types.py          # shared pydantic models
│       ├── api/
│       │   ├── documents.py  # upload, list, status, file, delete, reingest
│       │   └── chat.py       # sessions, history, SSE / non-stream answers
│       ├── db/
│       │   ├── models.py     # Document, ChunkRecord, ChatSession, Message, EvalLog
│       │   └── session.py
│       ├── ingestion/
│       │   ├── pdf.py        # parse PDF → elements
│       │   ├── chunker.py    # element → chunks
│       │   ├── contextual.py # contextual prefixes
│       │   ├── nlp_enhancements.py  # semantic chunking, doc2query, table summary
│       │   ├── embedder.py
│       │   └── pipeline.py   # full ingest orchestration
│       ├── retrieval/
│       │   ├── vector_store.py
│       │   ├── bm25.py
│       │   ├── hybrid_search.py
│       │   ├── reranker.py
│       │   └── query_transform.py  # rewrite / HyDE / second-hop
│       └── generation/
│           ├── agentic.py          # full RAG loop
│           ├── prompt.py           # prompts, refusal text, citations
│           ├── grounding_check.py  # relevance gate + groundedness
│           └── llm_client.py       # Groq/Gemini + CoT cleanup
├── frontend/
│   ├── package.json
│   ├── next.config.js        # proxies /api → backend :8000
│   └── src/
│       ├── app/              # layout, page, styles
│       ├── components/       # Sidebar, ChatPanel, PdfViewer, UploadModal
│       └── lib/              # api client + types
├── uploads/                  # runtime: uploaded PDFs (gitignored)
├── qdrant_data/              # runtime: local Qdrant (gitignored)
└── ragchatbot.db             # runtime: SQLite DB (gitignored)
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- At least one LLM key: **Groq** and/or **Gemini**
- Optional: **Qdrant Cloud** URL + API key (otherwise local on-disk Qdrant is used)

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `backend/.env` and set at least:

```env
GROQ_API_KEY=your_groq_key_here
# and/or
GEMINI_API_KEY=your_gemini_key_here
```

Start the API:

```powershell
# PowerShell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)  
Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```powershell
# PowerShell
cd frontend
npm install
npm run dev
```

```bash
# bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).  
Frontend calls go to `/api/...` and Next.js proxies them to the backend on port **8000**.

> **Windows PowerShell tip:** older PowerShell does not support `&&`. Use `;` or separate lines, e.g. `cd backend; uvicorn app.main:app --reload --port 8000`.

---

## Configuration reference

All options live in `backend/.env` (see `backend/.env.example`).

| Variable | Role |
|----------|------|
| `GROQ_API_KEY` / `GEMINI_API_KEY` | LLM credentials |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` | Which provider is primary vs fallback |
| `LLM_MODEL` / `LLM_FALLBACK_MODEL` | Model IDs |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Embedding model (must match Qdrant collection dim) |
| `HYBRID_ENABLED` | Dense + BM25 fusion |
| `RERANK_ENABLED` / `RERANK_MODEL` | Cross-encoder reranking |
| `RELEVANCE_THRESHOLD` | Minimum dense similarity to consider retrieval usable |
| `RERANK_THRESHOLD` | Soft rerank threshold (logits; not a hard 0–1 probability) |
| `GROUNDEDNESS_ENABLED` | Post-generation “is this supported by context?” check |
| `AGENTIC_ENABLED` | Second-hop retrieval + self-correction on ungrounded drafts |
| `CONTEXTUAL_RETRIEVAL_LLM` | LLM contextual prefixes during ingest |
| `HYDE_ENABLED` | Hypothetical document embeddings for search (off by default) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Use Qdrant Cloud when set |
| `QDRANT_PATH` | Local Qdrant path when cloud URL is empty |
| `UPLOAD_DIR` / `MAX_UPLOAD_MB` | Upload location and size limit |
| `BACKEND_CORS_ORIGINS` | Allowed frontend origins |

---

## Usage walkthrough

1. Start **backend** (`:8000`) and **frontend** (`:3000`).
2. In the sidebar, **upload a PDF**.
3. Wait until status shows **Ready** (first ingest can take a bit: parsing, chunking, embeddings, optional LLM enrichment).
4. Type a question in the chat (a session is auto-created for ready documents if needed).
5. Observe one of two outcomes:
   - **In-document question** → short answer + citation pills → PDF opens on the cited page with the supporting line highlighted.
   - **Out-of-document question** → clear out-of-scope message → **no** citations, **no** highlight panel for that answer.
6. Click a citation pill anytime to jump to that page/snippet again.

### Example behavior

| Question | Expected behavior |
|----------|-------------------|
| “What is the currency of Bangladesh?” *(if stated in the PDF)* | Answer like “Bangladeshi Taka” with `[1]`, page citation, highlight on that sentence |
| “What is the national bird of Bangladesh?” *(if bird is not in the PDF)* | Out-of-scope refusal, no citations |

---

## API overview

Useful backend endpoints (see `/docs` for full schemas):

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/documents/upload` | Upload PDF (multipart) |
| `GET` | `/documents` | List documents + status |
| `GET` | `/documents/{id}/status` | Poll processing status |
| `GET` | `/documents/{id}/file` | Serve original PDF (for viewer) |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/documents/{id}/reingest` | Re-run ingestion |
| `POST` | `/chat/sessions` | Create chat scoped to document IDs |
| `GET` | `/chat/sessions` | List sessions |
| `POST` | `/chat/{session_id}/message` | Ask a question (`stream: true` for SSE) |
| `GET` | `/chat/{session_id}/history` | Message history |
| `DELETE` | `/chat/{session_id}` | Delete session |

SSE events for streaming answers:

- `{ "type": "token", "content": "..." }` — answer text chunks  
- `{ "type": "done", "citations": [...], "refused": bool, "refusal_reason": "..." }` — final metadata  

---

## Design choices worth knowing

- **Refuse early when retrieval is weak** — out-of-scope questions should not burn LLM quota or invent answers.
- **Preserve dense cosine scores through hybrid RRF** — RRF fusion scores are tiny (~0.01–0.03) and must not be mistaken for similarity when gating relevance.
- **Don’t rewrite every follow-up** — rewriting standalone questions with reasoning models can dump chain-of-thought into the *search query* and break retrieval.
- **Strip model reasoning from user-visible answers** — cleans patterns like “Provide citation… Use only context. So final.…”
- **Citations point at supporting sentences** — snippet selection prefers overlap with the *answer text*, then the query; the PDF highlighter matches those phrases tightly.

---

## Troubleshooting

| Symptom | Things to check |
|---------|-----------------|
| Frontend can’t reach API | Backend running on `:8000`? `next.config.js` proxy? CORS origins? |
| Upload never becomes Ready | Backend logs during ingest; PDF parse errors; embedding/Qdrant connectivity |
| Every answer is “out of scope” | Is the doc **Ready**? Are LLM keys set? Is retrieval finding chunks (check logs for relevance gate scores)? |
| Answers include weird reasoning text | Ensure latest `llm_client` / `finalize_answer` path is running (restart uvicorn) |
| No PDF highlight | Answer must be grounded with citations; snippet must match text layer tokens on that page |
| Rate-limit / unavailable message | Provider quota (Groq/Gemini); fallback key configured? |

---

## What is not committed to Git

For safety and size, these stay local (see `.gitignore`):

- `backend/.env` (secrets)
- `frontend/node_modules/`, `frontend/.next/`
- `uploads/`, `qdrant_data/`, `ragchatbot.db`

Clone → create `.env` from `.env.example` → install → run. Your PDFs and vectors are created again when you upload documents.
