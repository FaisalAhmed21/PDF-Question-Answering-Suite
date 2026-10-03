# Grounded PDF Question Answering Suite: RAG Assistant and Transformer-Based Extractive QA

A pair of NLP systems that let you upload a PDF, ask natural-language questions, and receive answers that are **strictly grounded in the document**, with a clear refusal whenever the PDF does not contain the answer. The repository provides the same capability through two different architectures:

- a full-stack **Retrieval-Augmented Generation (RAG)** web application, where an LLM writes the answer, with page citations and in-viewer highlighting of the exact supporting lines; and
- a standalone **Transformer-Based Extractive QA** notebook, which needs no LLM or API keys and answers by selecting the exact text from the PDF.

> **Repository at a glance**
>
> | | Solution | Where | Approach | Needs an LLM / API key? |
> |---|----------|-------|----------|--------------------------|
> | 1 | **RAG-Powered PDF Q&A Assistant** (web app) | `backend/` + `frontend/` | Retrieval-Augmented Generation | Yes (Groq and/or Gemini) |
> | 2 | **Transformer-Based PDF Question Answering System** (notebook) | `Transformer_Based_PDF_Question_Answering_System.ipynb` | Retrieve-and-read *extractive* QA | **No** |
>
> Solution 1 is documented throughout this README. Solution 2 has its own section:
> [Companion Notebook: Transformer-Based PDF Question Answering System](#companion-notebook-transformer-based-pdf-question-answering-system).

---

## Solution 1: What the RAG web app does

Typical “chat with PDF” demos often invent facts when the document does not contain the answer. The RAG web app is built around the opposite goal:

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
├── Transformer_Based_PDF_Question_Answering_System.ipynb   # standalone notebook: extractive QA, no LLM / no API keys
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

#### Qdrant configuration

Qdrant is required for the dense/vector part of retrieval; it is not an unused
optional setting. During ingestion, chunk embeddings are upserted into the
`document_chunks` Qdrant collection. During chat, the question embedding is
searched in that collection and combined with the SQLite-backed BM25 results
when hybrid retrieval is enabled.

The default configuration uses Qdrant's local on-disk mode, so no separate
Qdrant server is needed.

To use Qdrant Cloud (or a self-hosted Qdrant server), set `QDRANT_URL`. When
`QDRANT_URL` is non-empty, it takes precedence over `QDRANT_PATH`, and the
backend connects to that URL instead of using the local `qdrant_data/`
directory. Qdrant Cloud also requires the matching API key:

```env
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=document_chunks
```

`QDRANT_API_KEY` is passed to the Qdrant client only when a URL is configured.
Keep it in `backend/.env`; do not commit it. If you switch between local and
remote Qdrant, the vectors are stored in different places, so documents must
be uploaded/re-ingested in the selected store.


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

## Companion Notebook: Transformer-Based PDF Question Answering System

**File:** `Transformer_Based_PDF_Question_Answering_System.ipynb` (repository root)

A single, self-contained Jupyter / Google Colab notebook that solves the same core problem as the web app (*answer a question from a PDF, and clearly say so when the PDF does not contain the answer*) using a classic NLP architecture instead of a generative LLM. Running the notebook gives you an upload button; after the PDF is processed it asks for your question and answers **only** from the document. If the answer is not supported by the PDF, it replies:

> *I don't have proper context to answer that question.*

### Why this notebook exists and why it matters

| Reason | Explanation |
|--------|-------------|
| **Shows the NLP fundamentals explicitly** | Text extraction and cleaning, sentence segmentation, sentence embeddings, semantic similarity search, tokenization, transformer span prediction, and SQuAD 2.0-style *unanswerable-question* detection are all visible, readable code. In the web app most of this is delegated to libraries and an LLM. |
| **Zero infrastructure** | One file. No FastAPI, Next.js, SQLite or Qdrant, no `.env`, no API keys. It runs in a free Colab session or any local Jupyter in a few minutes, so anyone can reproduce and verify it. |
| **Cannot hallucinate by construction** | The model never writes new text. It *selects* a span or sentence that already exists in the PDF, so every answer is traceable to a page and an evidence snippet. |
| **Private and free to run** | After the one-time model download nothing leaves your machine/runtime, and there is no API quota, rate limit or cost. |
| **A non-LLM baseline** | It gives the project an honest baseline to compare against the RAG app: same documents, same questions, two very different architectures. This makes the trade-offs between extractive QA and generative RAG concrete instead of theoretical. |
| **Explainable output** | Every answer comes with a confidence score, the page number, and the surrounding evidence text. |

### What it does (step by step)

```
Upload PDF
   │
   ▼
1. Extract text per page                 (pypdf)
   │
   ▼
2. Clean the text                        remove decorative junk such as "$$$$%%%%&&&&" or "(_ ... _)"
   │
   ▼
3. Split into sentences                  keep page numbers; glue tiny fragments; cap very long sentences
   │
   ▼
4. Embed every sentence                  sentence-transformers/all-MiniLM-L6-v2 (normalized vectors)
   │
   ▼  ── at question time ──
5. Embed the question, rank sentences    cosine similarity, keep the top 5
   │
   ▼
6. Relevance gate                        best similarity too low?  → "I don't have proper context"
   │
   ▼
7. Key-word check                        a key word of the question (e.g. "bird") never appears in the PDF?
   │                                     → "I don't have proper context"
   ▼
8. Tier 1 – extractive QA                for each candidate sentence (plus its neighbours on the same page)
   │                                     run deepset/roberta-base-squad2 and pick the best answer span
   │                                     confidence = P(real answer) vs. the model's own "no answer" option
   │                                     confidence >= 0.50 → return the short answer + page + evidence
   ▼
9. Tier 2 – sentence fallback            no confident short span, but a clearly matching sentence exists
   │                                     (typical for "why / for what purpose / how" questions)
   │                                     → return that sentence (+ the next one) with its page
   ▼
10. Otherwise                            "I don't have proper context to answer that question."
```

**Models used**

| Role | Model | Notes |
|------|-------|-------|
| Retrieval (embeddings) | `sentence-transformers/all-MiniLM-L6-v2` | Small BERT-style encoder (about 22M parameters), 384-dimensional vectors |
| Reader (answer extraction) | `deepset/roberta-base-squad2` | RoBERTa-base fine-tuned on SQuAD 2.0, which includes unanswerable questions (about 125M parameters) |

Both are encoder-only transformers that run locally. Neither is a generative LLM. The reader is loaded with `AutoTokenizer` / `AutoModelForQuestionAnswering` rather than the `"question-answering"` pipeline, so it also works on newer `transformers` releases where that pipeline was removed.

**How the "no proper context" decision is made.** Three independent safeguards must all be passed:

1. **Semantic gate**: the best-matching sentence must be similar enough to the question (`MIN_RETRIEVAL_SIM`).
2. **Key-word gate**: the question's content words (after removing stop words and generic words such as *purpose*, *important*, *crucial*) must actually occur in the PDF, using light stemming. This is what stops a question such as *"What is the national bird?"* from being answered with *"Royal Bengal Tiger"* when the PDF only mentions a national *animal*. Questions with four or more key words may miss one.
3. **Confidence gate**: the reader must prefer a real span over its own "no answer" option (`MIN_ANSWER_SCORE`), or a sentence must be clearly relevant (`MIN_SENTENCE_SIM`) for the fallback.

### How to run it

1. Open the notebook in **Google Colab** (recommended) or a local **Jupyter / VS Code** environment.
2. Run all cells from top to bottom (Colab: *Runtime → Run all*). The first run installs packages and downloads the two models (about 500 MB); a GPU is not required.
3. When the last cell runs, upload your PDF:
   - **Colab:** a *Choose Files* button appears; afterwards you are prompted `Your question (or type 'exit'):` in a loop.
   - **Jupyter / VS Code:** an *Upload PDF* button appears; once processing finishes, a question box and an *Ask* button appear.
4. Ask questions. Each answer shows the extracted answer, a confidence score, the page, and the supporting evidence.

**Installed by the notebook:** `pypdf`, `transformers`, `sentence-transformers`, `torch`, `ipywidgets`.

### Tunable settings

All are defined at the top of the *Core logic* cell.

| Setting | Default | Meaning |
|---------|---------|---------|
| `TOP_K` | `5` | Candidate sentences examined per question |
| `CONTEXT_WINDOW` | `1` | Neighbouring sentences added on each side for the reader |
| `MIN_RETRIEVAL_SIM` | `0.25` | Best similarity below this means nothing in the PDF relates to the question |
| `MIN_ANSWER_SCORE` | `0.50` | Reader confidence needed for a short extracted answer |
| `MIN_SENTENCE_SIM` | `0.40` | Similarity needed to return a whole sentence (fallback) |
| `MAX_SENT_WORDS` | `60` | Very long sentences are split into pieces of at most this many words |

If the tool refuses valid questions, lower the thresholds slightly. If it answers things it should not, raise `MIN_SENTENCE_SIM` / `MIN_ANSWER_SCORE`.

### Example behaviour

Using the sample test PDF:

| Question | Result |
|----------|--------|
| *What is the currency of Bangladesh?* | `Bangladeshi Taka`, high confidence, with the supporting sentence (page 2) |
| *What is the official language of Bangladesh?* | `Bengali`, high confidence (page 1) |
| *For what purpose Padma river is crucial for?* | `transportation, agriculture, fishing, and daily life` (page 1) |
| *What is the national bird of Bangladesh?* | *I don't have proper context to answer that question.* (the PDF mentions a national **animal**, not a bird) |

### How it differs from the main project (the RAG web app)

Both solutions share the same idea: ingest a PDF, retrieve the most relevant passages, check that the document really covers the question, answer from that context, and refuse otherwise. They differ in **how the answer is produced and how much machinery surrounds it**.

| Aspect | RAG web app (`backend/` + `frontend/`) | Notebook (extractive QA) |
|--------|----------------------------------------|--------------------------|
| **Paradigm** | Retrieval-**Augmented Generation**: an LLM writes the answer | Retrieve-and-**read**: a reader model *extracts* the answer from the text |
| **Answer produced by** | Groq / Gemini LLM (context-only prompt) | `roberta-base-squad2` span prediction, or a returned sentence |
| **Uses an LLM?** | Yes | No (small encoder-only transformers) |
| **API keys / cost / rate limits** | Required; subject to provider quotas | None |
| **Parsing** | PyMuPDF / `pymupdf4llm` (headings, tables, optional Docling) | `pypdf` plus junk-symbol cleaning |
| **Chunking** | Semantic chunks, optional LLM contextual enrichment, Doc2Query | Sentences with page numbers |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (fastembed) | `all-MiniLM-L6-v2` |
| **Search** | Dense (Qdrant) + BM25, fused with RRF | Dense (cosine) only |
| **Reranking** | Cross-encoder reranker | None |
| **"Is it in the PDF?" check** | Relevance gate (dense similarity + lexical overlap + soft rerank signal) | Similarity gate + key-word gate + confidence gate |
| **Verification** | Groundedness check and optional second retrieval hop | Reader confidence vs. its "no answer" score |
| **Multiple documents / chat history** | Yes (sessions, history in SQLite) | One PDF per run, no memory between questions |
| **Output** | Streamed, fluent answer with `[n]` citations and highlighted PDF viewer | Short extracted answer, confidence, page, evidence text |
| **Hallucination risk** | Mitigated (grounding prompt and checks) but inherent to generation | None: text is copied from the PDF |
| **Setup** | Python backend, Node frontend, Qdrant, `.env` | One notebook |
| **Best suited for** | A product-style experience over many documents | Demonstration, coursework, reproducible baseline, offline use |

**In one sentence:** the web app *reads the retrieved text and writes an answer*, while the notebook *finds the exact words in the retrieved text that answer the question*.

**When to use which**

- Use the **notebook** when you need a quick, reproducible, offline, no-cost demonstration of the NLP pipeline, a non-LLM baseline, or guaranteed answers that are literal quotations from the PDF.
- Use the **web app** when you need fluent, multi-sentence answers, many documents, chat history, page citations with in-viewer highlighting, or multi-hop questions.

### Limitations of the notebook

- **Extractive only:** answers are short spans or sentences copied from the PDF. It cannot summarise, combine facts from far-apart parts of the document, or rephrase.
- **Text-based PDFs only:** scanned/image-only PDFs have no extractable text and need OCR first (the notebook reports this).
- **Strict key-word check:** a question using a word the PDF never uses (for example *people* when the PDF says *population*) is treated as having no context. Rephrase the question or relax `question_terms` / `keywords_supported` in the code.
- **One document at a time**, English text, and no conversational memory.
- **No in-viewer highlighting:** evidence is shown as text with the page number.

### Notebook troubleshooting

| Symptom | Things to check |
|---------|-----------------|
| `KeyError: Unknown task question-answering` | You are running a notebook version that uses `pipeline("question-answering")` on a new `transformers` release. Use the version in this repository, which loads the model directly. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. Optionally set an `HF_TOKEN` for faster downloads. |
| "No readable text found in this PDF" | The PDF is probably scanned images; run OCR first. |
| Valid questions answered with "no proper context" | The question may use a word the PDF never contains (key-word gate), or lower `MIN_RETRIEVAL_SIM` / `MIN_ANSWER_SCORE` / `MIN_SENTENCE_SIM` slightly. |
| Wrong or loosely related answers | Raise `MIN_ANSWER_SCORE` / `MIN_SENTENCE_SIM`. |
| First run is slow | Models (about 500 MB) are downloaded once; later runs reuse the cache. |

---

## Troubleshooting (RAG web app)

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
