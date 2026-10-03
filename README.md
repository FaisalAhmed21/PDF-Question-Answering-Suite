# PDF Question Answering (RAG)

RAG-powered PDF Q&A with hybrid retrieval, grounded generation, and page citations.

## Stack

- **Backend:** FastAPI, SQLite, Qdrant, fastembed, Groq/Gemini
- **Frontend:** Next.js 14, React, Tailwind, react-pdf

## Prerequisites

- Python 3.10+
- Node.js 18+
- Groq and/or Gemini API key
- Optional: Qdrant Cloud account (or local disk storage)

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e .

# Copy env template and fill in your keys
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Paths for the database, uploads, and local Qdrant are resolved from the **project root** (parent of `backend/`), so the server works the same whether you start it from `backend/` or elsewhere.

**PowerShell (Windows):**
```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**bash / macOS / Linux:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: http://localhost:8000/health

### 2. Frontend

**PowerShell (Windows):**
```powershell
cd frontend
npm install
npm run dev
```

**bash / macOS / Linux:**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — API calls under `/api/*` are proxied to the backend.

> **Note:** On Windows PowerShell 5.x, use `;` or separate lines instead of `&&` (e.g. `cd backend; uvicorn app.main:app --reload --port 8000`).

## Usage

1. Upload a PDF from the sidebar.
2. Wait until status is **Ready**.
3. Start a chat and ask questions.
4. Click citation pills to open the source page in the PDF viewer.

## Project layout

```
├── backend/
│   ├── app/
│   │   ├── api/           # documents + chat routes
│   │   ├── db/            # SQLAlchemy models
│   │   ├── generation/    # LLM + agentic RAG
│   │   ├── ingestion/     # PDF parse → chunk → embed
│   │   └── retrieval/     # hybrid search + rerank
│   ├── .env.example
│   └── pyproject.toml
├── frontend/
│   └── src/
├── uploads/               # uploaded PDFs
├── qdrant_data/           # local Qdrant (if no cloud URL)
└── ragchatbot.db          # SQLite database
```

## Configuration

See `backend/.env.example` for all options. Important flags:

| Variable | Meaning |
|----------|---------|
| `GROQ_API_KEY` / `GEMINI_API_KEY` | LLM providers |
| `QDRANT_URL` | Cloud Qdrant; leave empty to use `QDRANT_PATH` locally |
| `AGENTIC_ENABLED` | Second-hop retrieval + ungrounded self-correction |
| `GROUNDEDNESS_ENABLED` | Post-answer grounding check |
| `HYBRID_ENABLED` / `RERANK_ENABLED` | Retrieval quality features |
