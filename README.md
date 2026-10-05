# SciQuery — NCERT Class 10 Science Chatbot

A doubt-solving chatbot for the NCERT Class 10 Science textbook, featuring a multi-guard semantic cache that safely reuses past answers with near-zero latency.

**Live Link**: [Streamlit App Demo (Deployment Placeholder)](https://share.streamlit.io)  
**Repository**: [https://github.com/Dhanugupta0/sciquery-cache.git](https://github.com/Dhanugupta0/sciquery-cache.git)  
**Author**: Dhanu Gupta | AI Intern Assignment, Prepzy.ai

---

## Features

- Answers strictly from the **NCERT Class 10 Science** textbook
- Cites textbook chapter(s) for every in-scope answer
- Smart cache with 5 safety guards (number, symbol, contrast, key-term, question-type)
- Multi-turn conversation handling with autonomous follow-up rewriting
- Style detection ("explain simpler", "in points") to restyle without caching
- Zero-LLM out-of-scope question rejection with empty citations
- Measured sub-millisecond store-level exact cache hits

## Latency Benchmarks (Measured)

| Path | Store Latency | End-to-End API Latency |
|------|---------------|------------------------|
| **Exact Cache Hit** | 0.08 ms median | 4.59 ms median |
| **Semantic Cache Hit** (FAISS + 5 guards) | 5.79 ms median | 10.73 ms median |
| **Fresh LLM Generation** (uncontended) | N/A | 613.96 ms median (386 – 1739 ms) |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Dhanugupta0/sciquery-cache.git
cd sciquery-cache
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your Groq API key
```

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | API key | `gsk_...` |
| `LLM_MODEL` | Active LLM model | `qwen/qwen3.8-27b` |
| `API_URL` | (Optional) Backend URL. If unset, Streamlit starts the backend itself | `http://localhost:8000` |

### 3. Build the Book Index (Already committed in `data/index/`)

```bash
python scripts/ingest.py
```

This reads NCERT PDFs from `data/pdfs/`, creates chunk embeddings, and saves the FAISS index to `data/index/`.

### 4. Run the Application

**Option A: Streamlit UI (Embedded backend)**

```bash
streamlit run streamlit_app.py
```

**Option B: Standalone FastAPI Backend + Streamlit UI**

```bash
# Terminal 1: Run FastAPI backend
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Run Streamlit frontend pointing to FastAPI
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

---

## API Format

### `GET /health`

```json
{"status": "ok"}
```

### `POST /session`

```json
// Response
{"session_id": "a1b2c3d4e5f6"}
```

### `POST /chat`

```json
// Request
{
  "session_id": "a1b2c3d4e5f6",
  "message": "What is refraction?"
}

// Response (Cache Hit)
{
  "reply": "Refraction is the bending of light when it travels obliquely from one transparent medium into another, causing a change in its direction.",
  "citations": ["Light – Reflection and Refraction"],
  "cache_hit": true,
  "latency_ms": 4.59
}
```

**Out-of-Scope Response:**
```json
{
  "reply": "I'm sorry, this question doesn't seem to be covered in the NCERT Class 10 Science textbook. I can only help with topics from that book. Could you ask something from the textbook?",
  "citations": [],
  "cache_hit": false,
  "latency_ms": 84.9
}
```

**Status & Error Codes:**
- `200` — Success
- `404` — Unknown session ID (`{"detail": "Session not found"}`)
- `422` — Empty message (`{"detail": "Message cannot be empty"}`)
- `429` — Upstream rate limit with friendly retry message (`{"detail": "Too many requests right now. Please wait a moment and try again."}`)

---

## Running Tests

Install test dependencies and run pytest:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Key test files:
- `tests/test_cache_cases.py` — Exact/semantic cache decisions, lookalikes, guards, and style tests
- `tests/test_pipeline.py` — End-to-end pipeline, out-of-scope detection, and multi-turn tests
- `tests/test_api.py` — FastAPI endpoint format, session validation, and error tests

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (`data/index/` included, `.env` excluded)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repository: `Dhanugupta0/sciquery-cache`
4. Set main file path: `streamlit_app.py`
5. Under **Settings → Secrets**, add:
   ```toml
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_API_KEY = "gsk_your_key_here"
   LLM_MODEL = "qwen/qwen3.8-27b"
   ```
6. Deploy. The Streamlit app runs FastAPI in a background thread and communicates via HTTP.

---

## Project Structure

```
sciquery-cache/
├── streamlit_app.py          # Streamlit UI frontend
├── app/
│   ├── api.py                # FastAPI routes & error handling
│   ├── config.py             # Configuration & threshold tuning
│   ├── graph.py              # LangGraph workflow pipeline
│   ├── llm.py                # Groq/OpenAI client & prompts
│   ├── retriever.py          # NCERT FAISS chunk retriever
│   ├── sessions.py           # In-memory session store
│   └── cache/
│       ├── normalize.py      # Query normalization & symbol extraction
│       ├── guards.py         # Multi-stage safety guards
│       ├── policy.py         # Caching & style detection policies
│       └── store.py          # SQLite + FAISS cache index
├── scripts/
│   └── ingest.py             # NCERT PDF chunker & index builder
├── tests/
│   ├── test_cache_cases.py   # Cache decision & guard tests
│   ├── test_pipeline.py      # Pipeline & out-of-scope tests
│   └── test_api.py           # API endpoint format tests
├── data/
│   ├── pdfs/                 # NCERT textbook PDFs
│   └── index/                # Committed book FAISS vector index
├── docs/
│   ├── explainer.md          # 1-page architecture explainer
│   ├── explainer.pdf         # Exported 1-page PDF documentation
│   ├── flowchart.mmd         # Pipeline flowchart (Mermaid)
│   └── flowchart.png         # Rendered flowchart diagram
├── NOTES.md                  # Experimentation log with measured numbers
├── README.md                 # Project documentation
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development & testing dependencies
└── .env.example              # Sample environment configuration
```
