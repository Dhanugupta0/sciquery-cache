# SciQuery — NCERT Class 10 Science Chatbot

A doubt-solving chatbot for the NCERT Class 10 Science textbook, with a smart cache that reuses past answers safely.

**By Dhanu Gupta** | AI Intern Assignment, Prepzy.ai

## Features

- Answers using **only** the NCERT Class 10 Science textbook
- Cites the chapter(s) used in every answer
- Smart cache with 4 safety guards (number, contrast, key-term, question-type)
- Multi-turn conversations with follow-up understanding
- Style requests ("explain simpler", "in points")
- Out-of-scope detection without calling the LLM
- Cache hit latency < 1 ms

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd ncert-chatbot
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your API key
```

Required env vars:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | API key | `gsk_...` |
| `LLM_MODEL` | Model name | `llama-3.1-8b-instant` |
| `API_URL` | (Optional) Backend URL. If unset, Streamlit starts the backend itself | `http://localhost:8000` |

### 3. Build the book index (once)

```bash
python scripts/ingest.py
```

This reads the NCERT PDFs from `data/pdfs/`, creates embeddings, and saves the FAISS index to `data/index/`.

### 4. Run

**Option A: Just Streamlit (recommended for Streamlit Cloud)**

```bash
streamlit run streamlit_app.py
```

The app starts the FastAPI backend automatically in a background thread.

**Option B: Separate backend and frontend**

```bash
# Terminal 1
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

### 5. (Optional) Seed the cache

```bash
python scripts/seed_cache.py
```

Pre-fills the cache with common questions to reduce cold-start LLM calls.

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
{"session_id": "a1b2c3d4e5f6", "message": "What is refraction?"}

// Response
{
  "reply": "Refraction is the bending of light...",
  "citations": ["Light – Reflection and Refraction"],
  "cache_hit": false,
  "latency_ms": 1234.5
}
```

**Error codes:**
- `404` — Unknown session ID
- `422` — Empty message

## Running Tests

```bash
python -m pytest tests/ -v
```

Key test files:
- `tests/test_cache_cases.py` — Cache decision tests (6 required pairs + 16 extras)
- `tests/test_api.py` — Endpoint format tests

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (including `data/index/` but **not** `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `streamlit_app.py`
5. Add secrets in the Streamlit secrets manager (Settings → Secrets):

```toml
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY = "gsk_your_key_here"
LLM_MODEL = "llama-3.1-8b-instant"
```

6. Deploy. The app starts the backend internally — no separate server needed.

> ⚠️ **Keep the app running until evaluation ends.** Check that your Groq API key hasn't expired.

## Project Structure

```
├── streamlit_app.py          # Frontend entry point
├── app/
│   ├── api.py                # FastAPI routes
│   ├── config.py             # Env vars, thresholds
│   ├── graph.py              # LangGraph pipeline
│   ├── llm.py                # LLM client + prompts
│   ├── retriever.py          # Book FAISS search
│   ├── sessions.py           # Session store
│   └── cache/
│       ├── normalize.py      # Text normalizing, extraction
│       ├── guards.py         # Safety guards
│       ├── policy.py         # What to cache / not cache
│       └── store.py          # SQLite + cache FAISS
├── scripts/
│   ├── ingest.py             # Build the book index
│   └── seed_cache.py         # Pre-fill cache
├── tests/
│   ├── test_cache_cases.py   # Cache decision tests
│   └── test_api.py           # API tests
├── data/
│   ├── pdfs/                 # NCERT PDF files
│   └── index/                # FAISS index (committed)
├── docs/
│   ├── explainer.md          # One-page explainer
│   └── flowchart.mmd         # Request flow diagram
├── NOTES.md                  # Experiment log
├── README.md
├── requirements.txt
└── .env.example
```

## Docs

- [One-page explainer](docs/explainer.md) — How it all works
- [NOTES.md](NOTES.md) — Honest experiment log with real numbers
- [Flowchart](docs/flowchart.mmd) — Request flow in Mermaid
