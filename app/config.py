"""Central config — loads env vars and defines tunable thresholds."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from project root (one level up from app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# --- Paths ---
BOOK_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache.db")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")

# --- Embeddings ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Retriever ---
RETRIEVAL_TOP_K = 4
# Minimum cosine similarity for retrieved chunks to be considered relevant.
# Below this → out-of-scope decline, no LLM call.
SCOPE_THRESHOLD = 0.25

# --- Cache ---
CACHE_SIMILARITY_THRESHOLD = 0.88   # cosine sim for semantic cache match
CACHE_JACCARD_THRESHOLD = 0.65      # key-term overlap required (tuned from 0.80)
CACHE_TOP_K = 3                     # candidates to pull from cache FAISS

# --- Versioning (bump when textbook or prompt changes) ---
TEXTBOOK_VERSION = "ncert-10-sci-v1"
PROMPT_VERSION = "v1"

# --- API ---
API_URL = os.getenv("API_URL", "")
