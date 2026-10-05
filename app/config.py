"""Central config — loads env vars and defines tunable thresholds."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from project root (one level up from app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Sync Streamlit secrets if running inside Streamlit Cloud
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
        elif isinstance(_v, dict):
            for _subk, _subv in _v.items():
                if isinstance(_subv, str) and _subk not in os.environ:
                    os.environ[_subk] = _subv
except Exception:
    pass

def get_api_key() -> str:
    """Resolve API key dynamically from env or Streamlit secrets."""
    for var in ("LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(var)
        if val and val.strip() and val.strip() != "dummy-key-pending-secrets":
            return val.strip().strip('"').strip("'")

    try:
        import streamlit as st
        for var in ("LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
            if var in st.secrets and st.secrets[var]:
                val = str(st.secrets[var]).strip().strip('"').strip("'")
                if val and val != "dummy-key-pending-secrets":
                    return val
        for sec in st.secrets.values():
            if isinstance(sec, dict):
                for var in ("LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
                    if var in sec and sec[var]:
                        val = str(sec[var]).strip().strip('"').strip("'")
                        if val and val != "dummy-key-pending-secrets":
                            return val
    except Exception:
        pass

    return ""


# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = get_api_key() or "dummy-key-pending-secrets"
os.environ["LLM_API_KEY"] = LLM_API_KEY
os.environ["OPENAI_API_KEY"] = LLM_API_KEY

LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_NUMERIC_MODEL = os.getenv("LLM_NUMERIC_MODEL", "openai/gpt-oss-120b")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")

# --- Paths ---
BOOK_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache.db")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")

# --- Embeddings ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Retriever ---
RETRIEVAL_TOP_K = 6
# --- Scope Thresholds (Borderline Band) ---
# Below SCOPE_MIN_THRESHOLD (0.58) -> declines without calling the LLM.
# Between 0.58 and 0.68 -> borderline band: LLM decides using in_scope.
# 0.68 and above -> answers.
SCOPE_MIN_THRESHOLD = 0.58
SCOPE_THRESHOLD = 0.68

# --- Cache ---
CACHE_SIMILARITY_THRESHOLD = 0.88   # cosine sim for semantic cache match
CACHE_JACCARD_THRESHOLD = 0.65      # key-term overlap required (tuned from 0.80)
CACHE_TOP_K = 3                     # candidates to pull from cache FAISS

# --- Versioning (bump when textbook or prompt changes) ---
TEXTBOOK_VERSION = "ncert-10-sci-v1"
PROMPT_VERSION = "v1"

# --- API ---
API_URL = os.getenv("API_URL", "")
