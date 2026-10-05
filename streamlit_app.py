"""Streamlit frontend for the NCERT Science Chatbot."""

import os
import threading
import time

import httpx
import streamlit as st

# Sync Streamlit secrets to os.environ so FastAPI backend thread has them
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ[_k] = _v.strip().strip('"').strip("'")
        elif isinstance(_v, dict):
            for _subk, _subv in _v.items():
                if isinstance(_subv, str):
                    os.environ[_subk] = _subv.strip().strip('"').strip("'")
except Exception:
    pass

from app.config import get_api_key
_resolved_key = get_api_key()
os.environ["LLM_API_KEY"] = _resolved_key
os.environ["OPENAI_API_KEY"] = _resolved_key

# ---- config ----

API_URL = os.getenv("API_URL", "").rstrip("/")


# ---- backend server ----

@st.cache_resource
def start_backend() -> str:
    """Start FastAPI in a background thread and return its base URL.

    This is the default mode — used on Streamlit Cloud and local dev.
    All Streamlit ↔ backend communication goes through HTTP.
    Set API_URL env var only if you want to point at an external server.
    """
    if API_URL:
        return API_URL

    import uvicorn
    from app.api import app

    port = 8321  # unlikely to conflict

    def run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Wait for server to be ready
    base = f"http://127.0.0.1:{port}"
    for _ in range(30):
        try:
            r = httpx.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                return base
        except Exception:
            pass
        time.sleep(0.5)

    st.error("Backend failed to start")
    st.stop()


# ---- helpers ----

def api_post(base_url: str, path: str, json_data: dict = None) -> dict:
    """POST to the backend API."""
    r = httpx.post(f"{base_url}{path}", json=json_data or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def api_get(base_url: str, path: str) -> dict:
    r = httpx.get(f"{base_url}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def create_session(base_url: str) -> str:
    data = api_post(base_url, "/session")
    return data["session_id"]


# ---- page config ----

st.set_page_config(
    page_title="SciQuery — NCERT Class 10 Science",
    page_icon="🔬",
    layout="centered",
)

# ---- custom CSS ----

st.markdown("""
<style>
    .meta-info {
        font-size: 0.78em;
        color: #888;
        margin-top: 4px;
        padding: 4px 8px;
        border-left: 2px solid #444;
    }
    .cache-hit { color: #4CAF50; font-weight: 600; }
    .cache-miss { color: #FF9800; font-weight: 600; }
    .stChatMessage { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---- init ----

base_url = start_backend()

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session(base_url)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- sidebar ----

with st.sidebar:
    st.title("🔬 SciQuery")
    st.caption("NCERT Class 10 Science Chatbot")
    st.divider()

    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.session_id = create_session(base_url)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # API Key Configuration
    active_key = get_api_key()
    if active_key and active_key != "dummy-key-pending-secrets":
        st.caption("🟢 Groq API Key active")
    else:
        st.warning("⚠️ Groq API Key missing")
        key_input = st.text_input(
            "Enter Groq Key:",
            type="password",
            placeholder="gsk_...",
            help="Get a free key from console.groq.com or set LLM_API_KEY in Streamlit Secrets.",
        )
        if key_input:
            clean = key_input.strip().strip('"').strip("'")
            os.environ["LLM_API_KEY"] = clean
            os.environ["OPENAI_API_KEY"] = clean
            st.success("Key applied!")
            st.rerun()

    st.divider()
    st.markdown(
        "Ask any doubt from the **NCERT Class 10 Science** textbook. "
        "I'll answer using only the textbook and cite the chapter."
    )
    st.caption("Built by Dhanu Gupta")


# ---- chat display ----

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            citations = ", ".join(meta.get("citations", [])) or "—"
            pages = meta.get("pages", [])
            pages_str = f' (pp. {", ".join(str(p) for p in pages)})' if pages else ""
            cache_class = "cache-hit" if meta.get("cache_hit") else "cache-miss"
            cache_label = "✅ Cache hit" if meta.get("cache_hit") else "🔄 Fresh answer"
            latency = meta.get("latency_ms", 0)
            st.markdown(
                f'<div class="meta-info">'
                f'📖 {citations}{pages_str} &nbsp;|&nbsp; '
                f'<span class="{cache_class}">{cache_label}</span> &nbsp;|&nbsp; '
                f'⏱️ {latency:.0f} ms'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---- input ----

if prompt := st.chat_input("Ask a question from NCERT Class 10 Science..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                data = api_post(base_url, "/chat", {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                })
                reply = data["reply"]
                meta = {
                    "citations": data.get("citations", []),
                    "pages": data.get("pages", []),
                    "cache_hit": data.get("cache_hit", False),
                    "latency_ms": data.get("latency_ms", 0),
                }
            except Exception as e:
                reply = f"Error: {e}"
                meta = {"citations": [], "pages": [], "cache_hit": False, "latency_ms": 0}

        st.markdown(reply)

        citations = ", ".join(meta.get("citations", [])) or "—"
        pages = meta.get("pages", [])
        pages_str = f' (pp. {", ".join(str(p) for p in pages)})' if pages else ""
        cache_class = "cache-hit" if meta.get("cache_hit") else "cache-miss"
        cache_label = "✅ Cache hit" if meta.get("cache_hit") else "🔄 Fresh answer"
        latency = meta.get("latency_ms", 0)
        st.markdown(
            f'<div class="meta-info">'
            f'📖 {citations}{pages_str} &nbsp;|&nbsp; '
            f'<span class="{cache_class}">{cache_label}</span> &nbsp;|&nbsp; '
            f'⏱️ {latency:.0f} ms'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "meta": meta,
    })
