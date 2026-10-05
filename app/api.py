"""FastAPI routes — /session, /chat, /health."""

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from app.graph import init_pipeline, run_pipeline, get_sessions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="NCERT Science Chatbot", version="1.0.0")


# ---- startup ----

@app.on_event("startup")
def startup():
    log.info("Starting up — loading models and indexes...")
    init_pipeline()
    log.info("Ready")


# ---- schemas ----

class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


class ChatResponse(BaseModel):
    reply: str
    citations: list[str]
    cache_hit: bool
    latency_ms: float


class SessionResponse(BaseModel):
    session_id: str


# ---- routes ----

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=SessionResponse)
def create_session():
    sid = get_sessions().create()
    log.info(f"Created session: {sid}")
    return SessionResponse(session_id=sid)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sessions = get_sessions()
    if sessions.get(req.session_id) is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    t0 = time.perf_counter()
    try:
        result = run_pipeline(req.session_id, req.message)
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log.exception("Pipeline error")
        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str or getattr(e, "status_code", None) == 429:
            msg = "Too many requests right now. Please wait a moment and try again."
        elif "api key" in err_str or "401" in err_str or "credentials" in err_str or "invalid_api_key" in err_str:
            msg = "⚠️ Groq API key is missing or invalid. Please configure LLM_API_KEY in Streamlit Cloud Secrets (Settings → Secrets) or enter it in the sidebar."
        else:
            msg = "Sorry, something went wrong. Please try again."
        return ChatResponse(
            reply=msg,
            citations=[],
            cache_hit=False,
            latency_ms=elapsed,
        )

    return ChatResponse(
        reply=result.get("reply", ""),
        citations=result.get("citations", []),
        cache_hit=result.get("cache_hit", False),
        latency_ms=result.get("latency_ms", 0),
    )
