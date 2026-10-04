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

    try:
        result = run_pipeline(req.session_id, req.message)
    except Exception as e:
        log.exception("Pipeline error")
        return ChatResponse(
            reply="Sorry, something went wrong. Please try again.",
            citations=[],
            cache_hit=False,
            latency_ms=0,
        )

    return ChatResponse(
        reply=result.get("reply", ""),
        citations=result.get("citations", []),
        cache_hit=result.get("cache_hit", False),
        latency_ms=result.get("latency_ms", 0),
    )
