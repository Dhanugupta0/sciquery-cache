"""LangGraph pipeline — the request flow from classify to finalize."""

import json
import logging
import re
import time
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from app.llm import (
    get_llm, ANSWER_SYSTEM, ANSWER_HUMAN,
    REWRITE_SYSTEM, REWRITE_HUMAN,
    STYLE_SYSTEM, STYLE_HUMAN,
)
from app.retriever import BookRetriever
from app.sessions import SessionStore
from app.cache.store import CacheStore
from app.cache.normalize import normalize, contextual_key, extract_numbers, STOPWORDS
from app.cache.policy import should_cache, is_style_request
from app.config import SCOPE_THRESHOLD

log = logging.getLogger(__name__)


# ---- state ----

class PipelineState(TypedDict, total=False):
    # Input
    session_id: str
    message: str
    # Computed
    msg_type: str           # STYLE, FOLLOW_UP, STANDALONE
    standalone_question: str
    # Cache
    cache_hit: bool
    cache_reason: str
    cached_answer: str
    cached_citations: list[str]
    # Retrieval
    retrieved_docs: list[dict]
    best_score: float
    # Generation
    in_scope: bool
    answer: str
    used_chapters: list[str]
    citations: list[str]
    pages: list[int]
    # Output
    reply: str
    latency_ms: float
    start_time: float


# ---- standard decline message ----

STANDARD_DECLINE_MESSAGE = (
    "I'm sorry, this question doesn't seem to be covered in the "
    "NCERT Class 10 Science textbook. I can only help with topics "
    "from that book. Could you ask something from the textbook?"
)


# ---- classify ----

_FOLLOWUP_PRONOUNS = {"it", "its", "this", "that", "these", "those", "they", "them", "their"}
_FOLLOWUP_STARTS = ("what about", "how about", "what else", "why so")


def is_followup_message(msg: str) -> bool:
    """A message is FOLLOW_UP only if:
    (a) it starts with "what about", "and", "how about", "what else" or "why so", OR
    (b) it has a pronoun (it, its, this, that, these, those, they, them, their)
        AND two or fewer topic words after removing stopwords, OR
    (c) it has zero topic words.
    """
    raw_lower = msg.strip().lower()

    # (a) starts with "what about", "and", "how about", "what else" or "why so"
    for prefix in _FOLLOWUP_STARTS:
        if raw_lower.startswith(prefix):
            return True
    if raw_lower == "and" or raw_lower.startswith("and ") or raw_lower.startswith("and,"):
        return True

    words = re.findall(r"\b\w+\b", raw_lower)
    has_pronoun = any(w in _FOLLOWUP_PRONOUNS for w in words)
    topic_words = [w for w in words if w not in STOPWORDS]

    # (b) has a pronoun AND two or fewer topic words after removing stopwords
    if has_pronoun and len(topic_words) <= 2:
        return True

    # (c) zero topic words
    if len(topic_words) == 0:
        return True

    return False


def classify_message(state: PipelineState) -> PipelineState:
    """Rule-based classifier: STYLE / FOLLOW_UP / STANDALONE. No LLM call."""
    msg = state["message"].strip()

    session = _sessions.get(state.get("session_id", ""))
    has_prev_answer = bool(session and session.get("last_answer"))

    # Check for style request first
    if is_style_request(msg, has_previous_answer=has_prev_answer):
        state["msg_type"] = "STYLE"
        log.info("Classified as STYLE")
        return state

    # Check for follow-up signals
    if is_followup_message(msg):
        # Only classify as follow-up if there's conversation history or standalone test
        if session is None or session.get("last_standalone_question"):
            state["msg_type"] = "FOLLOW_UP"
            log.info("Classified as FOLLOW_UP")
            return state

    state["msg_type"] = "STANDALONE"
    log.info("Classified as STANDALONE")
    return state


# ---- cache lookup ----

def cache_lookup(state: PipelineState) -> PipelineState:
    """Check cache. STYLE always skips. FOLLOW_UP uses contextual exact only."""
    msg_type = state["msg_type"]

    if msg_type == "STYLE":
        state["cache_hit"] = False
        state["cache_reason"] = "skip: style request"
        log.info("Cache: skip (style)")
        return state

    session = _sessions.get(state["session_id"])
    prev_standalone = session.get("last_standalone_question") if session else None

    if msg_type == "FOLLOW_UP":
        entry, reason = _cache.lookup(
            state["message"], prev_standalone=prev_standalone, is_followup=True
        )
    else:
        entry, reason = _cache.lookup(state["message"])

    if entry:
        state["cache_hit"] = True
        state["cache_reason"] = reason
        state["cached_answer"] = entry["answer"]
        state["cached_citations"] = entry["citations"]
        state["cached_pages"] = entry.get("pages", [])
        log.info(f"Cache: {reason}")
    else:
        state["cache_hit"] = False
        state["cache_reason"] = reason
        log.info(f"Cache: {reason}")

    return state


def _invoke_llm(messages: list[dict], max_retries: int = 1):
    """Invoke the LLM, retrying once on 429 rate limit."""
    for attempt in range(max_retries + 1):
        try:
            return _llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            is_429 = "429" in err_str or "rate limit" in err_str or getattr(e, "status_code", None) == 429
            if is_429 and attempt < max_retries:
                log.warning("Rate limit (429) hit, retrying in 1s...")
                time.sleep(1.0)
                continue
            raise


# ---- rewrite (follow-up only) ----

def rewrite_question(state: PipelineState) -> PipelineState:
    """Rewrite a follow-up into a standalone question using conversation history."""
    session = _sessions.get(state["session_id"])
    history = session.get("history", []) if session else []
    # Last 3 turns
    recent = history[-6:]  # 3 pairs of user+assistant

    history_text = "\n".join(
        f"{'Student' if h['role'] == 'user' else 'Tutor'}: {h['content']}"
        for h in recent
    )

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM.format(history=history_text)},
        {"role": "user", "content": REWRITE_HUMAN.format(message=state["message"])},
    ]

    response = _invoke_llm(messages)
    standalone = response.content.strip().strip('"')
    state["standalone_question"] = standalone
    log.info(f"Rewritten: '{state['message']}' → '{standalone}'")
    return state


# ---- retrieve ----

def retrieve(state: PipelineState) -> PipelineState:
    """Get top chunks from the book index."""
    query = state.get("standalone_question") or state["message"]
    results = _retriever.search(query)
    state["retrieved_docs"] = results
    state["best_score"] = results[0]["score"] if results else 0.0
    log.info(f"Retrieved {len(results)} docs, best score: {state['best_score']:.4f}")
    return state


# ---- scope check ----

def scope_check(state: PipelineState) -> PipelineState:
    """If best retrieval score is too low, decline without calling LLM."""
    if state["best_score"] < SCOPE_THRESHOLD:
        state["in_scope"] = False
        state["answer"] = STANDARD_DECLINE_MESSAGE
        state["citations"] = []
        state["used_chapters"] = []
        log.info(f"Out of scope: best score {state['best_score']:.4f} < {SCOPE_THRESHOLD}")
    else:
        state["in_scope"] = True
    return state


# ---- generate ----

def generate(state: PipelineState) -> PipelineState:
    """Call the LLM to generate an answer from retrieved context."""
    msg_type = state["msg_type"]

    if msg_type == "STYLE":
        return _generate_style(state)

    if not state.get("in_scope", True):
        return state  # already declined in scope_check

    docs = state["retrieved_docs"]
    context = "\n\n---\n\n".join(
        f"[Chapter: {d['chapter']}, Page: {d['page']}]\n{d['text']}"
        for d in docs
    )
    question = state.get("standalone_question") or state["message"]

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM.format(context=context)},
        {"role": "user", "content": ANSWER_HUMAN.format(question=question)},
    ]

    response = _invoke_llm(messages)
    raw = response.content.strip()

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        in_scope = data.get("in_scope", True)
        state["in_scope"] = in_scope
        if not in_scope:
            state["answer"] = STANDARD_DECLINE_MESSAGE
            state["used_chapters"] = []
        else:
            state["answer"] = data.get("answer", raw)
            state["used_chapters"] = data.get("used_chapters", [])
    except json.JSONDecodeError:
        log.warning(f"LLM returned non-JSON: {raw[:100]}")
        state["in_scope"] = True
        state["answer"] = raw
        state["used_chapters"] = [docs[0]["chapter"]] if docs else []

    return state


def _generate_style(state: PipelineState) -> PipelineState:
    """Handle STYLE request: rewrite previous answer per style instruction."""
    session = _sessions.get(state["session_id"])
    prev_answer = session.get("last_answer", "") if session else ""
    prev_citations = session.get("last_citations", []) if session else []

    if not prev_answer:
        state["answer"] = "I don't have a previous answer to restyle. Could you ask a question first?"
        state["citations"] = []
        state["in_scope"] = True
        return state

    messages = [
        {"role": "system", "content": STYLE_SYSTEM.format(
            previous_answer=prev_answer,
            citations=", ".join(prev_citations),
        )},
        {"role": "user", "content": STYLE_HUMAN.format(request=state["message"])},
    ]

    response = _invoke_llm(messages)
    state["answer"] = response.content.strip()
    state["citations"] = prev_citations
    state["used_chapters"] = prev_citations
    state["in_scope"] = True
    return state


# ---- validate ----

def validate(state: PipelineState) -> PipelineState:
    """Ensure used_chapters are actually from retrieved docs.
    Citations = chapter names only. Pages = page numbers from matching docs."""
    if state["msg_type"] == "STYLE":
        return state

    # On out-of-scope decline, return standard decline message and empty citations/pages
    if not state.get("in_scope", True):
        state["in_scope"] = False
        state["answer"] = STANDARD_DECLINE_MESSAGE
        state["citations"] = []
        state["pages"] = []
        return state

    docs = state.get("retrieved_docs", [])
    if docs:
        chapter_scores = {}
        for d in docs:
            ch = d["chapter"]
            chapter_scores[ch] = chapter_scores.get(ch, 0.0) + d.get("score", 0.0)
        best_chapter = max(chapter_scores.items(), key=lambda x: x[1])[0]
        valid = [best_chapter]
        pages = sorted({d["page"] for d in docs if d["chapter"] == best_chapter})
    else:
        valid = []
        pages = []

    state["citations"] = valid
    state["pages"] = pages
    return state


# ---- maybe store ----

def maybe_store(state: PipelineState) -> PipelineState:
    """Store in cache if policy allows."""
    if state.get("cache_hit"):
        return state  # don't re-store a hit

    msg_type = state["msg_type"]
    question = state.get("standalone_question") or state["message"]
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    in_scope = state.get("in_scope", False)

    store_ok, reason = should_cache(
        question, answer, citations, in_scope, is_style=(msg_type == "STYLE")
    )

    if store_ok:
        ctx_key = None
        if msg_type == "FOLLOW_UP":
            session = _sessions.get(state["session_id"])
            prev = session.get("last_standalone_question") if session else None
            if prev:
                ctx_key = contextual_key(prev, state["message"])

        _cache.store(question, answer, citations,
                     pages=state.get("pages", []), ctx_key=ctx_key)
        log.info(f"Stored in cache: {question[:60]}...")
    else:
        log.info(f"Not cached: {reason}")

    return state


# ---- finalize ----

def finalize(state: PipelineState) -> PipelineState:
    """Build final response and record latency."""
    if state.get("cache_hit"):
        state["reply"] = state["cached_answer"]
        state["citations"] = state.get("cached_citations", [])
        state["pages"] = state.get("cached_pages", [])
    else:
        if not state.get("in_scope", True):
            state["reply"] = STANDARD_DECLINE_MESSAGE
            state["citations"] = []
            state["pages"] = []
        else:
            state["reply"] = state.get("answer", "Something went wrong. Please try again.")
        # citations and pages already set by validate()

    elapsed = time.perf_counter() - state["start_time"]
    state["latency_ms"] = round(elapsed * 1000, 1)

    # Update session
    sid = state["session_id"]
    session = _sessions.get(sid)
    if session is not None:
        _sessions.add_turn(sid, "user", state["message"])
        _sessions.add_turn(sid, "assistant", state["reply"])
        _sessions.set_last_answer(sid, state["reply"], state.get("citations", []))

        # Track standalone question for follow-up context
        if state["msg_type"] == "STANDALONE":
            _sessions.set_last_standalone(sid, state["message"])
        elif state["msg_type"] == "FOLLOW_UP" and state.get("standalone_question"):
            _sessions.set_last_standalone(sid, state["standalone_question"])

    return state


# ---- routing ----

def route_after_cache(state: PipelineState) -> str:
    """Route based on cache result."""
    if state.get("cache_hit"):
        return "finalize"
    if state["msg_type"] == "STYLE":
        return "generate"
    if state["msg_type"] == "FOLLOW_UP":
        return "rewrite_question"
    return "retrieve"


def route_after_scope(state: PipelineState) -> str:
    """Skip LLM if out of scope."""
    if not state.get("in_scope", True):
        return "validate"
    return "generate"


# ---- build graph ----

# Module-level singletons, initialized via init_pipeline()
_llm = None
_retriever = None
_cache = None
_sessions = None
_graph = None


def init_pipeline(sessions: SessionStore | None = None):
    """Initialize singletons and compile the LangGraph."""
    global _llm, _retriever, _cache, _sessions, _graph

    log.info("Initializing pipeline...")
    _llm = get_llm()
    _retriever = BookRetriever()
    _cache = CacheStore()
    _sessions = sessions or SessionStore()

    builder = StateGraph(PipelineState)

    builder.add_node("classify_message", classify_message)
    builder.add_node("cache_lookup", cache_lookup)
    builder.add_node("rewrite_question", rewrite_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("scope_check", scope_check)
    builder.add_node("generate", generate)
    builder.add_node("validate", validate)
    builder.add_node("maybe_store", maybe_store)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("classify_message")
    builder.add_edge("classify_message", "cache_lookup")

    builder.add_conditional_edges("cache_lookup", route_after_cache, {
        "finalize": "finalize",
        "generate": "generate",          # STYLE → skip retrieval
        "rewrite_question": "rewrite_question",
        "retrieve": "retrieve",
    })

    builder.add_edge("rewrite_question", "retrieve")
    builder.add_edge("retrieve", "scope_check")
    builder.add_conditional_edges("scope_check", route_after_scope, {
        "validate": "validate",
        "generate": "generate",
    })
    builder.add_edge("generate", "validate")
    builder.add_edge("validate", "maybe_store")
    builder.add_edge("maybe_store", "finalize")
    builder.add_edge("finalize", END)

    _graph = builder.compile()
    log.info("Pipeline ready")
    return _sessions


def run_pipeline(session_id: str, message: str) -> dict:
    """Run the full pipeline. Returns the final state dict."""
    if _graph is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")

    state: PipelineState = {
        "session_id": session_id,
        "message": message,
        "start_time": time.perf_counter(),
        "cache_hit": False,
    }

    result = _graph.invoke(state)
    return result


def get_sessions() -> SessionStore:
    return _sessions


def get_cache() -> CacheStore:
    return _cache
