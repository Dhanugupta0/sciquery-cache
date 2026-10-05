"""LangGraph pipeline — the request flow from classify to finalize."""

import json
import logging
import re
import time
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from app.llm import (
    get_llm, get_numeric_llm, get_fallback_llm, ANSWER_SYSTEM, ANSWER_HUMAN,
    NUMERIC_SYSTEM,
    REWRITE_SYSTEM, REWRITE_HUMAN,
    STYLE_SYSTEM, STYLE_HUMAN,
)
from app.retriever import BookRetriever
from app.sessions import SessionStore
from app.cache.store import CacheStore
from app.cache.normalize import normalize, contextual_key, extract_numbers, STOPWORDS
from app.cache.policy import should_cache, is_style_request, is_numerical_question, matches_style_pattern
from app.config import SCOPE_THRESHOLD, SCOPE_MIN_THRESHOLD

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


_GREETINGS = {"hi", "hello", "hii", "hey", "thanks"}


def classify_message(state: PipelineState) -> PipelineState:
    """Rule-based classifier: GREETING / STYLE / FOLLOW_UP / STANDALONE. No LLM call."""
    msg = state["message"].strip()

    # Check for greeting first
    clean_greeting = re.sub(r"[^\w\s]", "", msg.lower()).strip()
    if clean_greeting in _GREETINGS:
        state["msg_type"] = "GREETING"
        greeting_text = "Hi! Ask me any doubt from the NCERT Class 10 Science textbook."
        state["answer"] = greeting_text
        state["reply"] = greeting_text
        state["citations"] = []
        state["pages"] = []
        state["in_scope"] = True
        state["cache_hit"] = False
        log.info(f"Classified as GREETING: '{msg}'")
        return state

    session = _sessions.get(state.get("session_id", ""))
    has_prev_answer = bool(session and session.get("last_answer"))
    has_prev_question = bool(session and session.get("last_standalone_question"))

    # Check for style request first
    if matches_style_pattern(msg):
        if not has_prev_answer:
            reply_text = "Ask me a question first, then I can explain it more simply."
            state["msg_type"] = "NO_CONTEXT_STYLE"
            state["answer"] = reply_text
            state["reply"] = reply_text
            state["citations"] = []
            state["pages"] = []
            state["in_scope"] = True
            state["cache_hit"] = False
            state["cache_reason"] = "skip: style without previous answer"
            log.info(f"Style request without previous answer: '{msg}'")
            return state
        state["msg_type"] = "STYLE"
        log.info("Classified as STYLE")
        return state

    # Check for follow-up signals
    if is_followup_message(msg):
        if not has_prev_question:
            reply_text = "Which topic do you mean? Please ask a full question first, then I can follow up."
            state["msg_type"] = "NO_CONTEXT_FOLLOWUP"
            state["answer"] = reply_text
            state["reply"] = reply_text
            state["citations"] = []
            state["pages"] = []
            state["in_scope"] = True
            state["cache_hit"] = False
            state["cache_reason"] = "skip: follow-up without previous question"
            log.info(f"Follow-up without previous question: '{msg}'")
            return state
        state["msg_type"] = "FOLLOW_UP"
        log.info("Classified as FOLLOW_UP")
        return state

    state["msg_type"] = "STANDALONE"
    log.info("Classified as STANDALONE")
    return state


# ---- cache lookup ----

def cache_lookup(state: PipelineState) -> PipelineState:
    """Check cache. GREETING, STYLE, and numerical questions always skip. FOLLOW_UP uses contextual exact only."""
    msg_type = state["msg_type"]

    if msg_type in ("GREETING", "NO_CONTEXT_FOLLOWUP", "NO_CONTEXT_STYLE"):
        state["cache_hit"] = False
        state["cache_reason"] = f"skip: {msg_type.lower()}"
        log.info(f"Cache: skip ({msg_type.lower()})")
        return state

    if msg_type == "STYLE":
        state["cache_hit"] = False
        state["cache_reason"] = "skip: style request"
        log.info("Cache: skip (style)")
        return state

    if is_numerical_question(state["message"]):
        state["cache_hit"] = False
        state["cache_reason"] = "skip: numerical question"
        log.info("Cache: skip (numerical)")
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


def _invoke_llm(messages: list[dict], llm_client=None, max_retries: int = 1):
    """Invoke the LLM, retrying on 429 rate limit or failing over to LLM_FALLBACK_MODEL on 429/404."""
    from app.config import get_api_key, LLM_BASE_URL, LLM_MODEL

    client = llm_client if llm_client is not None else _llm

    # If client was initialized with a dummy key but a real key is now present, refresh
    current_key = get_api_key()
    if current_key and current_key != "dummy-key-pending-secrets":
        client_key = getattr(client, "openai_api_key", None)
        client_key_val = client_key.get_secret_value() if hasattr(client_key, "get_secret_value") else str(client_key or "")
        if client_key_val == "dummy-key-pending-secrets":
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                base_url=getattr(client, "base_url", LLM_BASE_URL),
                api_key=current_key,
                model=getattr(client, "model_name", LLM_MODEL),
                temperature=getattr(client, "temperature", 0),
                max_tokens=getattr(client, "max_tokens", 800),
                max_retries=0,
            )

    for attempt in range(max_retries + 1):
        try:
            return client.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            status_code = getattr(e, "status_code", None)
            is_429 = "429" in err_str or "rate limit" in err_str or status_code == 429
            is_404 = "404" in err_str or "not found" in err_str or status_code == 404
            is_401 = "401" in err_str or "invalid api key" in err_str or "invalid_api_key" in err_str or status_code == 401

            if is_401:
                log.error(f"LLM authentication (401) error: {e}")
                raise RuntimeError(
                    "Invalid or missing Groq API Key. Please provide a valid LLM_API_KEY in Streamlit Secrets or sidebar."
                ) from e

            # Failover to fallback model on 429 or 404
            if (is_429 or is_404) and _fallback_llm is not None and client != _fallback_llm:
                log.warning(f"Error ({'429' if is_429 else '404'}) on primary model, failing over to LLM_FALLBACK_MODEL...")
                client = _fallback_llm
                try:
                    return client.invoke(messages)
                except Exception as fb_err:
                    log.error(f"Fallback model also failed: {fb_err}")
                    raise

            if is_429 and attempt < max_retries:
                log.warning("Rate limit (429) hit, retrying in 1s...")
                time.sleep(1.0)
                continue
            raise


def _parse_llm_json(raw: str, docs: list[dict]) -> tuple[str, bool, list[str]]:
    """Parse JSON reply from the LLM, extracting (answer, in_scope, used_chapters)."""
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        data = json.loads(raw)
        in_scope = data.get("in_scope", True)
        if not in_scope:
            return STANDARD_DECLINE_MESSAGE, False, []
        return data.get("answer", raw), True, data.get("used_chapters", [])
    except json.JSONDecodeError:
        log.warning(f"LLM returned non-JSON: {raw[:100]}")
        return raw, True, [docs[0]["chapter"]] if docs else []


def is_incomplete_answer(text: str) -> bool:
    """Check if the answer says the context does not cover something,
    or mentions a figure or table that is not included."""
    if not text:
        return False
    lower = text.lower()

    # 1. Context does not cover / mention / provide / contain / include
    context_patterns = [
        r"\b(context|provided context|text)\b.*?\b(does not|doesn't|cannot|can't|not)\b.*?\b(cover|mention|provide|contain|include|state|have|give|explain|specify)\b",
        r"\bnot\s+(covered|mentioned|provided|contained|included|given)\s+in\s+(the\s+)?context\b",
        r"\b(not enough|insufficient|lacks|missing)\s+(information|context|details)\b",
        r"\bprovided\s+context\s+(does\s+not|doesn't)\b",
    ]
    for pat in context_patterns:
        if re.search(pat, lower):
            return True

    # 2. Mentions a figure or table that is not included / missing
    fig_patterns = [
        r"\b(figure|fig\.|table|diagram)\b.*?\b(not\s+(included|provided|shown|available|given|present)|is\s+missing|missing)\b",
        r"\b(not\s+(included|provided|shown|available|given|present)|missing)\b.*?\b(figure|fig\.|table|diagram)\b",
    ]
    for pat in fig_patterns:
        if re.search(pat, lower):
            return True

    return False


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
    """Borderline band:
    - score < SCOPE_MIN_THRESHOLD (0.58): declines without calling the LLM.
    - 0.58 to 0.68: lets the LLM decide using in_scope.
    - 0.68 and above: answers.
    """
    score = state["best_score"]
    if score < SCOPE_MIN_THRESHOLD:
        state["in_scope"] = False
        state["answer"] = STANDARD_DECLINE_MESSAGE
        state["citations"] = []
        state["used_chapters"] = []
        log.info(f"Out of scope: best score {score:.4f} < {SCOPE_MIN_THRESHOLD} (no LLM call)")
    else:
        state["in_scope"] = True
        log.info(f"Scope check passed: best score {score:.4f} >= {SCOPE_MIN_THRESHOLD}")
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
    question = state.get("standalone_question") or state["message"]

    is_num = is_numerical_question(question)
    system_tmpl = NUMERIC_SYSTEM if is_num else ANSWER_SYSTEM
    llm_client = _numeric_llm if (is_num and _numeric_llm is not None) else _llm

    def _call_model(current_docs):
        ctx = "\n\n---\n\n".join(
            f"[Chapter: {d['chapter']}, Page: {d['page']}]\n{d['text']}"
            for d in current_docs
        )
        msgs = [
            {"role": "system", "content": system_tmpl.format(context=ctx)},
            {"role": "user", "content": ANSWER_HUMAN.format(question=question)},
        ]
        resp = _invoke_llm(msgs, llm_client=llm_client)
        return _parse_llm_json(resp.content.strip(), current_docs)

    ans, in_scope, used_ch = _call_model(docs)

    # Check for incomplete answers: retry once with 8 chunks
    if in_scope and is_incomplete_answer(ans):
        log.info("Answer flagged as incomplete (attempt 1); retrying with 8 chunks...")
        docs8 = _retriever.search(question, top_k=8)
        state["retrieved_docs"] = docs8
        ans, in_scope, used_ch = _call_model(docs8)
        if not in_scope or is_incomplete_answer(ans):
            log.info("Answer still incomplete after 8 chunks; declining.")
            in_scope = False
            ans = STANDARD_DECLINE_MESSAGE
            used_ch = []
    elif not in_scope:
        # Borderline band rule:
        # 0.58 to 0.68 lets the LLM decide using in_scope.
        # 0.68 and above answers.
        if state.get("best_score", 0.0) >= SCOPE_THRESHOLD:
            in_scope = True
        else:
            ans = STANDARD_DECLINE_MESSAGE
            used_ch = []

    state["in_scope"] = in_scope
    state["answer"] = ans
    state["used_chapters"] = used_ch

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

    # Strip any line starting with "Chapter:" from replies
    if state.get("reply"):
        lines = state["reply"].splitlines()
        cleaned_lines = [l for l in lines if not l.strip().lower().startswith("chapter:")]
        state["reply"] = "\n".join(cleaned_lines).strip()

    elapsed = time.perf_counter() - state["start_time"]
    state["latency_ms"] = round(elapsed * 1000, 1)

    # Update session
    sid = state["session_id"]
    session = _sessions.get(sid)
    if session is not None:
        _sessions.add_turn(sid, "user", state["message"])
        _sessions.add_turn(sid, "assistant", state["reply"])
        if state["msg_type"] not in ("GREETING", "NO_CONTEXT_FOLLOWUP", "NO_CONTEXT_STYLE"):
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
    if state["msg_type"] in ("GREETING", "NO_CONTEXT_FOLLOWUP", "NO_CONTEXT_STYLE"):
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
_numeric_llm = None
_fallback_llm = None
_retriever = None
_cache = None
_sessions = None
_graph = None


def init_pipeline(sessions: SessionStore | None = None):
    """Initialize singletons and compile the LangGraph."""
    global _llm, _numeric_llm, _fallback_llm, _retriever, _cache, _sessions, _graph

    log.info("Initializing pipeline...")
    _llm = get_llm()
    _numeric_llm = get_numeric_llm()
    _fallback_llm = get_fallback_llm()
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
