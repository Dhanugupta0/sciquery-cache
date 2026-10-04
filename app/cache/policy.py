"""Cache policy — decides what to cache and what to never cache."""

from app.cache.normalize import extract_numbers


# ---- patterns we never cache ----

_GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "ok", "okay"}

_STYLE_WORDS = {
    "simpler", "shorter", "longer", "again", "points", "bullet",
    "example", "examples", "detail", "briefly", "simple",
}


def is_greeting(text: str) -> bool:
    words = set(text.lower().split())
    return len(words) <= 4 and bool(words & _GREETING_WORDS)


def is_style_request(text: str) -> bool:
    """Check if the message is purely a style/format change request."""
    lower = text.lower().strip()
    # Direct style phrases
    style_phrases = [
        "explain it more simply", "explain more simply", "make it simpler",
        "in points", "in bullet points", "give an example", "give examples",
        "say that again", "repeat", "shorter", "make it shorter",
        "elaborate", "more detail", "explain in detail",
        "summarize", "summarise",
    ]
    for phrase in style_phrases:
        if phrase in lower:
            return True
    # Very short + style word
    words = lower.split()
    if len(words) <= 5 and any(w in _STYLE_WORDS for w in words):
        return True
    return False


def should_cache(question: str, answer: str, citations: list[str],
                 in_scope: bool, is_style: bool) -> tuple[bool, str]:
    """Decide if this answer should be stored in the cache.
    Returns (should_store, reason)."""

    if is_style:
        return False, "never cache: style request"

    if not in_scope:
        return False, "never cache: out-of-scope decline"

    if not answer or not answer.strip():
        return False, "never cache: empty answer"

    if not citations:
        return False, "never cache: no citations"

    if is_greeting(question):
        return False, "never cache: greeting"

    return True, "cacheable"


def should_serve_semantic(question: str, has_numbers: bool) -> tuple[bool, str]:
    """Pre-check before semantic cache lookup.
    If the question has numbers, only exact match is allowed."""
    if has_numbers:
        return False, "skip semantic: question contains numbers"
    if is_greeting(question):
        return False, "skip semantic: greeting"
    return True, "semantic lookup allowed"
