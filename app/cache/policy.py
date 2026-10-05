"""Cache policy — decides what to cache and what to never cache."""

from app.cache.normalize import normalize, STOPWORDS, extract_numbers


# ---- patterns we never cache ----

_GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "ok", "okay"}

STYLE_WORDS = {
    "simpler", "simply", "simple", "shorter", "short", "longer", "long",
    "again", "repeat", "points", "bullet", "bullets",
    "example", "examples", "detail", "details", "briefly", "brief",
    "summarize", "summarise", "summary", "elaborate", "more", "less",
    "easy", "easier",
}


def is_greeting(text: str) -> bool:
    words = set(text.lower().split())
    return len(words) <= 4 and bool(words & _GREETING_WORDS)


def is_style_request(text: str, has_previous_answer: bool = True) -> bool:
    """A message is STYLE only if there is a previous answer AND no real topic words remain."""
    if not has_previous_answer:
        return False

    words = normalize(text).split()
    if not words:
        return False

    # Must contain at least one style word
    if not any(w in STYLE_WORDS for w in words):
        return False

    # Must have no real topic words after removing stopwords and style words
    topic_words = [w for w in words if w not in STOPWORDS and w not in STYLE_WORDS]
    return len(topic_words) == 0


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
