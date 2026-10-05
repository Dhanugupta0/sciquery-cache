"""Text normalization, number extraction, key-term extraction, question-type detection."""

import re
import hashlib


# ---- normalize ----

def normalize(text: str) -> str:
    """Lowercase, strip punctuation (keep hyphens between words), collapse spaces."""
    text = text.lower()
    # Keep hyphens between word chars (e.g. "non-metal"), remove other punctuation
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_text(text: str) -> str:
    """SHA-256 hash of normalized text for exact-match lookup."""
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def contextual_key(prev_standalone: str, followup: str) -> str:
    """Hash of (previous standalone + follow-up) for contextual exact match."""
    combined = normalize(prev_standalone or "") + "|||" + normalize(followup)
    return hashlib.sha256(combined.encode()).hexdigest()


# ---- number extraction ----

_NUMBER_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(f\d*|cm|mm|m|km|kg|g|mg|ohm|ω|v|a|w|hz|"
    r"dioptre|d|mol|l|ml|s|sec|min|°c|°f|k|n|pa|j|cal|ev|"
    r"%|percent)?\b",
    re.IGNORECASE,
)

def extract_numbers(text: str) -> set[str]:
    """Extract number+unit pairs. Returns e.g. {'20cm', '30', '15ohm', '2f', '2f1'}."""
    text_lower = text.lower()
    results = set()
    for m in _NUMBER_PATTERN.finditer(text_lower):
        num = m.group(1)
        unit = (m.group(2) or "").strip()
        results.add(f"{num}{unit}")
    return results


def extract_symbols(text: str) -> set[str]:
    """Extract tokens that contain a digit, or are a single letter (except 'a' and 'i')."""
    words = normalize(text).split()
    symbols = set()
    for w in words:
        if any(c.isdigit() for c in w):
            symbols.add(w)
        elif len(w) == 1 and w.isalpha() and w not in ("a", "i"):
            symbols.add(w)
    return symbols



# ---- stopwords and synonyms ----

STOPWORDS = {
    "what", "is", "are", "was", "were", "does", "do", "did", "the", "a", "an",
    "in", "of", "for", "to", "and", "or", "it", "its", "this", "that", "with",
    "by", "on", "at", "from", "as", "be", "been", "being", "have", "has", "had",
    "will", "would", "can", "could", "shall", "should", "may", "might", "about",
    "how", "why", "when", "where", "which", "who", "whom", "whose", "not", "no",
    "so", "if", "then", "than", "also", "very", "just", "but", "nor", "yet",
    "explain", "define", "describe", "discuss", "state", "mention", "write",
    "give", "name", "list", "draw", "between", "difference", "mean", "meaning",
    "called", "known", "important", "good", "used", "using", "use",
    "we", "our", "us", "they", "them", "their", "you", "your",
    "get", "gets", "got", "make", "makes", "made", "need", "needs",
    "tell", "some", "many", "more", "most", "much", "such", "these", "those",
    "each", "every", "any", "all", "both", "other", "another",
    "there", "here", "own", "same", "different", "new", "old",
    "well", "still", "even", "only", "like", "really", "actually",
    "please", "say",
}

# Synonym map: all map to a canonical form
SYNONYM_MAP = {
    "mean": "define", "meaning": "define", "definition": "define",
    "means": "define", "define": "define",
    # Singular/plural normalization
    "conductors": "conductor", "conductor": "conductor",
    "organisms": "organism", "organism": "organism",
    "reactions": "reaction", "reaction": "reaction",
    "equations": "equation", "equation": "equation",
    "circuits": "circuit", "circuit": "circuit",
    "mirrors": "mirror", "mirror": "mirror",
    "lenses": "lens", "lens": "lens",
    "images": "image", "image": "image",
    "metals": "metal", "metal": "metal",
    "acids": "acid", "acid": "acid",
    "bases": "base", "base": "base",
    "salts": "salt", "salt": "salt",
    "laws": "law", "law": "law",
    "properties": "property", "property": "property",
    # Verb form normalization
    "conduct": "conductor", "conducting": "conductor",
    "balance": "balance", "balancing": "balance", "balanced": "balance",
    "living": "life", "alive": "life",
    "formed": "form", "forming": "form", "forms": "form",
}


def extract_key_terms(text: str) -> set[str]:
    """Extract content words after removing stopwords and applying synonyms."""
    words = normalize(text).split()
    terms = set()
    for w in words:
        if w in STOPWORDS:
            continue
        # Apply synonym mapping
        w = SYNONYM_MAP.get(w, w)
        terms.add(w)
    return terms


# ---- question type detection ----

# Order matters: more specific types first, 'define' is a broad catch-all
_TYPE_RULES = [
    ("numerical", re.compile(r"\b(calculate|find|compute|numerical|formula|value of|if .* is \d)\b", re.I)),
    ("difference", re.compile(r"\b(difference|distinguish|compare|contrast|differentiate)\b", re.I)),
    ("diagram", re.compile(r"\b(diagram|draw|sketch|label|figure)\b", re.I)),
    ("list", re.compile(r"\b(list|enumerate|name the|mention the|write the names)\b", re.I)),
    ("derive", re.compile(r"\b(derive|derivation|prove|show that)\b", re.I)),
    # 'law' before 'define' so "What is Ohm's law?" matches 'law' not 'define'
    ("law", re.compile(r"\b(state .* law|law of|laws of|.+'s law|law$)\b", re.I)),
    ("why_how", re.compile(r"^(why|how)\b", re.I)),
    ("explain", re.compile(r"\b(explain|describe|discuss|elaborate)\b", re.I)),
    # 'define' last — broadest catch
    ("define", re.compile(r"\b(define|definition|what is|what are|what do you mean|what does .+ mean|state\b)\b", re.I)),
]


def detect_question_type(text: str) -> str:
    """Detect question type using keyword rules. Returns type string."""
    for qtype, pattern in _TYPE_RULES:
        if pattern.search(text):
            return qtype
    return "general"
