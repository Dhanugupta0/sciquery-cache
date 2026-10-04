"""Safety guards that prevent serving the wrong cached answer.

Each guard takes two questions and returns (passed: bool, reason: str).
ALL guards must pass for a cache hit. If ANY fails → MISS.
"""

import re
from app.cache.normalize import extract_numbers, extract_key_terms, detect_question_type
from app.config import CACHE_JACCARD_THRESHOLD


# ---- contrast term groups ----
# If a question mentions "concave", a cached answer about "convex" is WRONG.
CONTRAST_GROUPS = [
    {"concave", "convex"},
    {"real", "virtual"},
    {"converging", "diverging"},
    {"acid", "base", "alkali"},
    {"acidic", "basic", "alkaline"},
    {"oxidation", "reduction"},
    {"endothermic", "exothermic"},
    {"series", "parallel"},
    {"ionic", "covalent"},
    {"metal", "non-metal", "nonmetal"},
    {"metallic", "non-metallic"},
    {"saturated", "unsaturated"},
    {"alkane", "alkene", "alkyne"},
    {"arteries", "veins", "artery", "vein"},
    {"aerobic", "anaerobic"},
    {"dominant", "recessive"},
    {"myopia", "hypermetropia", "near-sightedness", "far-sightedness"},
    {"step-up", "step-down"},
    {"ac", "dc"},
    {"sexual", "asexual"},
    {"autotroph", "heterotroph", "autotrophic", "heterotrophic"},
    {"biodegradable", "non-biodegradable"},
    {"sensory", "motor"},
    {"reflection", "refraction"},
    {"physical", "chemical"},
    {"current", "voltage", "resistance"},
    {"positive", "negative"},
    {"anode", "cathode"},
    {"mitosis", "meiosis"},
    {"ductile", "brittle"},
    {"malleable", "non-malleable"},
    {"conductor", "insulator"},
    {"producer", "consumer", "decomposer"},
    {"food chain", "food web"},
    {"concave mirror", "convex mirror"},
    {"concave lens", "convex lens"},
    {"ozone", "oxygen"},
    {"fission", "fusion"},
    {"binary fission", "multiple fission"},
    {"budding", "fragmentation", "regeneration", "spore formation"},
    {"unisexual", "bisexual"},
    {"pollination", "fertilisation", "fertilization"},
    {"self pollination", "cross pollination"},
    {"homologous", "analogous"},
    {"tropic", "nastic"},
    {"sympathetic", "parasympathetic"},
    {"cerebrum", "cerebellum", "medulla"},
]


def _mentioned_members(text_lower: str, group: set[str]) -> set[str]:
    """Which members of a contrast group appear in the text?"""
    found = set()
    for member in group:
        # Use word boundary to avoid partial matches
        if re.search(r"\b" + re.escape(member) + r"\b", text_lower):
            found.add(member)
    return found


# ---- guards ----

def number_guard(q1: str, q2: str) -> tuple[bool, str]:
    """Numbers + units in both questions must be identical."""
    nums1 = extract_numbers(q1)
    nums2 = extract_numbers(q2)
    if nums1 != nums2:
        return False, f"number guard: {nums1} vs {nums2}"
    return True, "number guard: passed"


def contrast_guard(q1: str, q2: str) -> tuple[bool, str]:
    """For each contrast group, mentioned members must be the same."""
    q1_lower = q1.lower()
    q2_lower = q2.lower()
    for group in CONTRAST_GROUPS:
        m1 = _mentioned_members(q1_lower, group)
        m2 = _mentioned_members(q2_lower, group)
        if m1 != m2:
            return False, f"contrast guard: {m1} vs {m2} in group {group}"
    return True, "contrast guard: passed"


def key_term_guard(q1: str, q2: str) -> tuple[bool, str]:
    """Content words must overlap strongly (Jaccard >= threshold)."""
    terms1 = extract_key_terms(q1)
    terms2 = extract_key_terms(q2)
    if not terms1 and not terms2:
        return True, "key-term guard: both empty"
    if not terms1 or not terms2:
        return False, f"key-term guard: one side empty ({terms1} vs {terms2})"
    jaccard = len(terms1 & terms2) / len(terms1 | terms2)
    passed = jaccard >= CACHE_JACCARD_THRESHOLD
    reason = f"key-term guard: Jaccard={jaccard:.2f} ({'passed' if passed else 'FAILED'})"
    return passed, reason


def question_type_guard(q1: str, q2: str) -> tuple[bool, str]:
    """Both questions must have the same detected type."""
    t1 = detect_question_type(q1)
    t2 = detect_question_type(q2)
    if t1 != t2:
        return False, f"question-type guard: {t1} vs {t2}"
    return True, f"question-type guard: both {t1}"


def run_all_guards(q1: str, q2: str) -> tuple[bool, str]:
    """Run all four guards. Returns (all_passed, combined_reason)."""
    guards = [number_guard, contrast_guard, key_term_guard, question_type_guard]
    reasons = []
    for guard in guards:
        passed, reason = guard(q1, q2)
        reasons.append(reason)
        if not passed:
            return False, reason  # fail fast on first failure
    return True, "; ".join(reasons)
