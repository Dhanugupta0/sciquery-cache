"""Cache decision tests — the required table + 16 extra pairs.

Tests that the cache makes the right HIT/MISS decision for paraphrases
and lookalikes, without calling an LLM for hits.
"""

import os
import sys
import time
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache.normalize import normalize, hash_text, extract_numbers, detect_question_type
from app.cache.guards import run_all_guards, number_guard, contrast_guard, key_term_guard, question_type_guard
from app.cache.policy import should_cache, is_style_request


# ============================================================
# REQUIRED test pairs from the assignment
# ============================================================

class TestRequiredPairs:
    """The 6 pairs specified in the assignment."""

    def test_refraction_paraphrase_hit(self):
        """'What is refraction?' vs 'What does refraction mean?' → HIT"""
        # Exact-match check: hashes differ (different wording)
        assert hash_text("What is refraction?") != hash_text("What does refraction mean?")
        # But guards should all pass (semantic hit)
        passed, reason = run_all_guards("What is refraction?", "What does refraction mean?")
        assert passed, f"Expected HIT but got MISS: {reason}"

    def test_concave_convex_miss(self):
        """'Image formed by a concave mirror' vs 'Image formed by a convex mirror' → MISS"""
        passed, reason = run_all_guards(
            "Image formed by a concave mirror",
            "Image formed by a convex mirror",
        )
        assert not passed, "Expected MISS (concave vs convex)"
        assert "contrast guard" in reason

    def test_different_numbers_miss(self):
        """'Focal length when R = 20 cm' vs 'Focal length when R = 30 cm' → MISS"""
        passed, reason = number_guard(
            "Focal length when R = 20 cm",
            "Focal length when R = 30 cm",
        )
        assert not passed, "Expected MISS (different numbers)"

    def test_same_numbers_exact_hit(self):
        """'Focal length when R = 20 cm' vs 'Focal length when R = 20 cm' → HIT (exact)"""
        h1 = hash_text("Focal length when R = 20 cm")
        h2 = hash_text("Focal length when R = 20 cm")
        assert h1 == h2, "Exact same question should have same hash"

    def test_followup_different_context_miss(self):
        """Same follow-up ('What about its laws?') in different contexts → MISS.
        This is enforced by contextual exact-match: different prev_standalone → different key."""
        from app.cache.normalize import contextual_key
        key_refraction = contextual_key("What is refraction?", "What about its laws?")
        key_reflection = contextual_key("What is reflection?", "What about its laws?")
        assert key_refraction != key_reflection, "Different contexts should produce different keys"

    def test_style_request_never_cached(self):
        """'Explain it more simply' → MISS (never cached)"""
        assert is_style_request("Explain it more simply")
        ok, reason = should_cache(
            "Explain it more simply", "...", ["Ch1"], in_scope=True, is_style=True
        )
        assert not ok, "Style requests should never be cached"


# ============================================================
# 8 should-HIT paraphrases
# ============================================================

class TestShouldHitParaphrases:
    """Paraphrases that should pass all guards (cache hit)."""

    @pytest.mark.parametrize("q1,q2", [
        # 1. Photosynthesis definition
        ("What is photosynthesis?",
         "Define photosynthesis."),
        # 2. Ohm's law
        ("State Ohm's law.",
         "What is Ohm's law?"),
        # 3. Reflection laws
        ("What are the laws of reflection?",
         "State the laws of reflection of light."),
        # 4. Metals reactivity
        ("Why are metals good conductors of electricity?",
         "Why do metals conduct electricity?"),
        # 5. Respiration
        ("What is respiration in organisms?",
         "Define respiration in living organisms."),
        # 6. Magnetic field
        ("What is a magnetic field?",
         "Define magnetic field."),
        # 7. Chemical equation balancing
        ("Why do we balance chemical equations?",
         "Why is it important to balance chemical equations?"),
        # 8. Decomposition reaction
        ("What is a decomposition reaction?",
         "Define decomposition reaction."),
    ])
    def test_paraphrase_hit(self, q1, q2):
        passed, reason = run_all_guards(q1, q2)
        assert passed, f"Expected HIT for '{q1}' vs '{q2}': {reason}"


# ============================================================
# 8 should-MISS lookalikes
# ============================================================

class TestShouldMissLookalikes:
    """Similar-looking questions that must NOT be cache hits."""

    @pytest.mark.parametrize("q1,q2", [
        # 1. Series vs parallel
        ("What is a series circuit?",
         "What is a parallel circuit?"),
        # 2. Acid vs base
        ("What are the properties of acids?",
         "What are the properties of bases?"),
        # 3. Alkane vs alkene
        ("What is an alkane?",
         "What is an alkene?"),
        # 4. Real vs virtual image
        ("What is a real image?",
         "What is a virtual image?"),
        # 5. Endothermic vs exothermic
        ("What is an endothermic reaction?",
         "What is an exothermic reaction?"),
        # 6. Arteries vs veins
        ("What are arteries?",
         "What are veins?"),
        # 7. Different numerical values
        ("A concave mirror of focal length 15 cm",
         "A concave mirror of focal length 10 cm"),
        # 8. Myopia vs hypermetropia
        ("What is myopia?",
         "What is hypermetropia?"),
    ])
    def test_lookalike_miss(self, q1, q2):
        """Guards fail-fast — we just verify the result is MISS."""
        passed, reason = run_all_guards(q1, q2)
        assert not passed, f"Expected MISS for '{q1}' vs '{q2}', got: {reason}"


# ============================================================
# Additional tests
# ============================================================

class TestCachePolicy:
    """Test cache storage policy."""

    def test_out_of_scope_not_cached(self):
        ok, _ = should_cache("Who won the World Cup?", "I can't answer that",
                             [], in_scope=False, is_style=False)
        assert not ok

    def test_no_citation_not_cached(self):
        ok, _ = should_cache("What is light?", "Light is...",
                             [], in_scope=True, is_style=False)
        assert not ok

    def test_valid_answer_cached(self):
        ok, _ = should_cache("What is refraction?", "Refraction is...",
                             ["Light – Reflection and Refraction"],
                             in_scope=True, is_style=False)
        assert ok

    def test_greeting_not_cached(self):
        ok, _ = should_cache("hello", "Hi there!", [], in_scope=True, is_style=False)
        assert not ok


class TestNumberExtraction:
    def test_extracts_with_units(self):
        nums = extract_numbers("R = 20 cm and f = 10 cm")
        assert nums == {"20cm", "10cm"}

    def test_extracts_without_units(self):
        nums = extract_numbers("What is 42?")
        assert "42" in nums

    def test_empty_for_no_numbers(self):
        nums = extract_numbers("What is refraction?")
        assert len(nums) == 0


class TestQuestionType:
    def test_define(self):
        assert detect_question_type("What is refraction?") == "define"

    def test_numerical(self):
        assert detect_question_type("Calculate the focal length") == "numerical"

    def test_difference(self):
        assert detect_question_type("Difference between acid and base") == "difference"

    def test_list(self):
        assert detect_question_type("List the properties of metals") == "list"

    def test_why(self):
        assert detect_question_type("Why is the sky blue?") == "why_how"

    def test_explain(self):
        assert detect_question_type("Explain the process of digestion") == "explain"


class TestStyleDetection:
    def test_style_phrases(self):
        assert is_style_request("explain it more simply")
        assert is_style_request("in points")
        assert is_style_request("give an example")
        assert is_style_request("shorter")

    def test_not_style(self):
        assert not is_style_request("What is refraction?")
        assert not is_style_request("Explain refraction of light")


# ============================================================
# Cache hit latency test
# ============================================================

class TestCacheLatency:
    """Cache hit must return in under 500ms."""

    def test_cache_hit_latency(self):
        """Store a question, look it up, and check latency."""
        from app.cache.store import CacheStore
        import tempfile
        from unittest.mock import patch

        # Use temp DB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        with patch("app.cache.store.CACHE_DB_PATH", tmp_db):
            store = CacheStore()
            store.store(
                "What is refraction of light?",
                "Refraction is the bending of light...",
                ["Light – Reflection and Refraction"],
            )

            start = time.perf_counter()
            entry, reason = store.lookup("What is refraction of light?")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert entry is not None, f"Expected cache hit, got: {reason}"
            assert elapsed_ms < 500, f"Cache hit took {elapsed_ms:.1f} ms (must be < 500)"
            print(f"\nCache hit latency: {elapsed_ms:.1f} ms")

        os.unlink(tmp_db)


# ============================================================
# Threshold tuning report
# ============================================================

def test_threshold_tuning_report(capsys):
    """Print similarity scores for all test pairs to help tune the threshold."""
    from fastembed import TextEmbedding
    import numpy as np

    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    pairs = [
        # Should HIT
        ("What is refraction?", "What does refraction mean?", True),
        ("What is photosynthesis?", "Define photosynthesis.", True),
        ("State Ohm's law.", "What is Ohm's law?", True),
        ("What are the laws of reflection?", "State the laws of reflection of light.", True),
        ("What is respiration in organisms?", "Define respiration in living organisms.", True),
        ("What is a magnetic field?", "Define magnetic field.", True),
        ("What is a decomposition reaction?", "Define decomposition reaction.", True),
        # Should MISS
        ("Image formed by concave mirror", "Image formed by convex mirror", False),
        ("What is a series circuit?", "What is a parallel circuit?", False),
        ("What are the properties of acids?", "What are the properties of bases?", False),
        ("What is an alkane?", "What is an alkene?", False),
        ("What is a real image?", "What is a virtual image?", False),
        ("What is myopia?", "What is hypermetropia?", False),
        ("Focal length when R = 20 cm", "Focal length when R = 30 cm", False),
    ]

    all_texts = []
    for q1, q2, _ in pairs:
        all_texts.extend([q1, q2])

    embeddings = list(model.embed(all_texts))
    vecs = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING REPORT")
    print("=" * 80)
    print(f"{'Expected':>8} {'Sim':>6} {'Guards':>7} {'Q1':<35} {'Q2':<35}")
    print("-" * 100)

    false_hits = 0
    missed_hits = 0

    for i, (q1, q2, should_hit) in enumerate(pairs):
        v1 = vecs[i * 2]
        v2 = vecs[i * 2 + 1]
        sim = float(np.dot(v1, v2))

        guards_pass, guard_reason = run_all_guards(q1, q2)

        # A "hit" happens if sim >= threshold AND guards pass
        would_hit = sim >= 0.88 and guards_pass

        if should_hit and not would_hit:
            missed_hits += 1
            status = "MISSED"
        elif not should_hit and would_hit:
            false_hits += 1
            status = "FALSE HIT"
        else:
            status = "OK"

        print(f"{'HIT' if should_hit else 'MISS':>8} {sim:>6.4f} "
              f"{'pass' if guards_pass else 'fail':>7} "
              f"{q1[:35]:<35} {q2[:35]:<35} {status}")

    print("-" * 100)
    print(f"False hits: {false_hits} (MUST be 0)")
    print(f"Missed hits: {missed_hits}")
    print("=" * 80)

    assert false_hits == 0, f"Found {false_hits} false hits — threshold too loose!"
