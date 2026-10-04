"""Pre-fill the cache with common questions to avoid cold-start LLM calls.

Usage: python scripts/seed_cache.py
Requires: a running book index (run ingest.py first) and LLM credentials in .env
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import init_pipeline, run_pipeline

# Common questions students ask, one per chapter
SEED_QUESTIONS = [
    "What is a chemical reaction?",
    "What is the difference between an acid and a base?",
    "What are the properties of metals?",
    "What is a covalent bond?",
    "What is photosynthesis?",
    "What are hormones?",
    "What is reproduction?",
    "What is heredity?",
    "What is refraction of light?",
    "What is myopia?",
    "What is Ohm's law?",
    "What is an electromagnet?",
    "What is an ecosystem?",
]


def main():
    sessions = init_pipeline()
    sid = sessions.create()

    print(f"Seeding cache with {len(SEED_QUESTIONS)} questions...\n")
    for i, q in enumerate(SEED_QUESTIONS, 1):
        print(f"[{i}/{len(SEED_QUESTIONS)}] {q}")
        try:
            result = run_pipeline(sid, q)
            hit = result.get("cache_hit", False)
            ms = result.get("latency_ms", 0)
            print(f"  → {'CACHED' if not hit else 'HIT'} ({ms:.0f} ms)")
        except Exception as e:
            print(f"  → ERROR: {e}")
        # New session for each to avoid follow-up classification
        sid = sessions.create()

    print("\nDone! Cache seeded.")


if __name__ == "__main__":
    main()
