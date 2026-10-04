"""Cache store — SQLite for persistence, in-memory FAISS for fast lookup."""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.config import (
    CACHE_DB_PATH, CACHE_SIMILARITY_THRESHOLD, CACHE_TOP_K,
    EMBED_MODEL, TEXTBOOK_VERSION, PROMPT_VERSION,
)
from app.cache.normalize import normalize, hash_text, contextual_key, extract_numbers
from app.cache.guards import run_all_guards
from app.cache.policy import should_serve_semantic

log = logging.getLogger(__name__)


class CacheStore:
    """SQLite + FAISS cache for question-answer pairs."""

    def __init__(self):
        os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(CACHE_DB_PATH, check_same_thread=False)
        self._create_table()

        # Embedding model (shared with retriever, loaded once)
        self.embed_model = TextEmbedding(EMBED_MODEL)
        self.dim = 384  # bge-small-en-v1.5 dimension

        # In-memory FAISS index for cache questions
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        # Parallel list: faiss_index row i → cache_entries[i]
        self.cache_entries: list[dict] = []

        # Hash map for exact-match lookup
        self.hash_map: dict[str, dict] = {}

        # Contextual key map for follow-up exact match
        self.contextual_map: dict[str, dict] = {}

        self._load_from_db()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                normalized TEXT NOT NULL,
                question_hash TEXT NOT NULL,
                embedding BLOB,
                answer TEXT NOT NULL,
                citations TEXT NOT NULL,
                created_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                textbook_version TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                contextual_key TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash ON cache(question_hash)"
        )
        self.conn.commit()

    def _load_from_db(self):
        """Load current-version entries into memory at startup."""
        rows = self.conn.execute(
            "SELECT id, question, normalized, question_hash, embedding, "
            "answer, citations, contextual_key "
            "FROM cache WHERE textbook_version = ? AND prompt_version = ?",
            (TEXTBOOK_VERSION, PROMPT_VERSION),
        ).fetchall()

        if not rows:
            log.info("Cache is empty — starting fresh")
            return

        embeddings = []
        for row in rows:
            entry = {
                "id": row[0],
                "question": row[1],
                "normalized": row[2],
                "question_hash": row[3],
                "answer": row[5],
                "citations": json.loads(row[6]),
                "contextual_key": row[7],
            }
            # Exact-match map
            self.hash_map[row[3]] = entry

            # Contextual key map
            if row[7]:
                self.contextual_map[row[7]] = entry

            # Embedding for FAISS
            if row[4]:
                vec = np.frombuffer(row[4], dtype="float32")
                embeddings.append(vec)
                self.cache_entries.append(entry)

        if embeddings:
            matrix = np.array(embeddings, dtype="float32")
            self.faiss_index.add(matrix)

        log.info(f"Loaded {len(rows)} cache entries ({self.faiss_index.ntotal} with embeddings)")

    def lookup(self, question: str, prev_standalone: str | None = None,
               is_followup: bool = False) -> tuple[dict | None, str]:
        """Look up a question in the cache.
        Returns (entry_or_None, reason_string)."""

        # Step 1: exact match on normalized hash
        qhash = hash_text(question)
        if qhash in self.hash_map:
            entry = self.hash_map[qhash]
            self._bump_hit(entry["id"])
            return entry, "hit: exact match"

        # For follow-ups: only contextual exact match allowed
        if is_followup:
            if prev_standalone:
                ckey = contextual_key(prev_standalone, question)
                if ckey in self.contextual_map:
                    entry = self.contextual_map[ckey]
                    self._bump_hit(entry["id"])
                    return entry, "hit: contextual exact match"
            return None, "miss: follow-up, no contextual exact match"

        # Step 2: check if semantic lookup is allowed
        has_numbers = bool(extract_numbers(question))
        allowed, reason = should_serve_semantic(question, has_numbers)
        if not allowed:
            return None, f"miss: {reason}"

        # Step 3: semantic search
        if self.faiss_index.ntotal == 0:
            return None, "miss: cache empty"

        vec = self._embed(question)
        k = min(CACHE_TOP_K, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(vec, k)

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_score < CACHE_SIMILARITY_THRESHOLD:
            return None, f"miss: best similarity {best_score:.4f} < {CACHE_SIMILARITY_THRESHOLD}"

        candidate = self.cache_entries[best_idx]

        # Step 4: run safety guards
        passed, guard_reason = run_all_guards(question, candidate["question"])
        if not passed:
            return None, f"miss: {guard_reason}"

        self._bump_hit(candidate["id"])
        return candidate, f"hit: semantic (sim={best_score:.4f}), {guard_reason}"

    def store(self, question: str, answer: str, citations: list[str],
              ctx_key: str | None = None):
        """Store a new entry in the cache."""
        norm = normalize(question)
        qhash = hash_text(question)

        # Don't store duplicates
        if qhash in self.hash_map:
            return

        vec = self._embed(question)
        embedding_blob = vec.tobytes()

        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO cache (question, normalized, question_hash, embedding, "
            "answer, citations, created_at, textbook_version, prompt_version, contextual_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question, norm, qhash, embedding_blob, answer,
             json.dumps(citations), now, TEXTBOOK_VERSION, PROMPT_VERSION, ctx_key),
        )
        self.conn.commit()
        entry_id = cur.lastrowid

        entry = {
            "id": entry_id,
            "question": question,
            "normalized": norm,
            "question_hash": qhash,
            "answer": answer,
            "citations": citations,
            "contextual_key": ctx_key,
        }

        # Update in-memory structures
        self.hash_map[qhash] = entry
        if ctx_key:
            self.contextual_map[ctx_key] = entry

        self.faiss_index.add(vec)
        self.cache_entries.append(entry)

        log.info(f"Cached: {question[:60]}...")

    def _embed(self, text: str) -> np.ndarray:
        vec = list(self.embed_model.embed([text]))[0]
        vec = np.array([vec], dtype="float32")
        vec = vec / np.linalg.norm(vec)
        return vec

    def _bump_hit(self, entry_id: int):
        self.conn.execute(
            "UPDATE cache SET hit_count = hit_count + 1 WHERE id = ?",
            (entry_id,),
        )
        self.conn.commit()
