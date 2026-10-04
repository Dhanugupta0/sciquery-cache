"""Book retriever — loads the FAISS index once and answers similarity queries."""

import json
import os

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.config import BOOK_INDEX_DIR, EMBED_MODEL, RETRIEVAL_TOP_K


class BookRetriever:
    """Wraps the book FAISS index for semantic search over textbook chunks."""

    def __init__(self):
        index_path = os.path.join(BOOK_INDEX_DIR, "book.index")
        meta_path = os.path.join(BOOK_INDEX_DIR, "book_meta.json")

        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)

        # Load embedding model once
        self.embed_model = TextEmbedding(EMBED_MODEL)

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        """Return top_k results with keys: text, chapter, page, score (cosine sim)."""
        # Embed query
        vec = list(self.embed_model.embed([query]))[0]
        vec = np.array([vec], dtype="float32")
        # Normalize for cosine similarity
        vec = vec / np.linalg.norm(vec)

        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "chapter": meta["chapter"],
                "page": meta["page"],
                "score": float(score),
            })
        return results
