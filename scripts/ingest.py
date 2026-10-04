"""Build the book FAISS index from NCERT Class 10 Science PDFs.

Usage: python scripts/ingest.py
Reads PDFs from data/pdfs/, detects chapter names from the table of contents,
cleans text, splits into chunks, and saves a FAISS index to data/index/.
"""

import os
import re
import sys
import json
import hashlib

from pypdf import PdfReader
from fastembed import TextEmbedding
import numpy as np

# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PDF_DIR = os.path.join(PROJECT_DIR, "data", "pdfs")
INDEX_DIR = os.path.join(PROJECT_DIR, "data", "index")

CHUNK_SIZE = 800   # characters
CHUNK_OVERLAP = 100


# ---------- step 1: detect chapter names from TOC ----------

def extract_toc(pdf_dir: str) -> dict[int, str]:
    """Parse the preliminary-section PDF for the table of contents.
    Returns {chapter_number: chapter_name}."""
    ps_path = os.path.join(pdf_dir, "jesc1ps.pdf")
    if not os.path.exists(ps_path):
        sys.exit(f"TOC PDF not found: {ps_path}")

    reader = PdfReader(ps_path)
    toc = {}
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            continue
        # Pattern: "Chapter 8 Heredity 128"
        for m in re.finditer(r"Chapter\s+(\d+)\s+(.+?)\s+\d+\s*$", text, re.MULTILINE):
            num = int(m.group(1))
            name = m.group(2).strip()
            toc[num] = name
    return toc


# ---------- step 2: extract + clean text per chapter ----------

def clean_page_text(raw: str, chapter_name: str) -> str:
    """Remove headers, footers, page numbers, and repeated chapter headings."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip page headers like "Science128" or "Heredity 129"
        if re.match(r"^Science\s*\d+$", stripped):
            continue
        if re.match(r"^" + re.escape(chapter_name) + r"\s*\d*$", stripped):
            continue
        # Skip standalone page numbers
        if re.match(r"^\d+$", stripped):
            continue
        # Skip repeated "Activity X.Y" headers (5 times due to PDF formatting)
        if re.match(r"^(Activity \d+\.\d+){2,}", stripped):
            continue
        # Skip repeated "Figure X.Y" headers
        if re.match(r"^(Figure \d+\.\d+){2,}", stripped):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def extract_chapter_text(pdf_path: str, chapter_name: str) -> list[dict]:
    """Extract cleaned text from a chapter PDF.
    Returns list of {text, page} dicts (one per page)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text()
        except Exception:
            print(f"  Warning: could not extract page {i+1} of {pdf_path}")
            continue
        if not raw:
            continue
        text = clean_page_text(raw, chapter_name)
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


# ---------- step 3: chunk within a chapter ----------

def chunk_chapter(pages: list[dict], chapter: str) -> list[dict]:
    """Split chapter pages into overlapping chunks.
    Never mixes chapters. Keeps metadata: chapter, page."""
    chunks = []
    # Combine all pages into one string but track page boundaries
    combined = ""
    page_breaks = []  # (char_offset, page_number)
    for p in pages:
        page_breaks.append((len(combined), p["page"]))
        combined += p["text"] + "\n"

    if not combined.strip():
        return []

    # Sliding window over characters
    start = 0
    while start < len(combined):
        end = start + CHUNK_SIZE
        chunk_text = combined[start:end]

        # Try to break at a sentence or paragraph boundary
        if end < len(combined):
            # Look for last period/newline in the chunk
            last_break = max(chunk_text.rfind(". "), chunk_text.rfind(".\n"),
                             chunk_text.rfind("\n\n"))
            if last_break > CHUNK_SIZE // 2:
                end = start + last_break + 1
                chunk_text = combined[start:end]

        # Find which page this chunk starts on
        page_num = pages[0]["page"]
        for offset, pn in page_breaks:
            if offset <= start:
                page_num = pn

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "chapter": chapter,
                "page": page_num,
            })

        start = end - CHUNK_OVERLAP

    return chunks


# ---------- step 4: embed + build FAISS ----------

def build_index(chunks: list[dict]):
    """Embed all chunks and save a FAISS index + metadata."""
    import faiss

    print(f"Embedding {len(chunks)} chunks with fastembed...")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = list(model.embed(texts))
    matrix = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity (IndexFlatIP = dot product on unit vecs)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "book.index"))

    # Save metadata alongside (one JSON line per chunk)
    meta = [{"chapter": c["chapter"], "page": c["page"], "text": c["text"]}
            for c in chunks]
    with open(os.path.join(INDEX_DIR, "book_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved FAISS index ({dim}d, {len(chunks)} vectors) to {INDEX_DIR}")


# ---------- main ----------

def main():
    # Detect chapter names from TOC
    toc = extract_toc(os.path.join(PROJECT_DIR, "data", "pdfs"))
    print(f"Detected {len(toc)} chapters from TOC:")
    for num, name in sorted(toc.items()):
        print(f"  Chapter {num}: {name}")

    if len(toc) < 13:
        print("Warning: expected 13 chapters, got", len(toc))

    # Process each chapter PDF
    all_chunks = []
    for ch_num in sorted(toc.keys()):
        pdf_name = f"jesc1{ch_num:02d}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"  Skipping chapter {ch_num}: {pdf_path} not found")
            continue

        chapter_name = toc[ch_num]
        print(f"\nProcessing Chapter {ch_num}: {chapter_name} ({pdf_name})")
        pages = extract_chapter_text(pdf_path, chapter_name)
        print(f"  Extracted {len(pages)} pages")

        chunks = chunk_chapter(pages, chapter_name)
        print(f"  Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Show a sample chunk
    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]
        print(f"\nSample chunk (chapter: {sample['chapter']}, page: {sample['page']}):")
        print(sample["text"][:200] + "...")

    build_index(all_chunks)
    print("\nDone! Index saved to data/index/")


if __name__ == "__main__":
    main()
