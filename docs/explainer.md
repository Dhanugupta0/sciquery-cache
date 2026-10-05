# SciQuery: Architecture & Smart Caching Explainer

**SciQuery** is a RAG doubt-solving assistant for NCERT Class 10 Science that pairs semantic retrieval with a high-precision, safety-guarded semantic cache.

### 1. How the Chatbot Works
1. **Classification**: User queries are categorized into `STANDALONE` (new doubt), `FOLLOW_UP` (context-dependent query), or `STYLE` (formatting/simplification request) without calling an LLM.
2. **Follow-Up Rewrite**: Ambiguous follow-ups are contextualized into self-contained questions by an LLM prompt. Self-contained queries are preserved unchanged.
3. **Retrieval**: Relevant passages are retrieved from a FAISS index of 734 NCERT book chunks using `bge-small-en-v1.5` embeddings.
4. **Out-of-Scope Detection**: If similarity falls below 0.25 (retrieval threshold), the system politely declines with empty citations and skips LLM generation.
5. **Generation**: The LLM answers strictly using retrieved context and provides chapter-level citations.

### 2. How the Cache Works
The cache sits before retrieval and LLM stages to deliver near-zero latency answers safely:
- **Exact Hash Match**: Normalized queries (lowercased, punctuation-stripped) perform an instant SQLite lookup (~0.08 ms).
- **Semantic Candidate Retrieval**: Queries without numbers query a FAISS cache vector index (cosine similarity threshold ≥ 0.88).
- **Five Multi-Stage Safety Guards**:
  - *Number Guard*: Ensures numeric values and units match identically.
  - *Symbol Guard*: Enforces exact matching of single letters (e.g., optical points `C`, `F`, `P`) and alphanumeric symbols (`2f`, `2f1`).
  - *Contrast Guard*: Detects mutually exclusive pairs (e.g., `concave`/`convex`, `acid`/`base`, plurals) to reject opposite concepts.
  - *Key-Term Guard*: Verifies keyword overlap (Jaccard similarity ≥ 0.65) after stopword and synonym filtering.
  - *Question-Type Guard*: Ensures intent compatibility (`define`, `law`, `numerical`, `difference`).

### 3. What We Cache
- Verified standalone NCERT textbook questions with valid chapter citations.
- Rewritten follow-up queries that resolve to standalone canonical questions.
- Contextual exact follow-ups keyed with the prior session question.

### 4. What We NEVER Cache
- `STYLE` modifications ("simpler", "in points", "give an example") to keep formatting dynamic.
- Out-of-scope declines and empty responses.
- Queries containing specific numbers or calculations (exact hash only).
- Transient errors or rate-limit messages.

### 5. What Did Not Work
- **Pure Cosine Similarity**: Embeddings for opposing concepts scored deceptively high (e.g., `concave` vs `convex` mirror scored 0.9505; `R = 20` vs `R = 30 cm` scored 0.9244). Pure similarity yields critical factual hallucinations, necessitating explicit contrast and symbol guards.
- **Strict Jaccard Thresholds**: A 0.80 Jaccard threshold produced false misses on natural phrasing variations ("laws of reflection" vs "laws of reflection of light"). Lowering to 0.65 preserved accuracy while accommodating minor wording differences.
- **Unconditional Style Classification**: Filtering without session history or topic-word checks misclassified legitimate topic queries (e.g., "Give an example of a combination reaction") as style requests.

### Pipeline Flowchart
![Architecture Flowchart](flowchart.png)
