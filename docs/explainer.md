# SciQuery: Architecture & Smart Caching Explainer

**SciQuery** is a RAG doubt-solving assistant for NCERT Class 10 Science that pairs semantic retrieval with a high-precision, safety-guarded semantic cache.

### 1. How the Chatbot Works
1. **Classification**: User queries are categorized into `STANDALONE` (new doubt), `FOLLOW_UP` (context-dependent query), or `STYLE` (formatting/simplification request) without calling an LLM.
2. **Follow-Up Rewrite**: Ambiguous follow-ups are contextualized into self-contained questions by an LLM prompt. Self-contained queries are preserved unchanged.
3. **Retrieval**: Relevant passages are retrieved from a FAISS index of 734 NCERT book chunks using `bge-small-en-v1.5` embeddings.
4. **Out-of-Scope Detection**: Calibrated borderline band: retrieval score < 0.58 declines without LLM calls; 0.58–0.68 lets the LLM decide; ≥ 0.68 answers.
5. **Generation**: The LLM answers strictly using retrieved context and provides chapter-level citations.

### 2. How the Cache Works
The cache sits before retrieval and LLM stages to deliver safe answers at ultra-low latency:
- **Latency Profile**: Exact hit ~5 ms end to end (0.08 ms store), semantic hit ~11 ms (5.8 ms store), fresh answer about 0.4 to 2 s normally and up to ~10 s when rate-limited.
- **Exact Hash Match**: Normalized queries perform an instant SQLite lookup.
- **Semantic Candidate Retrieval**: Queries without numbers search a FAISS vector index (cosine similarity ≥ 0.88).
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
- Out-of-scope declines, greeting replies, and context-free prompts.
- Numerical and calculation questions (containing digits or calculate/compute triggers) to guarantee fresh arithmetic solutions.
- Transient errors or rate-limit messages.

### 5. What Did Not Work
- **Lens Formula Sign Error Cached**: A sign error in the lens formula occurred where the model got 12 cm instead of 60 cm for $f = 20\text{ cm}, u = -30\text{ cm}$, and this wrong answer was cached. Fix: a sign-convention prompt, a stronger model for numericals (`openai/gpt-oss-120b`), and never cache numericals.
- **Scope Threshold Calibration**: With a 0.50 cutoff, off-topic questions that scored 0.60 to 0.67 (for example Mendeleev's periodic table at 0.6665 and renewable energy at 0.6741) got past retrieval. A single 0.68 cutoff then wrongly declined real questions ("What is a neuron?" at 0.6666, "What is saponification?" at 0.6158). So there are now three bands: below 0.58 decline without the LLM, 0.58 to 0.68 let the LLM decide, 0.68 and above answer.
- **Pure Cosine Similarity**: Opposing concepts scored deceptively high (e.g., `concave` vs `convex` scored 0.9505; `R=20` vs `R=30` scored 0.9244), causing severe hallucinations without contrast and symbol guards.
- **Strict Jaccard Thresholds**: A 0.80 Jaccard threshold caused false misses on natural phrasing variations; lowering to 0.65 preserved safety while allowing natural phrasing.

### Pipeline Flowchart
![Architecture Flowchart](flowchart.png)
