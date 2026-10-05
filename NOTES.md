# NOTES.md — Experiment Log

> Honest record of what I tried and what happened. Real numbers, real failures.

## Threshold Tuning

### Embedding Similarity (cosine)

Ran the tuning report across 14 test pairs (7 should-hit, 7 should-miss):

| Pair | Expected | Similarity | Guards |
|------|----------|-----------|--------|
| "What is refraction?" / "What does refraction mean?" | HIT | 0.9565 | pass |
| "What is photosynthesis?" / "Define photosynthesis." | HIT | 0.9311 | pass |
| "State Ohm's law." / "What is Ohm's law?" | HIT | 0.8863 | pass |
| "Laws of reflection" / "State laws of reflection of light" | HIT | 0.8744 | pass |
| "Respiration in organisms" / "Respiration in living organisms" | HIT | 0.9327 | pass |
| "What is a magnetic field?" / "Define magnetic field." | HIT | 0.9329 | pass |
| "Decomposition reaction" (2 phrasings) | HIT | 0.9592 | pass |
| "Image by concave mirror" / "Image by convex mirror" | MISS | **0.9505** | fail (contrast) |
| "Series circuit" / "Parallel circuit" | MISS | 0.7784 | fail (contrast) |
| "Properties of acids" / "Properties of bases" | MISS | 0.7273 | fail (contrast) |
| "Alkane" / "Alkene" | MISS | 0.8269 | fail (contrast) |
| "Real image" / "Virtual image" | MISS | 0.7772 | fail (contrast) |
| "Myopia" / "Hypermetropia" | MISS | 0.6989 | fail (contrast) |
| "R = 20 cm" / "R = 30 cm" | MISS | 0.9244 | fail (number) |

### Key Insight: Embeddings alone are NOT safe

**The concave/convex pair has similarity 0.9505** — higher than some valid paraphrases! A pure threshold approach would serve a WRONG answer with high confidence. The contrast guard is what catches this.

Similarly, **R = 20 vs R = 30 scores 0.9244** — embeddings don't care about numbers. The number guard catches this.

### Chosen thresholds

- **Similarity**: 0.88 — catches most paraphrases. One false negative ("laws of reflection" at 0.8744) is acceptable; it just costs an LLM call.
- **Jaccard key-term overlap**: 0.65 — tuned down from 0.80. At 0.80, valid paraphrases with one extra word failed (e.g. "laws of reflection" vs "laws of reflection of light" — "light" adds an extra term). At 0.65, all dangerous lookalikes still fail (acids/bases = 0.33, arteries/veins = 0.00) while paraphrases pass.

## What Didn't Work

### 1. Jaccard threshold = 0.80
Too strict for short questions. Adding one word drops Jaccard from 1.0 to 0.67 in a 3-word overlap scenario. Had to lower to 0.65.

### 2. Question-type "define" as first rule
"What is Ohm's law?" matched "define" before "law", causing a type mismatch with "State Ohm's law." (which matched "law"). Fixed by reordering rules: specific types (law, numerical) come before the broad "define" catch-all.

### 3. Minimal stopword list
Words like "good", "important", "living" were treated as content words, causing key-term mismatches for valid paraphrases ("good conductors" vs "conduct"). Expanded stopwords and added synonym mappings (conductors→conductor, living→life).

### 4. Single-pass guard ordering
Guards run in order: number → contrast → key-term → type. Fail-fast design. Initially tests asserted on WHICH guard caught the mismatch, but since guards run in sequence, the first failure depends on the specific pair. Changed tests to only assert MISS, not the specific guard.

## Performance Numbers

- **Exact-hit latency (store lookup)**: 0.08 ms median
- **Exact-hit latency (end-to-end API)**: 4.59 ms median
- **Semantic-hit latency (store lookup + guards)**: 5.79 ms median
- **Semantic-hit latency (end-to-end API)**: 10.73 ms median
- **Fresh LLM generation (uncontended)**: 613.96 ms median (386 ms – 1739 ms range)
- **Embedding model load**: ~1.5s first time (cached after in memory)
- **FAISS index**: 734 chunks, 384-dimensional, ~1.1 MB
- **Ingestion**: 13 chapters, 734 chunks, ~10s including embedding

## PDF Extraction Issues

- Chapter 8 (Heredity): page 3 failed to extract (pypdf `LimitReachedError` — compressed page too large). Lost ~1 page of content but the chapter is short and mostly has diagrams on that page.
- Page headers vary: some pages show "Science128", others show "Heredity 129". Cleaning both patterns.
- Repeated figure/activity labels (PDF formatting artifact): "Figure 8.1Figure 8.1Figure 8.1..." — filtered with regex.
