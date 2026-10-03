# Complexity Analysis

Time and space complexity for every stage, the query path end to end, and how each claim
is checked against measurement. Analysis says what should happen; `benchmarks/scale.py`
checks that it does.

## Notation

| Symbol | Meaning |
|---|---|
| N | chunks in the corpus |
| L | average tokens per chunk |
| d | embedding dimension (384 for all-MiniLM-L6-v2) |
| q | terms in the query |
| k | results kept per retriever |
| C | candidates passed to the reranker |
| m | multi-query paraphrases |
| U | union of ids across ranked lists, U ≤ (number of lists) · k |
| M, ef | HNSW graph degree and search breadth |
| E(x) | cost of one transformer forward pass over x tokens, roughly layers · (x² · d + x · d²) |

## Ingestion

| Stage | Time | Space | Notes |
|---|---|---|---|
| Content hash (idempotency) | O(bytes) | O(1) | Skips re-ingest of identical files |
| Fixed-window chunking | O(len) | O(len) | Single pass |
| Recursive chunking | O(len · levels) | O(len) | Levels = separator fallbacks, a small constant |
| Semantic chunking | O(s · E(l) + s · d) | O(s · d) | s sentences, l tokens each; embedding dominates, breakpoint scan is O(s) |
| Embedding all chunks | O(N · E(L)) | O(N · d) | Batching improves constants, not asymptotics |
| Chroma HNSW build | O(N log N) expected | O(N · (d + M)) | Heuristic bound; no worst-case guarantee |
| BM25 build (`rank_bm25`) | O(N · L) | O(N · L) worst case | Tokenize + per-document term-frequency dicts + IDF over vocabulary |
| BM25 rebuild on every add | O(N · L) per add | O(N · L) | Adding n documents one at a time costs O(n · N · L), i.e. quadratic in corpus growth. Known bottleneck |

Vector storage: N · d · 4 bytes in float32. For d = 384 that is 1,536 bytes per chunk,
about 154 MB for 100,000 chunks, plus the HNSW graph.

## Query path

| Stage | Time | Space | Notes |
|---|---|---|---|
| History-aware rewrite | one LLM call, O(history tokens) | O(history) | Latency dominated by network and decoding |
| Query embedding | O(E(q)) | O(d) | Cached; cache hit is O(1) |
| Dense retrieval (HNSW) | O(ef · log N) expected | O(k) | Brute force would be O(N · d); HNSW is approximate |
| BM25 retrieval (`rank_bm25`) | O(q · N) | O(N) score array | Scans every document for every query term |
| BM25 top-k | O(N log N) with argsort, O(N + k log k) with argpartition | O(N) | Library uses argsort; a wrapper can use argpartition |
| BM25 with inverted index (FTS5) | O(sum of posting lengths of query terms) | O(index) | Sublinear for rare terms; the fix if benchmarks show linear scaling hurts |
| Dense and BM25 in parallel | max(dense, BM25) wall clock | additive | Parallelism changes latency, not total work |
| Confidence: overlap@k | O(k) | O(k) | Set intersection |
| Confidence: RBO (truncated) | O(k) | O(k) | Running overlap with hash sets |
| Confidence: Kendall τ on intersection r | O(r²) naive, O(r log r) merge-based | O(r) | r ≤ k, negligible |
| RRF fusion | O(U) with hash map, O(U log U) with final sort | O(U) | Rank-based, no score normalization |
| Router | O(1) | O(1) | Threshold lookup |
| Multi-query expansion | m LLM calls (parallel: about one round trip) + m · (dense + BM25) + O(m · k) merge | O(m · k) | Latency is LLM-bound; retrieval part is dominated by BM25's O(q · N) |
| Verified reformulations | O(m · (dense + BM25 + confidence)) | O(m · k) | Reuses paraphrases; extra retrieval, no extra LLM call |
| Cross-encoder rerank | O(C · E(Lq + Lp)) | O(C) | One transformer pass per pair; linear in C, which is why C stays small |
| Generation | O(prompt) prefill + O(output) decode | O(prompt) | Prompt is about k · L tokens plus history |

### End to end

```
T_query ≈ T_rewrite
        + max(T_dense, T_bm25)          # parallel first pass
        + T_confidence + T_rrf          # both tiny
        + [ T_multiquery  if bucket ≠ High ]
        + T_rerank(C)
        + T_generate
```

With `rank_bm25`, retrieval time grows as O(q · N) and eventually dominates at large N.
With HNSW plus an inverted index it is roughly polylogarithmic in N for the dense side
and sublinear for the sparse side. LLM stages are constant in N and dominate at small N,
which is why routing (skipping multi-query for High confidence) matters most on small and
mid-size corpora.

## Space summary

| Component | Space |
|---|---|
| Dense vectors | O(N · d) |
| HNSW graph | O(N · M) |
| BM25 term-frequency dicts | O(N · L) worst case |
| Chunk text | O(N · L) |
| Embedding cache | O(cache size · d) |
| Traces and study log | O(events), append-only |

## Predicted vs measured (fill in from `benchmarks/scale.py`)

Fit latency against N on a log-log scale; the slope is the empirical exponent.

| Operation | Predicted scaling | Measured slope | Fit R² | Verdict |
|---|---|---|---|---|
| BM25 query (`rank_bm25`) | linear, slope about 1 | [ ] | [ ] | [ ] |
| Dense query (HNSW) | sublinear, slope well below 1 | [ ] | [ ] | [ ] |
| BM25 rebuild per add | linear per add, quadratic cumulative | [ ] | [ ] | [ ] |
| Ingestion throughput | roughly constant chunks/s (embedding-bound) | [ ] | [ ] | [ ] |
| Rerank | linear in C | [ ] | [ ] | [ ] |
| Memory vs N | linear | [ ] | [ ] | [ ] |

If a measured slope disagrees with the prediction, that is a result: either the analysis
missed a term (cache effects, batching, GIL) or the implementation has an accidental
inefficiency. Investigate and document which.

## Optimization candidates (each becomes a before/after benchmark)

1. Replace per-add BM25 rebuild with an incremental or FTS5-backed inverted index.
2. Use `argpartition` for BM25 top-k.
3. Batch embedding during ingestion; measure chunks/s against batch size.
4. Embedding cache; measure hit rate and latency saved.
5. Tune HNSW `ef_search` against recall (recall vs latency curve).
6. Tune reranker candidate count C against nDCG (accuracy vs latency curve).
