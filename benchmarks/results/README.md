# Benchmark results

Raw per-query results and summaries for the retrieval and confidence-signal experiments reported in
the main README.

| File | Contents |
|---|---|
| `beir.json` | Per-query records for dense, BM25, hybrid and (when run with `--rerank`) hybrid plus cross-encoder, on the BEIR datasets run, with summaries and paired comparisons. |
| `signal_validation.json` | Does the confidence score predict retrieval success? AUROC, success rate per bucket (held-out half), isotonic calibration. |
| `signal_studies.json` | Overlap vs RBO vs a tau-based score, and whether cutoffs and calibration transfer between datasets. |

## Reproduce

```bash
uv run python -m benchmarks.beir_eval --datasets scifact nfcorpus --rerank
uv run python -m eval.validate_signal --results benchmarks/results/beir.json
uv run python -m eval.signal_studies --results benchmarks/results/beir.json
```

The first run downloads the datasets from the public BEIR site into `data/beir/` and embeds the
documents once (cached on disk afterwards). Reranking 50 candidates per query on a CPU takes tens
of minutes; leave off `--rerank` to skip it. Each run rewrites `beir.json`.

## Setup behind the numbers

- Environment: WSL2 (Ubuntu 24.04) on a laptop CPU with 16 GB RAM and no GPU.
- Embeddings: `all-MiniLM-L6-v2` with exact (brute-force) cosine search, not the approximate index
  the app uses.
- BM25: `rank_bm25` BM25Okapi with a plain word tokenizer (no stemming, no stopword removal).
- Fusion: Reciprocal Rank Fusion with k = 60; 100 candidates per retriever.
- Confidence: rank-biased overlap of the top 10 of each list, computed before fusion.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 50 fused candidates; the rest of the
  ranking keeps its fused order.
- Intervals: percentile bootstrap, 2,000 resamples, fixed seed. Comparisons between configurations
  are paired bootstraps over the same queries.

## Notes on reading them

- "Success" means at least one relevant document is in the top 10. It measures retrieval, not
  whether a generated answer is correct.
- Bucket cutoffs are tertiles fit per group on a random dev half and reported on the other half.
  They differ by dataset and must not be reused on another corpus.
- Retrieval and signal numbers are identical across reruns. Latency is not: it varies between runs,
  so only the large reranker gap (roughly 0.1 s against 3 s median on this machine) should be
  read as meaningful.
- The tau-based score is Kendall's tau on the shared items, rescaled to [0, 1] and weighted by the
  shared fraction.
- Transfer Brier skill is reported against both the training set's base rate and the test set's own
  base rate; the second is the fairer one when base rates differ.
