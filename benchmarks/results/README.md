# BEIR results (v1)

Reproduce:

    uv run python -m benchmarks.beir_eval --datasets scifact nfcorpus
    uv run python -m eval.validate_signal --results benchmarks/results/beir.json

Run on 2026-10-03, commit 5c2a204, CPU Intel(R) Core(TM) Ultra 5 125H, 7.6Gi RAM, no GPU, WSL2.

Setup: all-MiniLM-L6-v2 embeddings with exact (brute-force) cosine search; BM25 is rank_bm25 BM25Okapi with a plain word tokenizer (no stemming, no stopword removal); RRF k=60; confidence is RBO over the top 10 of each list, computed before fusion.

Caveats:
- Latency comparisons across modes are NOT valid yet (warm-up and order effects). Only BM25 timings are reliable.
- "success" means at least one relevant document is in the top 10. It measures retrieval, not answer correctness.
- Bucket thresholds are tertile cutoffs fit per group on a random dev half. They differ by dataset and must not be reused on another corpus.

## Signal studies

    uv run python -m eval.signal_studies --results benchmarks/results/beir.json

Compares overlap, RBO and a tau-based score (AUROC with paired bootstrap differences) and tests
whether cutoffs and calibration fit on one dataset transfer to the other. The tau score is
Kendall's tau on shared items, rescaled to [0, 1] and weighted by the shared fraction.
Transfer Brier skill is reported against both the training set's base rate and the test set's
own base rate; use the second, which does not flatter a transfer between different base rates.
