# RAG Project — Confidence-Aware Hybrid Retrieval

[![ci](https://github.com/parthshah-tech/calibrated-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/parthshah-tech/calibrated-rag/actions/workflows/ci.yml)

A hybrid retrieval-augmented generation pipeline (dense embeddings + BM25, fused with
Reciprocal Rank Fusion, cross-encoder reranking, multi-query expansion, history-aware
rewriting) that **knows when it is unsure**. The disagreement between the dense and BM25
rankings is turned into a calibrated confidence signal, which is then used three ways:

1. **Shown to users** as a Low / Medium / High badge (the subject of the HCI study).
2. **Used by the system** to route each query through a cheaper or more thorough pipeline.
3. **Used to suggest better phrasings**, verified by re-running retrieval, not guessed.

Everything is measured by an eval harness that reports Recall@k, MRR, latency and
confidence intervals, so every claim below is backed by a number, not "it seemed to work".

> Course context: BITS F364 (Human-Computer Interaction) group project,
> *Calibrated Trust in RAG Answers via Retrieval Disagreement Signals*.
> This repository is the system; the user study, analysis and paper live alongside it.

---

## Status

| Area | State |
|---|---|
| Ingestion (content-hashed, idempotent), chunking (fixed, recursive), dense (Chroma) + BM25 + RRF, tracing, upload/search API, CI | built, Phase 0 |
| `confidence.py` (overlap, RBO, tau-on-intersection) wired into retrieval | built (thresholds are placeholders) |
| Eval harness, BEIR runner, signal validation, signal studies | built, Phase 1 |
| Reranking, multi-query, history-aware rewrite, badge UI | planned, Phase 2 |
| Verified reformulations, study mode, export | planned, Phase 3 |
| Confidence-gated routing, Retrieval Inspector, semantic chunking, full ablation | planned, Phase 4 |

Update this table as phases land. Do not claim numbers in this README until the eval
harness has produced them.

---

## Results (v1)

All numbers come from `benchmarks/results/` and reproduce with the commands in
[`benchmarks/results/README.md`](benchmarks/results/README.md). Retrieval and signal numbers
are deterministic (fixed seeds, cached embeddings); latency varies run to run and is not
reported yet.

**Hybrid retrieval vs the single retrievers**, BEIR test sets, nDCG@10 (95% bootstrap CI):

| Dataset | Dense | BM25 | Hybrid (RRF) | Hybrid minus dense |
|---|---|---|---|---|
| SciFact (300 queries) | 0.645 | 0.652 | 0.682 | +0.037 [+0.009, +0.064] |
| NFCorpus (323 queries) | 0.317 | 0.306 | 0.342 | +0.025 [+0.010, +0.041] |

**Does the confidence signal predict retrieval success?** The signal is how much the dense and
BM25 rankings agree (rank-biased overlap of their top 10, computed before fusion). Success means
a relevant document appears in the top 10. Held-out half, 95% CI:

| Dataset | AUROC | Success in Low, Medium, High |
|---|---|---|
| SciFact | 0.810 [0.742, 0.869] | 52%, 91%, 100% |
| NFCorpus | 0.752 [0.692, 0.806] | 48%, 74%, 85% |

**Which agreement metric is best?** AUROC [95% CI]:

| | SciFact | NFCorpus | Pooled |
|---|---|---|---|
| RBO (default) | 0.810 [0.742, 0.869] | 0.752 [0.692, 0.806] | 0.784 [0.740, 0.823] |
| Top-10 overlap | 0.700 [0.621, 0.777] | 0.740 [0.678, 0.800] | 0.733 [0.684, 0.778] |
| Tau-based score* | 0.720 [0.651, 0.786] | 0.719 [0.661, 0.776] | 0.726 [0.683, 0.768] |

RBO beats both significantly on SciFact and pooled (paired bootstrap on AUROC differences);
on NFCorpus the three are statistically indistinguishable. *Our tau score is Kendall's tau on
the shared items, rescaled to [0, 1] and weighted by the shared fraction (a provisional
definition).

**Do thresholds transfer between datasets?** The ordering does: under the other dataset's
cutoffs, success still rises from Low to High (58%, 86%, 92% on NFCorpus using SciFact's
cutoffs; 48%, 79%, 94% in reverse). Bucket sizes do not: under SciFact's cutoffs 55% of
NFCorpus queries fall in Low. Calibration error after transfer stayed close to within-dataset
calibration (ECE 0.068 and 0.042 vs 0.046 and 0.058, within noise at this sample size).

**Caveats.** Success means a relevant document was retrieved, not that an answer was correct.
Two datasets, both scientific or biomedical text. Cutoffs and calibration must be refit
on any new corpus.

---

## Architecture

```
Upload → Ingestion (PDF/DOCX/TXT/MD, content-hashed, idempotent)
       → Chunking (fixed | recursive | semantic)
       → Index: Chroma (dense)  +  BM25 (sparse, rebuilt from stored chunks at startup)

Query
  → History-aware rewrite            (standalone query from prior turns)
  → First-pass retrieval, in parallel:
        dense top-N   ─┐
        BM25  top-N   ─┴→ CONFIDENCE SIGNAL (rank agreement, pre-fusion)
  → RRF fusion
  → Router (uses confidence bucket)
        High    → rerank
        Medium  → multi-query expansion → RRF merge → rerank
        Low     → multi-query, wider candidate pool → rerank
        Floor   → abstain ("not covered by your documents")
  → Grounded generation + span-level source attribution
  → Response: answer, citations, confidence badge, verified reformulations
```

Design rules:

- **The confidence signal is computed once**, on the history-rewritten *original* query,
  from the raw dense and BM25 ranked lists, before fusion and before any multi-query
  merging. It measures the question, not the paraphrases.
- **Every stage is a swappable component** behind a small interface (`Retriever`,
  `Fuser`, `Reranker`, `Router`). The eval harness sweeps configurations by swapping
  them, not by editing code.
- **Every request is traced**: per-stage timings, list contents, and the routing
  decision are recorded in a structured trace.

---

## The confidence signal

`backend/confidence.py` — pure functions, unit-tested, no model needed.

Inputs: two ranked lists of chunk IDs (dense, BM25), top-k each.
Output: a score in [0, 1] and a bucket.

Metrics implemented (pluggable, and compared against each other in validation):

| Metric | Why |
|---|---|
| Top-k overlap (Jaccard / overlap@k) | Simple, interpretable baseline |
| Rank-Biased Overlap (RBO) | Handles lists with different members; weights top ranks more |
| Kendall's τ on the intersection | The metric named in the course proposal; ill-defined when lists barely overlap, so it is reported with the intersection size |

Bucketing (Low / Medium / High) uses thresholds fit on a **dev split** of the eval set
and reported on a **held-out split**.

### Validation (before any user study)

`eval/validate_signal.py` answers: *does the badge mean anything?*

- Retrieval success label: gold chunk in the final top-k.
- Reports success rate per bucket, AUROC of score vs success, and a reliability diagram.
- Calibration: isotonic regression (cross-validated), reported as ECE before and after.
- Metric comparison: overlap vs RBO vs τ, optionally combined with top-1 score margin and
  reranker score spread.
- Unanswerable questions are evaluated separately (correct behaviour = low confidence /
  abstain).

If the signal does not predict retrieval success, that is a finding, and it changes what
the user study can claim.

---

## Adaptive routing

`backend/router.py` maps confidence bucket to pipeline cost:

- High agreement skips multi-query expansion (saves one LLM round trip).
- Medium/Low escalate to expansion and a wider candidate pool.
- Below a floor (and with a weak top reranker score), the system abstains.

Claim to test in the eval: **routing matches the recall of always-full-pipeline at lower
p50 latency**. The harness reports both, with bootstrap CIs.

---

## Verified reformulations

`backend/reformulation.py` — when confidence is Medium/Low, each multi-query paraphrase
is run through dense + BM25, and its agreement score is computed. Only paraphrases whose
score beats the original by a margin are surfaced (max 3), each with its own bucket. If
none qualify, none are shown. This makes the suggestions a measured feature instead of
raw LLM output.

---

## Retrieval Inspector

A developer/demo view (disabled in study mode) showing:

- dense list and BM25 list side by side, with rank movement through RRF and reranking,
- the confidence score, bucket, and routing decision,
- per-stage timings from the trace.

Citations in the answer are span-level: clicking one scrolls the document viewer to the
exact source span and highlights it.

---

## Study mode (for the HCI study)

Enabled with `STUDY_MODE=1`. Designed so the survey/analysis team never needs code changes.

- **Participant code + condition assignment.** `/study/start?code=P017` assigns a
  condition (baseline / badge / badge + reformulations) by balanced random assignment,
  stored server-side so refreshes cannot change it.
- **Frozen configuration.** Pipeline config is hashed and logged with every session.
  Abstention and the Inspector are off in study mode so the badge is the only variable.
- **Same backend and corpus across conditions.** Only the frontend rendering differs.
- **Event log (SQLite, append-only).** One row per event:

  | Field | Notes |
  |---|---|
  | `participant_code`, `condition`, `session_id`, `config_hash` | identity and design |
  | `question_id`, `event_type`, `timestamp` | ordering and timing |
  | `event_type` values | `question_shown`, `answer_shown`, `doc_viewer_opened`, `doc_viewer_closed`, `reformulation_shown`, `reformulation_clicked`, `stated_confidence`, `final_answer_submitted` |
  | `payload` (JSON) | stated confidence (1–7), duration, clicked text, confidence score and bucket |

- **Export.** `uv run python -m backend.study.export --out data/` writes an anonymized
  per-participant, per-question CSV (no free-text identifiers). Correctness is joined from
  the frozen ground-truth question set, not labelled by hand.

Column contract for the export is versioned in `docs/data_contract.md`.

---

## Setup

```bash
mise install                      # pinned Python from mise.toml
uv sync --group dev --extra models   # runtime + dev deps; `models` adds sentence-transformers
                                     # (omit it for tests; EMBEDDER=hash runs fully offline)

cp .env.example .env
# set GROQ_API_KEY (https://console.groq.com)

uv run uvicorn backend.main:app --reload
```

Phase 0 exposes the API (frontend arrives in Phase 2):

```bash
curl -F file=@notes.md http://localhost:8000/ingest
curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "how does fusion work", "k": 5, "mode": "hybrid"}'
```

`mode` is `dense`, `bm25` or `hybrid`; hybrid responses include the confidence bucket and
a per-stage timing trace. API docs are served at `http://localhost:8000/docs`.

Key `.env` settings: `CHUNK_STRATEGY` (fixed | recursive | semantic), `CONFIDENCE_METRIC`
(overlap | rbo | tau), `ROUTING` (on | off), `STUDY_MODE` (0 | 1).

## Verify

```bash
uv run ruff check .
uv run ruff format .
uv run pytest -q
```

Unit tests cover the pure parts: chunking, BM25 ranking, RRF math, confidence metrics,
bucketing, router decisions, condition assignment. Model-backed stages (embedder,
reranker, LLM calls) are exercised by the eval harness on a real corpus. Pre-commit runs
the ruff checks; CI (`.github/workflows/ci.yml`) re-runs lint, format check and tests, and
runs a small fixed eval as a regression gate.

## Docker

```bash
docker build -t ragproject .
docker run -p 8000:8000 --env-file .env ragproject
```

Chroma runs embedded (`PersistentClient`), so no compose file is needed. First run
downloads `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face.

---

## Evaluation

```bash
# put 3–5 real documents in eval/corpus/
# write eval/sample_qa.json (schema below)

uv run python -m eval.eval_harness --corpus ./eval/corpus --qa ./eval/sample_qa.json
uv run python -m eval.validate_signal --corpus ./eval/corpus --qa ./eval/sample_qa.json
```

`eval/sample_qa.json` entry:

```json
{
  "id": "q017",
  "question": "…",
  "answer": "…",
  "source_file": "doc2.pdf",
  "gold_phrase": "a distinctive phrase from the gold chunk",
  "answerable": true,
  "coverage": "well-covered | partial | ambiguous | out-of-scope"
}
```

Public-benchmark and scale runs live in `benchmarks/`:

```bash
uv run python -m benchmarks.beir_eval --datasets scifact nfcorpus fiqa   # nDCG@10, Recall@100
uv run python -m benchmarks.scale --chunks 1000 10000 100000              # latency vs corpus size
uv run python -m benchmarks.load --clients 1 8 32                         # QPS, p50/p95
```

The harness sweeps chunking strategy × retrieval configuration (dense-only, BM25-only,
hybrid, hybrid+rerank, hybrid+rerank+multi-query, routed) and reports Recall@k, MRR and
p50/p95 latency to `eval_results.json`. Results include bootstrap 95% CIs and paired
bootstrap comparisons between configurations. Seeds are fixed; runs are reproducible from
one command.

---

## Repository layout

```
backend/
  main.py  config.py  pipeline.py  router.py  tracing.py
  ingestion.py  chunking.py  embedder.py
  hybrid_retrieval.py     # dense + BM25 + RRF
  confidence.py           # rank-agreement metrics + bucketing
  reformulation.py        # verified paraphrase suggestions
  query_expansion.py  history_aware.py  reranker.py  generation.py
  study/  assignment.py  logging.py  export.py
frontend/                 # chat, badge, reformulation chips, doc viewer, inspector
eval/  corpus/  sample_qa.json  eval_harness.py  validate_signal.py  stats.py
benchmarks/  beir_eval.py  scale.py  load.py
tests/
docs/  data_contract.md  complexity.md
```

---

## Complexity

Per-stage time and space analysis, the end-to-end query cost model, and a
predicted-vs-measured table (log-log scaling slopes from `benchmarks/scale.py`) are in
[`docs/complexity.md`](docs/complexity.md). Headline: dense retrieval scales sublinearly
(HNSW), BM25 via `rank_bm25` scales linearly in corpus size, and LLM stages are constant
in corpus size, which is what the confidence-gated router exploits.

## Known limitations

- **BM25 rebuilds on every add** (`rank_bm25` has no incremental API). The index is rebuilt
  from stored chunks at startup, so restarts are safe, but a large, frequently updated
  corpus would need a real inverted-index store (SQLite FTS5, OpenSearch).
- **Ingestion is idempotent by content hash**, but there is still no multi-user isolation;
  this is a single-tenant system.
- **Tiny corpora degrade the signal.** Found in the Phase 0 smoke test: with a single
  chunk, `rank_bm25`'s IDF goes non-positive, BM25 returns nothing, and the confidence
  score is 0 (Low). Overlap is also normalised by k, so a corpus with fewer than k chunks
  cannot reach High. Irrelevant for the 3-5 document study corpus, but the Phase 1
  validation must run on corpora with many chunks, and a non-negative IDF variant
  (Lucene-style) is a candidate fix.
- **Multi-query and history-aware rewriting each add an LLM round trip.** Routing reduces
  how often multi-query runs; the latency/recall trade-off is reported, not assumed.
- **Confidence is a retrieval-agreement signal, not a truth signal.** Both retrievers can
  agree on the wrong chunk, and the generator can still misread a right one. The badge
  should be described to users as "how much the search methods agree", not "how likely
  the answer is correct".
- **Small study sample.** Results from the user study are exploratory unless the
  participant count supports more.
