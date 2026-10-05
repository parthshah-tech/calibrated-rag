# calibrated-rag

[![ci](https://github.com/parthshah-tech/calibrated-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/parthshah-tech/calibrated-rag/actions/workflows/ci.yml)

A document question-answering app that tells you how much to trust its retrieval. It searches your
documents with two methods at once (embedding search and BM25 keyword search), merges them, and
writes a cited answer. The two searches rarely agree when a question is poorly covered, so the app
turns their **agreement into a Low / Medium / High confidence badge**. The badge is tested against
public benchmarks, and the results are below.

It started as a personal project and became the system behind a group study in a Human-Computer
Interaction course (BITS F364) on whether such a signal helps people trust answers appropriately.
This repository contains the system and its evaluation, not the study.

## What it does

- **Hybrid retrieval.** Dense search (all-MiniLM-L6-v2 embeddings in a persistent Chroma store) and
  BM25 (`rank_bm25`), fused with Reciprocal Rank Fusion. Dense-only and keywords-only modes are
  also available.
- **Confidence signal.** Rank-biased overlap (RBO) of the top 10 of each search, computed before
  fusion, shown as a badge. Top-k overlap and a Kendall-tau-based score are available through
  `CONFIDENCE_METRIC`.
- **Cited answers.** An LLM (any OpenAI-compatible endpoint, Groq by default) answers from the
  retrieved passages only and cites them as `[1]`, `[2]`. If the passages don't contain the answer
  it says so. Deterministic answers are cached on disk.
- **Follow-up questions.** Earlier turns are used to rewrite a follow-up ("and who merges it?")
  into a standalone search query.
- **Broaden search.** Optionally asks the LLM for three differently-worded queries and merges all
  results with RRF.
- **Sharper ranking.** Optionally re-scores the top 20 candidates with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Off by default; see the results for why.
- **Chat interface.** Chats on the left (kept in your browser), your documents with checkboxes on
  the right so you choose which ones are searched, and the passages behind each answer.
- **Idempotent ingestion.** `.txt`, `.md`, `.pdf` and `.docx`, hashed by content, so uploading the
  same file twice does nothing. Chunking is recursive (paragraph, then sentence) or fixed-window.
- **Per-request tracing.** Each response carries stage timings and counters (LLM calls, cache hits,
  reranker pairs).
- **Evaluation tooling.** A BEIR runner, ranking metrics with bootstrap confidence intervals,
  validation of the confidence signal, and a harness for your own documents and questions.

## Quick start

Requires `mise` and `uv` (see `mise.toml`), or Python 3.12 and `uv`.

```bash
mise install
uv sync --group dev --extra models     # `models` adds sentence-transformers (pulls PyTorch)
cp .env.example .env                   # add your LLM key; the app works without one (sources only)
uv run uvicorn backend.main:app --env-file .env
```

Open <http://localhost:8000>. The first upload downloads the embedding model (about 90 MB) once.
Use **Add a document**, tick the sources you want, and ask a question. API docs are at `/docs`.

For tests and development without model downloads: `uv sync --group dev` and `EMBEDDER=hash`
(a deterministic, offline stand-in that is lexical only and must not be used for reported results).

## Configuration

Set in `.env` (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` (or `GROQ_API_KEY`) | none | Key for the LLM. Without it the app returns sources only. |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint. |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name; check what your provider offers. |
| `LLM_REASONING_EFFORT` | unset | e.g. `low`, passed through for reasoning models. |
| `DATA_DIR` | `./data` | Chroma index and the LLM answer cache. |
| `CHUNK_STRATEGY` | `recursive` | `recursive` or `fixed`. `CHUNK_SIZE` 512, `CHUNK_OVERLAP` 64. |
| `EMBEDDER` / `EMBED_MODEL` | `sentence-transformers` / `all-MiniLM-L6-v2` | `hash` for offline tests. |
| `CONFIDENCE_METRIC` / `CONFIDENCE_K` | `rbo` / `10` | `overlap`, `rbo` or `tau`; list depth compared. |
| `CANDIDATE_POOL` / `RRF_K` | `50` / `60` | Candidates per retriever; RRF constant. |
| `RERANK_MODEL` / `RERANK_POOL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `20` | Reranker and how many candidates it re-scores. |

Answer text and passages are sent to the configured LLM provider when an answer is written.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | The chat interface. |
| `GET /health` | Status and whether an LLM key is configured. |
| `GET /documents` | Ingested documents with passage counts. |
| `POST /ingest` | Upload a file (multipart `file`). |
| `POST /search` | Retrieval only: `query`, `k`, `mode`, `rerank`, `doc_ids`. |
| `POST /ask` | Retrieval plus a cited answer. Also takes `history`, `expand` (0 to 5 extra queries). |

Responses include `results`, `confidence`, `trace`, and for `/ask` an `answer` (text, cited
passage numbers, model, whether it was cached, whether the model said the documents don't answer),
`answer_error`, `rewritten_query`, `expansions`, `notes` and `reranked`. `doc_ids` limits the search
to the listed documents; an empty list searches nothing.

## How the confidence signal works

For each question the app takes the top 10 passages from dense search and the top 10 from BM25,
before fusion, and measures how much those two ranked lists agree with rank-biased overlap, which
weights agreement at the top of the lists more. The score is in [0, 1]. The badge cutoffs are
0.3 and 0.6, which are **provisional**: they were not fit on your documents. With a history, the
signal is computed on the rewritten standalone question.

The badge says how well the search found passages. It does not say whether the answer is correct.

## Results

All numbers come from `benchmarks/results/` and reproduce with the commands in
[`benchmarks/results/README.md`](benchmarks/results/README.md). Retrieval and signal numbers are
deterministic (fixed seeds, cached embeddings).

**Hybrid retrieval vs the single retrievers**, BEIR test sets, nDCG@10 (95% bootstrap CI):

| Dataset | Dense | BM25 | Hybrid (RRF) | Hybrid minus dense |
|---|---|---|---|---|
| SciFact (300 queries) | 0.645 | 0.652 | 0.682 | +0.037 [+0.009, +0.064] |
| NFCorpus (323 queries) | 0.317 | 0.306 | 0.342 | +0.025 [+0.010, +0.041] |

**Does the confidence signal predict retrieval success?** Success means a relevant document is in the
top 10. Held-out half, 95% CI:

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

RBO beats both significantly on SciFact and pooled (paired bootstrap on AUROC differences); on
NFCorpus the three are statistically indistinguishable. *The tau score is Kendall's tau on the
shared items, rescaled to [0, 1] and weighted by the shared fraction.

**Do cutoffs transfer between datasets?** The ordering does: under the other dataset's cutoffs,
success still rises from Low to High (58%, 86%, 92% on NFCorpus using SciFact's cutoffs; 48%, 79%,
94% in reverse). Bucket sizes do not: under SciFact's cutoffs 55% of NFCorpus queries fall in Low.

**Does the cross-encoder help?** On SciFact, reranking the top 50 of the hybrid results gave nDCG@10
0.695 [0.648, 0.739] against 0.682, a difference of +0.013 [-0.016, +0.043], which is not
significant. It raised median retrieval time from about 0.1 s to about 2.8 s on a laptop CPU, so it
is off by default. Reranking was evaluated on SciFact only.

## Evaluation

```bash
uv run python -m benchmarks.beir_eval --datasets scifact nfcorpus [--rerank]
uv run python -m eval.validate_signal --results benchmarks/results/beir.json
uv run python -m eval.signal_studies --results benchmarks/results/beir.json
uv run python -m eval.eval_harness --corpus ./eval/corpus --qa ./eval/sample_qa.json
```

The last command evaluates your own documents. Put them in `eval/corpus/` and write the questions
yourself:

```json
[{"id": "q1", "question": "...", "answer": "...", "source_file": "doc.pdf",
  "gold_phrase": "a short phrase from the passage that answers it",
  "answerable": true, "coverage": "well-covered"}]
```

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

CI runs the same three commands. The tests cover the pure logic (chunking, BM25, RRF, confidence
metrics, ranking metrics, statistics), the LLM client against a local stub server, generation,
the pipeline with a stand-in embedder and real Chroma, and the API. The embedding model, the
cross-encoder and a live LLM are exercised by the benchmark runs and by hand, not by CI.

```
backend/       app, pipeline, retrieval, confidence signal, LLM client, generation, reranker
frontend/      the chat interface (a single HTML file)
eval/          metrics, bootstrap statistics, calibration, BEIR loader, validation and studies
benchmarks/    BEIR runner and committed results
tests/
```

## Limitations

- **The badge measures retrieval agreement, not truth.** Both searches can agree on the wrong
  passage, and the model can misread a right one.
- **Cutoffs are provisional.** The validation covers two scientific datasets; fit cutoffs on your
  own documents before relying on the labels.
- **Citations are written by the model and not verified.** The prompt asks for answers from the
  passages only, but an answer can include outside knowledge or cite a passage that only loosely
  supports a sentence. The page flags answers that cite nothing.
- **Small corpora weaken the signal.** With fewer than 10 passages the top-10 lists cannot fully
  overlap, and with a single passage BM25 returns nothing.
- **BM25 is rebuilt on every upload and scores every passage per query** (`rank_bm25`). It is fine
  for hundreds of documents, not for very large collections.
- **The cross-encoder did not help on SciFact** and is slow on a CPU.
- **Single user, no authentication.** Chats live in the browser's local storage. The app is meant
  to run on your own machine.
- **Benchmark latency varies between runs** and is not reported as a finding.

## License

MIT
