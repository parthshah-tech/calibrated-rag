"""Run dense / bm25 / hybrid(RRF) retrieval on BEIR datasets through the real pipeline.

    uv run python -m benchmarks.beir_eval --datasets scifact nfcorpus fiqa

Each BEIR document is one retrievable unit (no chunking). Dense search is exact (in-memory
cosine), so these numbers are an upper bound on what approximate HNSW would give. Per-query
records are saved so eval.validate_signal can test the confidence signal on them."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.confidence import compute_confidence
from backend.config import Settings
from backend.embedder import build_embedder, clear_cache
from backend.interfaces import Chunk
from backend.pipeline import RAGPipeline
from backend.vector_store import InMemoryVectorStore
from eval import beir
from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from eval.stats import bootstrap_ci, paired_bootstrap, percentile

MODES = ("dense", "bm25", "hybrid")
CONF_METRICS = ("overlap", "rbo", "tau")


def conf_by_metric(res, k: int) -> dict:
    """Confidence score under every metric for one hybrid result (None for other modes)."""
    if res.confidence is None:  # not a hybrid retrieval; an empty list is still a valid input
        return {f"conf_{m}": None for m in CONF_METRICS}
    d, b = [h.chunk_id for h in res.dense], [h.chunk_id for h in res.bm25]
    return {f"conf_{m}": compute_confidence(d, b, metric=m, k=k).score for m in CONF_METRICS}


def retrieval_ms(trace: dict) -> float:
    """Total time minus the final chunk-text fetch (not part of retrieval proper)."""
    fetch = sum(s["ms"] for s in trace["spans"] if s["name"] == "fetch_chunks")
    return trace["total_ms"] - fetch


def embed_corpus(ids, texts, embedder, cache: Path | None, batch: int = 128, log=print):
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == list(ids):
            log(f"  embeddings loaded from {cache}")
            return z["emb"].tolist()
    out: list[list[float]] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        out.extend(embedder.embed(texts[i : i + batch]))
        if (i // batch) % 20 == 0:
            done = min(i + batch, len(texts))
            log(f"  embedded {done}/{len(texts)} ({time.perf_counter() - t0:.0f}s)")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(ids), emb=np.asarray(out, dtype=np.float32))
    return out


def evaluate_dataset(name, corpus, queries, qrels, embedder, cache=None, log=print) -> dict:
    ids = list(corpus)
    chunks = [Chunk(d, d, corpus[d], name, 0, 0, len(corpus[d])) for d in ids]
    log(f"[{name}] {len(ids)} docs, {len(queries)} queries")
    store = InMemoryVectorStore()
    store.add(chunks, embed_corpus(ids, [c.text for c in chunks], embedder, cache, log=log))
    settings = Settings(candidate_pool=100, confidence_k=10)
    pipeline = RAGPipeline(store, embedder, settings)  # builds BM25 from the store's chunks
    records = []
    for mode in MODES:
        log(f"  mode={mode}")
        for qid, q in queries.items():
            clear_cache(embedder)  # no mode may reuse another mode's query embedding
            res = pipeline.retrieve(q, k=100, mode=mode)
            ranked = [h.chunk_id for h in res.fused]
            rels = qrels[qid]
            records.append(
                {
                    "dataset": name,
                    "mode": mode,
                    "qid": qid,
                    "ndcg10": ndcg_at_k(ranked, rels, 10),
                    "mrr10": mrr_at_k(ranked, rels, 10),
                    "recall100": recall_at_k(ranked, rels, 100),
                    "success": bool(hit_at_k(ranked, rels, 10)),
                    "conf_score": res.confidence.score if res.confidence else None,
                    "conf_bucket": res.confidence.bucket if res.confidence else None,
                    **conf_by_metric(res, settings.confidence_k),
                    "latency_ms": res.trace["total_ms"],
                    "retrieval_ms": retrieval_ms(res.trace),
                }
            )
    return {"records": records, "summary": summarize(records)}


def summarize(records: list[dict]) -> dict:
    out: dict = {"modes": {}, "comparisons": {}}
    by_mode = {m: {r["qid"]: r for r in records if r["mode"] == m} for m in MODES}
    for m, rows in by_mode.items():
        vals = list(rows.values())
        out["modes"][m] = {
            "n": len(vals),
            **{
                k: dict(zip(("mean", "lo", "hi"), bootstrap_ci([r[k] for r in vals]), strict=True))
                for k in ("ndcg10", "mrr10", "recall100")
            },
            "latency_p50_ms": percentile([r["retrieval_ms"] for r in vals], 50),
            "latency_p95_ms": percentile([r["retrieval_ms"] for r in vals], 95),
        }
    for other in ("dense", "bm25"):
        qids = sorted(by_mode["hybrid"])
        out["comparisons"][f"hybrid_vs_{other}_ndcg10"] = paired_bootstrap(
            [by_mode["hybrid"][q]["ndcg10"] for q in qids],
            [by_mode[other][q]["ndcg10"] for q in qids],
        )
    return out


def format_summary(name: str, summary: dict) -> str:
    lines = [
        f"\n== {name} ==",
        f"{'mode':8} {'nDCG@10':>22} {'MRR@10':>8} {'R@100':>7} {'retr p50':>9} {'retr p95':>9}",
    ]
    for m, s in summary["modes"].items():
        nd = s["ndcg10"]
        lines.append(
            f"{m:8} {nd['mean']:.3f} [{nd['lo']:.3f}, {nd['hi']:.3f}] "
            f"{s['mrr10']['mean']:8.3f} {s['recall100']['mean']:7.3f} "
            f"{s['latency_p50_ms']:9.1f} {s['latency_p95_ms']:9.1f}"
        )
    for k, c in summary["comparisons"].items():
        flag = "significant" if c["significant"] else "not significant"
        lines.append(f"{k}: {c['mean_diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({flag})")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact"])
    ap.add_argument("--data-dir", default="./data/beir")
    ap.add_argument(
        "--embedder", default="sentence-transformers", choices=["sentence-transformers", "hash"]
    )
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--limit-queries", type=int, default=None, help="quick runs only")
    ap.add_argument("--out", default="benchmarks/results/beir.json")
    args = ap.parse_args(argv)

    embedder = build_embedder(args.embedder, args.model)
    all_records, summaries = [], {}
    for name in args.datasets:
        path = beir.download(name, args.data_dir)
        corpus, queries, qrels = beir.load(path)
        if args.limit_queries:
            queries = dict(list(queries.items())[: args.limit_queries])
        tag = args.model.replace("/", "_") if args.embedder != "hash" else "hash"
        result = evaluate_dataset(
            name, corpus, queries, qrels, embedder, Path(args.data_dir) / name / f"emb-{tag}.npz"
        )
        all_records += result["records"]
        summaries[name] = result["summary"]
        print(format_summary(name, result["summary"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"embedder": args.embedder, "model": args.model, "exact_dense_search": True}
    out.write_text(
        json.dumps({"meta": meta, "summary": summaries, "records": all_records}, indent=1)
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
