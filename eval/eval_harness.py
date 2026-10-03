"""Chunking-strategy x retrieval-mode evaluation on your own documents and QA pairs.

    uv run python -m eval.eval_harness --corpus ./eval/corpus --qa ./eval/sample_qa.json

A retrieved chunk counts as correct when it comes from the QA pair's source file AND contains
its gold_phrase (whitespace/case-insensitive). Keep gold phrases short: a phrase that straddles
a chunk boundary is a miss for that strategy. Answerable and unanswerable questions are
reported separately; unanswerable ones only contribute confidence-bucket statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.config import Settings
from backend.embedder import build_embedder
from backend.ingestion import SUPPORTED
from backend.pipeline import RAGPipeline
from backend.vector_store import InMemoryVectorStore
from eval.metrics import hit_at_k, mrr_at_k
from eval.stats import bootstrap_ci, paired_bootstrap, percentile

MODES = ("dense", "bm25", "hybrid")
STRATEGIES = ("fixed", "recursive")  # semantic joins in Phase 4


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def run_harness(corpus_dir, qa, embedder, strategies=STRATEGIES, size=512, overlap=64, k=10):
    files = sorted(p for p in Path(corpus_dir).iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise ValueError(f"no supported documents in {corpus_dir}")
    records = []
    for strategy in strategies:
        settings = Settings(chunk_strategy=strategy, chunk_size=size, chunk_overlap=overlap)
        pipe = RAGPipeline(InMemoryVectorStore(), embedder, settings)
        for f in files:
            pipe.ingest(f.name, f.read_bytes())
        for item in qa:
            gold = _norm(item.get("gold_phrase", ""))
            for mode in MODES:
                res = pipe.retrieve(item["question"], k=k, mode=mode)
                ranked = [h.chunk_id for h in res.fused]
                answerable = bool(item.get("answerable", True))
                rels = {
                    cid: 1
                    for cid, c in res.chunks.items()
                    if answerable
                    and c.source == item["source_file"]
                    and gold
                    and gold in _norm(c.text)
                }
                records.append(
                    {
                        "dataset": f"custom/{strategy}",
                        "strategy": strategy,
                        "mode": mode,
                        "qid": item["id"],
                        "answerable": answerable,
                        "coverage": item.get("coverage"),
                        "hit5": hit_at_k(ranked, rels, 5) if answerable else None,
                        "mrr10": mrr_at_k(ranked, rels, 10) if answerable else None,
                        "success": bool(hit_at_k(ranked, rels, 10)) if answerable else None,
                        "conf_score": res.confidence.score if res.confidence else None,
                        "conf_bucket": res.confidence.bucket if res.confidence else None,
                        "latency_ms": res.trace["total_ms"],
                    }
                )
    return {"records": records, "summary": summarize(records, strategies)}


def summarize(records, strategies) -> dict:
    out: dict = {"configs": {}, "comparisons": {}, "unanswerable_vs_answerable_buckets": {}}
    for s in strategies:
        for m in MODES:
            rows = [r for r in records if r["strategy"] == s and r["mode"] == m and r["answerable"]]
            if not rows:
                continue
            out["configs"][f"{s}/{m}"] = {
                "n": len(rows),
                "hit5": dict(
                    zip(("mean", "lo", "hi"), bootstrap_ci([r["hit5"] for r in rows]), strict=True)
                ),
                "mrr10": dict(
                    zip(("mean", "lo", "hi"), bootstrap_ci([r["mrr10"] for r in rows]), strict=True)
                ),
                "latency_p50_ms": percentile([r["latency_ms"] for r in rows], 50),
                "latency_p95_ms": percentile([r["latency_ms"] for r in rows], 95),
            }
        h = {
            r["qid"]: r["hit5"]
            for r in records
            if r["strategy"] == s and r["mode"] == "hybrid" and r["answerable"]
        }
        d = {
            r["qid"]: r["hit5"]
            for r in records
            if r["strategy"] == s and r["mode"] == "dense" and r["answerable"]
        }
        if h:
            q = sorted(h)
            out["comparisons"][f"{s}: hybrid_vs_dense_hit5"] = paired_bootstrap(
                [h[i] for i in q], [d[i] for i in q]
            )
        for flag, label in ((True, "answerable"), (False, "unanswerable")):
            buckets = Counter(
                r["conf_bucket"]
                for r in records
                if r["strategy"] == s and r["mode"] == "hybrid" and r["answerable"] is flag
            )
            out["unanswerable_vs_answerable_buckets"][f"{s}/{label}"] = dict(buckets)
    return out


def format_summary(summary: dict) -> str:
    lines = [f"{'config':18} {'n':>3} {'hit@5':>22} {'MRR@10':>8} {'p50 ms':>8}"]
    for name, s in summary["configs"].items():
        h = s["hit5"]
        lines.append(
            f"{name:18} {s['n']:3} {h['mean']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}] "
            f"{s['mrr10']['mean']:8.2f} {s['latency_p50_ms']:8.1f}"
        )
    for k, c in summary["comparisons"].items():
        lines.append(
            f"{k}: {c['mean_diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] "
            f"({'significant' if c['significant'] else 'not significant'})"
        )
    lines.append("confidence buckets (hybrid):")
    for k, v in summary["unanswerable_vs_answerable_buckets"].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument(
        "--embedder", default="sentence-transformers", choices=["sentence-transformers", "hash"]
    )
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args(argv)
    qa = json.loads(Path(args.qa).read_text())
    result = run_harness(args.corpus, qa, build_embedder(args.embedder, args.model))
    print(format_summary(result["summary"]))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
