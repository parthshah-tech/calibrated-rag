import json

from backend.embedder import HashEmbedder
from benchmarks.beir_eval import MODES, evaluate_dataset, format_summary
from benchmarks.beir_eval import main as beir_main
from eval.validate_signal import main as validate_main
from eval.validate_signal import validate_group

WORDS = [f"term{i}x" for i in range(400)]


def synthetic(n_docs=60, n_queries=40):
    corpus, queries, qrels = {}, {}, {}
    for i in range(n_docs):
        corpus[f"d{i}"] = " ".join(WORDS[i * 5 : i * 5 + 5]) + " filler words here"
    for i in range(n_queries):
        queries[f"q{i}"] = " ".join(WORDS[i * 5 : i * 5 + 3])  # overlaps doc i
        qrels[f"q{i}"] = {f"d{i}": 1}
    return corpus, queries, qrels


def test_evaluate_dataset_end_to_end():
    corpus, queries, qrels = synthetic()
    out = evaluate_dataset("toy", corpus, queries, qrels, HashEmbedder(), log=lambda *_: None)
    recs = out["records"]
    assert len(recs) == len(MODES) * len(queries)
    hybrid = [r for r in recs if r["mode"] == "hybrid"]
    assert all(r["conf_score"] is not None for r in hybrid)
    assert all(r["conf_score"] is None for r in recs if r["mode"] != "hybrid")
    assert out["summary"]["modes"]["hybrid"]["ndcg10"]["mean"] > 0.9  # lexical toy task is easy
    assert "hybrid_vs_dense_ndcg10" in out["summary"]["comparisons"]
    assert "nDCG@10" in format_summary("toy", out["summary"])


def test_cli_writes_results_and_signal_validation(tmp_path):
    root = tmp_path / "beir" / "toy"
    corpus, queries, qrels = synthetic(60, 40)
    root.mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": k, "text": v}) for k, v in corpus.items())
    )
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": k, "text": v}) for k, v in queries.items())
    )
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        + "\n".join(f"{q}\t{d}\t1" for q, r in qrels.items() for d in r)
    )
    out = tmp_path / "res.json"
    beir_main(
        [
            "--datasets",
            "toy",
            "--data-dir",
            str(tmp_path / "beir"),
            "--embedder",
            "hash",
            "--out",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["meta"]["exact_dense_search"] is True and data["records"]
    validate_main(["--results", str(out)])
    assert (tmp_path / "signal_validation.json").exists()


def test_validate_group_reports_signal_when_there_is_one():
    import numpy as np

    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 400)
    labels = rng.uniform(0, 1, 400) < scores
    g = validate_group(scores, labels)
    assert g["auroc"]["lo"] > 0.5
    assert g["monotonic"]
    low, _, high = (r["rate"] for r in g["heldout_buckets"])
    assert high > low


def test_every_mode_pays_for_its_own_query_embedding():
    from backend.embedder import CachedEmbedder

    emb = CachedEmbedder(HashEmbedder())
    corpus, queries, qrels = synthetic(60, 40)
    out = evaluate_dataset("toy", corpus, queries, qrels, emb, log=lambda *_: None)
    # corpus once, then query embeddings for dense AND hybrid (bm25 needs none); no cache reuse
    assert emb.misses == 60 + 2 * 40
    assert emb.hits == 0
    assert all("retrieval_ms" in r and r["retrieval_ms"] <= r["latency_ms"] for r in out["records"])
