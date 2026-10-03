import json

from eval.beir import load


def make_dataset(root):
    root.mkdir(parents=True)
    docs = [
        {"_id": "d1", "title": "Cats", "text": "Cats purr."},
        {"_id": "d2", "title": "", "text": "Dogs bark."},
        {"_id": "d3", "title": "Birds", "text": "Birds sing."},
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    queries = [
        {"_id": "q1", "text": "which animal purrs"},
        {"_id": "q2", "text": "has no judgments"},
        {"_id": "q3", "text": "only zero grade"},
    ]
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t2\nq3\td2\t0\n"
    )
    return root


def test_load_builds_corpus_queries_qrels(tmp_path):
    corpus, queries, qrels = load(make_dataset(tmp_path / "toy"))
    assert corpus["d1"] == "Cats Cats purr."  # title prepended
    assert corpus["d2"] == "Dogs bark."
    assert queries == {"q1": "which animal purrs"}  # only queries with a positive judgment
    assert qrels == {"q1": {"d1": 1, "d3": 2}}
