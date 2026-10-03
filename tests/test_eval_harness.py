import json

from backend.embedder import HashEmbedder
from eval.eval_harness import format_summary, run_harness
from eval.eval_harness import main as harness_main

DOC1 = (
    "The espresso machine needs descaling every three months.\n\n"
    "Use the filtered water tank to reduce limescale buildup inside the boiler.\n\n"
    "Warranty claims require the original receipt and the serial number sticker."
)
DOC2 = (
    "Kubernetes schedules pods onto nodes based on resource requests.\n\n"
    "A readiness probe decides when a pod receives traffic from the service."
)
QA = [
    {
        "id": "q1",
        "question": "how often should the espresso machine be descaled",
        "answer": "every three months",
        "source_file": "coffee.txt",
        "gold_phrase": "descaling every three months",
        "answerable": True,
        "coverage": "well-covered",
    },
    {
        "id": "q2",
        "question": "what does a readiness probe do",
        "answer": "decides when a pod gets traffic",
        "source_file": "k8s.txt",
        "gold_phrase": "readiness probe decides",
        "answerable": True,
        "coverage": "well-covered",
    },
    {
        "id": "q3",
        "question": "who won the football world cup in 2010",
        "answer": "",
        "source_file": "",
        "gold_phrase": "",
        "answerable": False,
        "coverage": "out-of-scope",
    },
]


def make_corpus(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "coffee.txt").write_text(DOC1)
    (c / "k8s.txt").write_text(DOC2)
    return c


def test_harness_scores_answerable_and_separates_unanswerable(tmp_path):
    out = run_harness(make_corpus(tmp_path), QA, HashEmbedder(), size=120, overlap=20)
    recs = out["records"]
    assert {r["strategy"] for r in recs} == {"fixed", "recursive"}
    hyb = [
        r
        for r in recs
        if r["mode"] == "hybrid" and r["answerable"] and r["strategy"] == "recursive"
    ]
    assert all(r["hit5"] == 1.0 for r in hyb)
    unans = [r for r in recs if not r["answerable"]]
    assert unans and all(r["hit5"] is None and r["success"] is None for r in unans)
    assert "recursive/hybrid" in out["summary"]["configs"]
    assert "hit@5" in format_summary(out["summary"])


def test_cli_roundtrip(tmp_path):
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(QA))
    out = tmp_path / "out.json"
    harness_main(
        [
            "--corpus",
            str(make_corpus(tmp_path)),
            "--qa",
            str(qa),
            "--embedder",
            "hash",
            "--out",
            str(out),
        ]
    )
    assert json.loads(out.read_text())["records"]
