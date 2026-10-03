"""Minimal BEIR loader (no `beir` package, no extra dependencies).

Layout of a BEIR dataset directory: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def download(name: str, root: str | Path) -> Path:
    """Fetch and unzip a BEIR dataset into root/name (skipped if already present)."""
    root = Path(root)
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / f"{name}.zip"
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    zpath.unlink()
    if not (target / "corpus.jsonl").exists():
        raise FileNotFoundError(f"{target}/corpus.jsonl missing after extraction")
    return target


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(path: str | Path, split: str = "test"):
    """Returns (corpus, queries, qrels).

    corpus  {doc_id: "title text"}   (title prepended, the BEIR convention)
    queries {qid: text}              only queries that have at least one relevant document
    qrels   {qid: {doc_id: grade}}   grade > 0 only"""
    path = Path(path)
    corpus = {}
    for row in _jsonl(path / "corpus.jsonl"):
        corpus[str(row["_id"])] = f"{row.get('title', '')} {row.get('text', '')}".strip()
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, grade = parts[0], parts[1], int(float(parts[2]))
            if grade > 0:
                qrels.setdefault(qid, {})[did] = grade
    queries = {
        str(r["_id"]): r["text"] for r in _jsonl(path / "queries.jsonl") if str(r["_id"]) in qrels
    }
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return corpus, queries, qrels
