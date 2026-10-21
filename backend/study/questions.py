"""The frozen question set. The file holds ground truth, so it must never be served to
participants or committed publicly (study/questions.json is gitignored)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REQUIRED = ("id", "question", "source_file", "gold_phrase", "answerable")


def load_questions(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"study questions file not found: {p} (see docs/study.md)")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("questions file must be a non-empty JSON list")
    seen = set()
    for q in data:
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            raise ValueError(f"question {q.get('id', '?')!r} is missing {missing}")
        if q["id"] in seen:
            raise ValueError(f"duplicate question id {q['id']!r}")
        seen.add(q["id"])
        if not isinstance(q["answerable"], bool):
            raise ValueError(f"question {q['id']!r}: answerable must be true or false")
        q.setdefault("attention_check", False)
        q.setdefault("practice", False)
        q.setdefault("coverage", "")
    return data


def public_view(questions: list[dict]) -> list[dict]:
    """What participants may see: id, text and whether it is the practice item. Nothing that
    reveals ground truth or which items are attention checks."""
    return [
        {"id": q["id"], "question": q["question"], "practice": q["practice"]} for q in questions
    ]


def ordered_questions(code: str, questions: list[dict]) -> list[dict]:
    """Practice item first, the rest in an order seeded by the participant code, with attention
    checks never first or last, so order is random across participants but reproducible."""
    rng = random.Random("order:" + code)
    rest = [q for q in questions if not q["practice"]]
    rng.shuffle(rest)
    for edge in (0, -1):
        if len(rest) > 2 and rest[edge]["attention_check"]:
            j = next((i for i in range(1, len(rest) - 1) if not rest[i]["attention_check"]), None)
            if j is not None:
                rest[edge], rest[j] = rest[j], rest[edge]
    return [q for q in questions if q["practice"]] + rest


def questions_hash(questions: list[dict]) -> str:
    blob = json.dumps(questions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
