"""Save the study stimuli.

    uv run python -m backend.study.freeze

Needs the full install, the indexed documents, and an LLM key. Run it once after the question
set and cutoffs are final; the study server then runs from the saved file alone. Also writes a
blind grading sheet of every distinct assistant answer and an item report (the badge level and
automatic correctness per item) for choosing and checking the item set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..config import Settings
from ..main import build_pipeline
from .export import AI_SHEET_COLUMNS, ai_answer_id, judge, to_csv
from .questions import load_questions
from .stimuli import freeze

REPORT_COLUMNS = [
    "question_id",
    "practice",
    "attention_check",
    "answerable",
    "bucket",
    "score",
    "auto_correct",
    "n_suggestions",
]


def sheets(data: dict, questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """(blind grading sheet of every distinct assistant answer, item report for the team).
    The grading sheet carries no badge or condition; the report does, for choosing items."""
    qmap, seen, ai, report = {q["id"]: q for q in questions}, set(), [], []
    for qid, by_text in data["answers"].items():
        q = qmap[qid]
        for stim in by_text.values():
            if not stim["answer"]:
                continue
            pl = {"answer": stim["answer"], "passages": stim["passages"]}
            aid = ai_answer_id(pl)
            if (qid, aid) in seen:
                continue
            seen.add((qid, aid))
            ai.append(
                {
                    "question_id": qid,
                    "ai_answer_id": aid,
                    "question": q["question"],
                    "answer_key": q.get("answer", ""),
                    "ai_answer_text": stim["answer"]["text"],
                    "auto_correct": judge(q, pl),
                    "grader1": "",
                    "grader2": "",
                    "final": "",
                }
            )
        first = by_text[q["question"]]
        c = first["confidence"] or {}
        report.append(
            {
                "question_id": qid,
                "practice": q["practice"],
                "attention_check": q["attention_check"],
                "answerable": q["answerable"],
                "bucket": c.get("bucket", ""),
                "score": round(c["score"], 3) if "score" in c else "",
                "auto_correct": judge(
                    q, {"answer": first["answer"], "passages": first["passages"]}
                ),
                "n_suggestions": len(first["suggestions"]),
            }
        )
    return sorted(ai, key=lambda r: (r["question_id"], r["ai_answer_id"])), report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="study/frozen.json")
    ap.add_argument("--grading-sheet", default="study/ai_answers_to_grade.csv")
    ap.add_argument("--report", default="study/frozen_report.csv")
    args = ap.parse_args(argv)
    settings = Settings.from_env()
    questions = load_questions(settings.study_questions)
    holder: dict = {}

    def get():
        holder.setdefault("p", build_pipeline(settings))
        return holder["p"]

    data = freeze(settings, get, questions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in data["answers"].values())
    print(
        f"wrote {out}: {len(questions)} questions, {n} answers, config_hash {data['config_hash']}"
    )
    ai, report = sheets(data, questions)
    Path(args.grading_sheet).write_text(to_csv(AI_SHEET_COLUMNS, ai), encoding="utf-8")
    Path(args.report).write_text(to_csv(REPORT_COLUMNS, report), encoding="utf-8")
    print(f"wrote {args.grading_sheet} ({len(ai)} answers to grade, blind) and {args.report}")
    cells = Counter((r["bucket"], r["auto_correct"]) for r in report if not r["practice"])
    print("scored items by badge and automatic correctness (real grading decides):")
    for b in ("Low", "Medium", "High"):
        print(f"  {b:7} wrong {cells[(b, False)]}  right {cells[(b, True)]}")


if __name__ == "__main__":
    main()
