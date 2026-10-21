"""Turn the event log into the analysis dataset.

    uv run python -m backend.study.export --out data/study_export --salt SOMETHING-SECRET \
        [--db downloaded.sqlite] [--ai-grades graded_ai.csv] [--human-grades graded_human.csv]

Participant codes become salted hashes unless --keep-codes is given. Two grading sheets are
written for the team to fill in, and neither contains the condition or the badge:
  ai_answers.csv     each distinct assistant answer, once, with the answer key
  human_answers.csv  each answer a participant typed instead of accepting, shuffled
Give the filled-in sheets back with --ai-grades / --human-grades to get graded columns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import Settings
from .questions import load_questions, ordered_questions
from .store import StudyStore
from .survey import load_items

COLUMNS = [
    "participant_id",
    "condition",
    "config_hash",
    "position",
    "question_id",
    "practice",
    "attention_check",
    "coverage",
    "answerable",
    "unreached",
    "n_asks",
    "bucket",
    "badge_score",
    "shown_badge",
    "reformulations_shown",
    "n_reformulation_clicks",
    "final_via_reformulation",
    "final_bucket",
    "ai_answer_id",
    "ai_refused",
    "ai_correct",
    "ai_correct_source",
    "accepted",
    "final_answer_text",
    "human_correct",
    "final_correct",
    "appropriate_reliance",
    "overreliance",
    "underreliance",
    "stated_confidence",
    "consulted_doc",
    "doc_seconds",
    "time_to_confidence_s",
    "time_total_s",
    "submitted",
]
EVENT_COLUMNS = ["participant_id", "question_id", "type", "ts", "payload_json"]
AI_SHEET_COLUMNS = [
    "question_id",
    "ai_answer_id",
    "question",
    "answer_key",
    "ai_answer_text",
    "auto_correct",
    "grader1",
    "grader2",
    "final",
]
HUMAN_SHEET_COLUMNS = [
    "sheet_id",
    "question_id",
    "question",
    "answer_key",
    "human_answer_text",
    "grader1",
    "grader2",
    "final",
]


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def participant_id(code: str, salt: str, keep: bool) -> str:
    return code if keep else hashlib.sha256((salt + code).encode()).hexdigest()[:10]


def sheet_id(code: str, qid: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{code}|{qid}".encode()).hexdigest()[:8]


def ai_answer_id(payload: dict) -> str:
    text = (payload.get("answer") or {}).get("text")
    return hashlib.sha1(text.encode()).hexdigest()[:10] if text else "none"


def judge(q: dict, payload: dict) -> bool:
    """Automatic stand-in for grading: for an answerable question the answer is correct when the
    gold phrase is in the passages behind it and the model did not refuse; for an unanswerable
    one, when the model refused. No answer shown is never correct. Real grading replaces this."""
    answer = payload.get("answer")
    if not answer:
        return False
    refused = bool(answer.get("refused"))
    if not q["answerable"]:
        return refused
    gold = _norm(q["gold_phrase"])
    found = any(
        p["source"] == q["source_file"] and gold in _norm(p["text"]) for p in payload["passages"]
    )
    return found and not refused


def parse_grade(v: str) -> bool | None:
    v = (v or "").strip().lower()
    if v in ("1", "true", "yes", "y", "correct"):
        return True
    if v in ("0", "false", "no", "n", "incorrect"):
        return False
    return None


def load_grades(path: str | Path | None, key) -> dict:
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key(r): g for r in rows if (g := parse_grade(r.get("final", ""))) is not None}


def _doc_seconds(events: list[dict]) -> float:
    total, opened = 0.0, None
    end = max((e["ts"] for e in events), default=0.0)
    for e in events:
        if e["type"] == "doc_viewer_opened" and opened is None:
            opened = e["ts"]
        elif e["type"] == "doc_viewer_closed" and opened is not None:
            total += e["ts"] - opened
            opened = None
    if opened is not None:  # never closed: count until the last event for the question
        total += end - opened
    return round(total, 2)


def _row(part, q, position, events, pid, ai_grades, human_grades, salt) -> dict:
    row = {c: "" for c in COLUMNS}
    answers = [e for e in events if e["type"] == "answer_shown"]
    row |= {
        "participant_id": pid,
        "condition": part["condition"],
        "config_hash": part["config_hash"],
        "position": position,
        "question_id": q["id"],
        "practice": q["practice"],
        "attention_check": q["attention_check"],
        "coverage": q["coverage"],
        "answerable": q["answerable"],
        "unreached": not answers and not any(e["type"] == "question_shown" for e in events),
        "n_asks": len(answers),
        "reformulations_shown": any(e["type"] == "reformulation_shown" for e in events),
        "n_reformulation_clicks": sum(e["type"] == "reformulation_clicked" for e in events),
        "consulted_doc": any(e["type"] == "doc_viewer_opened" for e in events),
        "doc_seconds": _doc_seconds(events),
        "submitted": any(e["type"] == "final_answer_submitted" for e in events),
    }
    shown = next((e["ts"] for e in events if e["type"] == "question_shown"), None)
    conf = [e for e in events if e["type"] == "stated_confidence"]
    if conf:
        row["stated_confidence"] = conf[-1]["payload"]["value"]
        if shown is not None:
            row["time_to_confidence_s"] = round(conf[-1]["ts"] - shown, 2)
    done = [e for e in events if e["type"] == "final_answer_submitted"]
    if done and shown is not None:
        row["time_total_s"] = round(done[-1]["ts"] - shown, 2)
    if not answers:
        return row
    first, final = answers[0]["payload"], answers[-1]["payload"]
    c1, cf = first.get("confidence") or {}, final.get("confidence") or {}
    aid = ai_answer_id(final)
    graded = ai_grades.get((q["id"], aid))
    ai_correct = graded if graded is not None else judge(q, final)
    row |= {
        "bucket": c1.get("bucket", ""),
        "badge_score": round(c1["score"], 4) if "score" in c1 else "",
        "shown_badge": final["shown_badge"],
        "final_via_reformulation": final["via_reformulation"],
        "final_bucket": cf.get("bucket", ""),
        "ai_answer_id": aid,
        "ai_refused": bool((final.get("answer") or {}).get("refused")),
        "ai_correct": ai_correct,
        "ai_correct_source": "graded" if graded is not None else "auto",
    }
    fa = [e for e in events if e["type"] == "final_answer"]
    if fa:
        accepted, text = fa[-1]["payload"]["accepted"], fa[-1]["payload"]["text"]
        row |= {"accepted": accepted, "final_answer_text": text}
        if accepted:
            row["final_correct"] = ai_correct
        else:
            hg = human_grades.get(sheet_id(part["code"], q["id"], salt))
            if hg is not None:
                row |= {"human_correct": hg, "final_correct": hg}
        row |= {
            "appropriate_reliance": accepted == ai_correct,
            "overreliance": accepted and not ai_correct,
            "underreliance": (not accepted) and ai_correct,
        }
    return row


def build_rows(store, questions, salt="", keep_codes=False, ai_grades=None, human_grades=None):
    ai_grades, human_grades = ai_grades or {}, human_grades or {}
    by: dict = defaultdict(lambda: defaultdict(list))
    for e in store.events():
        if e["question_id"]:
            by[e["code"]][e["question_id"]].append(e)
    rows = []
    for part in store.participants():
        pid = participant_id(part["code"], salt, keep_codes)
        for pos, q in enumerate(ordered_questions(part["code"], questions)):
            evs = by[part["code"]].get(q["id"], [])
            rows.append(_row(part, q, pos, evs, pid, ai_grades, human_grades, salt))
    return rows


def ai_answer_sheet(store, questions) -> list[dict]:
    qmap, seen, rows = {q["id"]: q for q in questions}, set(), []
    for e in store.events():
        if e["type"] != "answer_shown":
            continue
        pl, q = e["payload"], qmap.get(e["question_id"])
        aid = ai_answer_id(pl)
        if q is None or aid == "none" or (q["id"], aid) in seen:
            continue
        seen.add((q["id"], aid))
        rows.append(
            {
                "question_id": q["id"],
                "ai_answer_id": aid,
                "question": q["question"],
                "answer_key": q.get("answer", ""),
                "ai_answer_text": pl["answer"]["text"],
                "auto_correct": judge(q, pl),
                "grader1": "",
                "grader2": "",
                "final": "",
            }
        )
    return sorted(rows, key=lambda r: (r["question_id"], r["ai_answer_id"]))


def human_answer_sheet(store, questions, salt="") -> list[dict]:
    qmap, rows = {q["id"]: q for q in questions}, []
    for e in store.events():
        overridden = e["type"] == "final_answer" and not e["payload"]["accepted"]
        if overridden and e["question_id"] in qmap:
            q = qmap[e["question_id"]]
            rows.append(
                {
                    "sheet_id": sheet_id(e["code"], q["id"], salt),
                    "question_id": q["id"],
                    "question": q["question"],
                    "answer_key": q.get("answer", ""),
                    "human_answer_text": e["payload"]["text"],
                    "grader1": "",
                    "grader2": "",
                    "final": "",
                }
            )
    latest = {r["sheet_id"]: r for r in rows}  # a participant's last typed answer wins
    out = list(latest.values())
    random.Random("grading").shuffle(out)  # no order cue about condition or participant
    return out


def survey_rows(store, items, salt="", keep_codes=False):
    cols = ["participant_id", "condition"] + [i["id"] for i in items]
    parts = {p["code"]: p for p in store.participants()}
    last: dict = {}
    for e in store.events():
        if e["type"] == "survey":
            last[e["code"]] = e["payload"]["values"]
    rows = [
        {
            "participant_id": participant_id(c, salt, keep_codes),
            "condition": parts[c]["condition"],
            **v,
        }
        for c, v in last.items()
    ]
    return cols, rows


def events_rows(store, salt="", keep_codes=False) -> list[dict]:
    return [
        {
            "participant_id": participant_id(e["code"], salt, keep_codes),
            "question_id": e["question_id"] or "",
            "type": e["type"],
            "ts": e["ts"],
            "payload_json": json.dumps(e["payload"], ensure_ascii=False),
        }
        for e in store.events()
    ]


def to_csv(columns, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_export(
    store,
    questions,
    items,
    out_dir,
    salt="",
    keep_codes=False,
    ai_grades_path=None,
    human_grades_path=None,
) -> dict[str, Path]:
    out = Path(out_dir)
    (out / "grading").mkdir(parents=True, exist_ok=True)
    ai = load_grades(ai_grades_path, lambda r: (r["question_id"], r["ai_answer_id"]))
    human = load_grades(human_grades_path, lambda r: r["sheet_id"])
    rows = build_rows(store, questions, salt, keep_codes, ai, human)
    scols, srows = survey_rows(store, items, salt, keep_codes)
    ai_sheet = ai_answer_sheet(store, questions)
    human_sheet = human_answer_sheet(store, questions, salt)
    files = {
        "per_question": (out / "per_question.csv", COLUMNS, rows),
        "events": (out / "events.csv", EVENT_COLUMNS, events_rows(store, salt, keep_codes)),
        "survey": (out / "survey.csv", scols, srows),
        "ai_answers": (out / "grading" / "ai_answers.csv", AI_SHEET_COLUMNS, ai_sheet),
        "human_answers": (
            out / "grading" / "human_answers.csv",
            HUMAN_SHEET_COLUMNS,
            human_sheet,
        ),
    }
    for path, cols, data in files.values():
        path.write_text(to_csv(cols, data), encoding="utf-8")
    return {k: v[0] for k, v in files.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/study_export")
    ap.add_argument("--db", default=None, help="a downloaded study.sqlite (default: STUDY_DB)")
    ap.add_argument("--salt", default=None, help="secret for the participant-code hash")
    ap.add_argument("--keep-codes", action="store_true", help="export raw codes (not anonymised)")
    ap.add_argument("--ai-grades", default=None)
    ap.add_argument("--human-grades", default=None)
    args = ap.parse_args(argv)
    s = Settings.from_env()
    salt = s.study_salt if args.salt is None else args.salt
    paths = write_export(
        StudyStore(args.db or s.study_db),
        load_questions(s.study_questions),
        load_items(),
        args.out,
        salt,
        args.keep_codes,
        args.ai_grades,
        args.human_grades,
    )
    for p in paths.values():
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
