"""Admin-only study archive. Correctness remains a human grading decision."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import zipfile
from typing import Annotated

from fastapi import Header, HTTPException
from fastapi.responses import Response

from .export import participant_id, to_csv


def build_archive(store, materials, salt, fingerprint):
    sessions, responses, human, ai, links, judgments, surveys, raw = [], [], [], [], [], [], [], []
    known_ai = set()
    rubric = {item.id: item for item in [materials.practice, *materials.items]}
    for person in store.participants():
        pid = participant_id(person["code"], salt, False)
        ev = store.events(person["code"])
        phases = [e for e in ev if e["type"] == "live_phase"]
        phase = phases[-1]["payload"]["phase"] if phases else "instructions"
        if phase == "evaluation" and len(
            {e["question_id"] for e in ev if e["type"] == "live_judgment"}
        ) == len(materials.evaluations):
            phase = "survey"
        main_start = next((e["ts"] for e in phases if e["payload"]["phase"] == "task"), None)
        main_end = next((e for e in phases if e["payload"].get("from_phase") == "task"), None)
        main_asks = [e for e in ev if e["type"] == "live_query" and e["payload"]["phase"] == "task"]
        opens = [
            e for e in ev if e["type"] == "live_view_opened" and e["payload"]["phase"] == "task"
        ]
        sessions.append(
            {
                "participant_id": pid,
                "participant_code": "P-" + person["code"][:8],
                "condition": person["condition"],
                "config_hash": fingerprint,
                "preview": materials.preview,
                "phase": phase,
                "withdrawn": phase == "withdrawn",
                "completed": phase == "complete",
                "task_started_at": main_start,
                "task_submitted_at": main_end["ts"] if main_end else None,
                "task_seconds": main_end["ts"] - main_start if main_end and main_start else None,
                "expired": main_end["payload"].get("expired") if main_end else None,
                "n_main_queries": len(main_asks),
                "n_document_opens": sum(e["payload"]["kind"] == "document" for e in opens),
                "n_citation_opens": sum(e["payload"]["kind"] == "citation" for e in opens),
                "seconds_to_first_query": main_asks[0]["ts"] - main_start
                if main_asks and main_start
                else None,
            }
        )
        for event in ev:
            payload = event["payload"]
            raw.append(
                {
                    **{k: event[k] for k in ("id", "question_id", "type", "ts", "payload")},
                    "participant_id": pid,
                }
            )
            if event["type"] == "live_phase" and "answers" in payload:
                for qid, answer in payload["answers"].items():
                    sid = hashlib.sha256(f"{salt}|{pid}|{qid}".encode()).hexdigest()[:16]
                    responses.append(
                        {
                            "participant_id": pid,
                            "condition": person["condition"],
                            "sheet_id": sid,
                            "phase": payload["from_phase"],
                            "item_id": qid,
                            **answer,
                            "withdrawn": phase == "withdrawn",
                        }
                    )
                    human.append(
                        {
                            "sheet_id": sid,
                            "item_id": qid,
                            "prompt": rubric[qid].prompt,
                            "answer_key": rubric[qid].answer_key,
                            "answer": answer["text"],
                            "grader1": "",
                            "grader2": "",
                            "final": "",
                        }
                    )
            if event["type"] == "live_answer":
                text = (payload["answer"] or {}).get("text", "")
                aid = hashlib.sha256(
                    json.dumps(
                        {
                            "query": payload["query"],
                            "answer": text,
                            "passages": payload["passages"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()[:16]
                links.append(
                    {
                        "participant_id": pid,
                        "condition": person["condition"],
                        "phase": payload["phase"],
                        "request_id": payload["request_id"],
                        "answer_id": aid,
                        "score": payload["score"],
                        "answer_error": payload["answer_error"],
                        "withdrawn": phase == "withdrawn",
                    }
                )
                if text and aid not in known_ai:
                    known_ai.add(aid)
                    ai.append(
                        {
                            "answer_id": aid,
                            "query": payload["query"],
                            "answer": text,
                            "passages": json.dumps(payload["passages"], ensure_ascii=False),
                            "grader1": "",
                            "grader2": "",
                            "final": "",
                        }
                    )
            if event["type"] == "live_judgment":
                item = next(i for i in materials.evaluations if i.id == payload["item_id"])
                judgments.append(
                    {
                        "participant_id": pid,
                        "condition": person["condition"],
                        **payload,
                        "ai_correct": item.ai_correct,
                        "score": item.score,
                        "withdrawn": phase == "withdrawn",
                    }
                )
            if event["type"] == "live_phase" and "survey" in payload:
                for item, value in payload["survey"].items():
                    surveys.append(
                        {
                            "participant_id": pid,
                            "condition": person["condition"],
                            "item_id": item,
                            "value": value,
                            "withdrawn": phase == "withdrawn",
                        }
                    )

    sheets = {
        "sessions.csv": (
            sessions,
            [
                "participant_id",
                "participant_code",
                "condition",
                "config_hash",
                "preview",
                "phase",
                "withdrawn",
                "completed",
                "task_started_at",
                "task_submitted_at",
                "task_seconds",
                "expired",
                "n_main_queries",
                "n_document_opens",
                "n_citation_opens",
                "seconds_to_first_query",
            ],
        ),
        "task_answers.csv": (
            responses,
            [
                "participant_id",
                "condition",
                "sheet_id",
                "phase",
                "item_id",
                "text",
                "confidence",
                "withdrawn",
            ],
        ),
        "human_answers_blind.csv": (
            sorted(human, key=lambda x: x["sheet_id"]),
            [
                "sheet_id",
                "item_id",
                "prompt",
                "answer_key",
                "answer",
                "grader1",
                "grader2",
                "final",
            ],
        ),
        "ai_answers_blind.csv": (
            sorted(ai, key=lambda x: x["answer_id"]),
            ["answer_id", "query", "answer", "passages", "grader1", "grader2", "final"],
        ),
        "answer_links.csv": (
            links,
            [
                "participant_id",
                "condition",
                "phase",
                "request_id",
                "answer_id",
                "score",
                "answer_error",
                "withdrawn",
            ],
        ),
        "judgments.csv": (
            judgments,
            [
                "participant_id",
                "condition",
                "item_id",
                "judgment",
                "confidence",
                "ai_correct",
                "score",
                "withdrawn",
            ],
        ),
        "survey.csv": (surveys, ["participant_id", "condition", "item_id", "value", "withdrawn"]),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, (rows, columns) in sheets.items():
            safe = [
                {
                    key: "'" + value
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                    else value
                    for key, value in row.items()
                }
                for row in rows
            ]
            archive.writestr(name, to_csv(columns, safe))
        archive.writestr("events.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in raw))
        archive.writestr("materials.json", materials.model_dump_json(indent=2))
        archive.writestr(
            "README.txt",
            (
                "Live voluntary-use study export\n"
                "Exclude preview sessions and withdrawn sessions from research analysis.\n"
                "Practice is identified by phase and must be kept separate from the main task.\n"
                "Use the blind grading sheets to assess correctness and reliance.\n"
                "task_answers links blind sheet IDs to participants.\n"
                "answer_links links AI grading IDs to participants.\n"
                "A completed task may have zero queries. Missing task rows are unanswered items.\n"
                "AI errors are in answer_links and retain empty answer fields.\n"
                "Event timestamps are Unix seconds. Durations measure elapsed viewing time.\n"
                "Null survey values mean skipped or not applicable. Scales run 1 to 7.\n"
                "The sample survey is project-specific and is not a validated SUS instrument.\n"
                "CSV formula-like text has a leading apostrophe. JSONL preserves exact text.\n"
                "IDs are pseudonyms. Free-text answers may contain personal information.\n"
                f"Configuration hash: {fingerprint}\n"
            ),
        )
    return output.getvalue()


def register_export(app, settings, store, materials, fingerprint, lock):
    def check(token):
        if not settings.study_admin_token:
            raise HTTPException(404, "Not found.")
        if not hmac.compare_digest((token or "").encode(), settings.study_admin_token.encode()):
            raise HTTPException(403, "Invalid administrator token.")

    @app.get("/study/admin/export.zip", include_in_schema=False)
    def download(x_admin_token: Annotated[str | None, Header()] = None):
        check(x_admin_token)
        with lock:
            data = build_archive(store, materials, settings.study_salt, fingerprint)
        return Response(
            data,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="live-study-export.zip"',
            },
        )

    @app.get("/study/admin/db", include_in_schema=False)
    def database(x_admin_token: Annotated[str | None, Header()] = None):
        check(x_admin_token)
        return Response(
            store.serialize(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="live-study.sqlite"',
            },
        )
