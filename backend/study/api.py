"""Study endpoints. The condition is enforced here, on the server: a baseline participant's
responses never contain the confidence signal or rewordings, so the page cannot leak them."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .admin import register_admin
from .config import REFORMULATION
from .questions import load_questions, ordered_questions, public_view
from .store import CONDITIONS, StudyStore
from .survey import load_items, public_items
from .survey import validate as validate_survey
from .texts import load_texts

CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
EVENT_TYPES = {
    "consent_given",
    "phase_started",
    "question_shown",
    "doc_viewer_opened",
    "doc_viewer_closed",
    "reformulation_clicked",
    "final_answer",
    "stated_confidence",
    "final_answer_submitted",
    "time_expired",
    "survey",
}
NO_QUESTION = {"consent_given", "phase_started", "time_expired", "survey"}
MAX_PAYLOAD_CHARS = 4000
MAX_ANSWER_CHARS = 1000


class StartReq(BaseModel):
    code: str


class AskReq(BaseModel):
    code: str
    question_id: str
    text: str | None = Field(default=None, max_length=500)


class EventReq(BaseModel):
    code: str
    question_id: str | None = None
    type: str
    payload: dict = Field(default_factory=dict)


def register_study(app: FastAPI, settings, stimuli, questions=None, store=None) -> None:
    store = store or StudyStore(settings.study_db)
    questions = questions or load_questions(settings.study_questions)
    by_id = {q["id"]: q for q in questions}
    texts = load_texts()
    items = load_items()

    def participant(code: str) -> dict:
        if not CODE_RE.match(code):
            raise HTTPException(422, "code must be 1 to 32 letters, digits, - or _")
        p = store.participant(code)
        if p is None:
            raise HTTPException(404, "unknown participant code; start a session first")
        return p

    def question(qid: str | None) -> dict:
        if qid not in by_id:
            raise HTTPException(404, f"unknown question {qid!r}")
        return by_id[qid]

    def shown_suggestions(code: str, qid: str) -> list[str]:
        shown: list[str] = []
        for e in store.events(code, qid):
            if e["type"] == "reformulation_shown":
                shown = e["payload"].get("suggestions", [])
        return shown

    @app.get("/study", include_in_schema=False)
    def study_page():
        page = Path(__file__).resolve().parents[2] / "frontend" / "study.html"
        if not page.exists():
            raise HTTPException(404, "frontend/study.html is missing")
        return FileResponse(page)

    @app.get("/study/info")
    def study_info():  # read-only: shown before consent, creates no participant
        return {"info": texts["info"], "consent": texts["consent"]}

    @app.get("/study/config")
    def study_config():
        h, parts = stimuli.fingerprint(questions)
        return {
            "config_hash": h,
            "components": parts,
            "conditions": list(CONDITIONS),
            "frozen": stimuli.frozen,
        }

    @app.post("/study/start")
    def start(req: StartReq):
        if not CODE_RE.match(req.code):
            raise HTTPException(422, "code must be 1 to 32 letters, digits, - or _")
        h, _ = stimuli.fingerprint(questions)
        condition, created = store.assign(req.code, CONDITIONS, h)
        ordered = ordered_questions(req.code, questions)
        if created:
            store.log(
                req.code,
                None,
                "session_started",
                {"condition": condition, "config_hash": h, "order": [q["id"] for q in ordered]},
            )
        events = store.events(req.code)
        phase = [e["ts"] for e in events if e["type"] == "phase_started"]
        return {  # the condition is deliberately not returned
            "questions": public_view(ordered),
            "texts": {
                "info": texts["info"],
                "consent": texts["consent"],
                "instructions": texts["instructions"],
                "badge_note": texts["badge_note"] if condition != "baseline" else "",
                "debrief": texts["debrief"],
            },
            "survey": public_items(condition, items),
            "time_budget_min": settings.study_time_budget_min,
            "phase_started_at": phase[-1] if phase else None,
            "server_time": time.time(),
            "submitted": [
                e["question_id"] for e in events if e["type"] == "final_answer_submitted"
            ],
            "survey_done": any(e["type"] == "survey" for e in events),
            "resumed": not created,
        }

    @app.post("/study/ask")
    def ask(req: AskReq):
        p = participant(req.code)
        q = question(req.question_id)
        cond = p["condition"]
        text = (req.text or q["question"]).strip()
        via_reformulation = text != q["question"]
        if via_reformulation and (
            cond != "badge_reformulation" or text not in shown_suggestions(req.code, q["id"])
        ):
            raise HTTPException(403, "only the question or a suggested reformulation may be asked")
        want = cond == "badge_reformulation" and not via_reformulation
        try:
            stim = stimuli.ask(q["id"], text, want)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        conf = stim["confidence"]
        show_badge = cond in ("badge", "badge_reformulation") and conf is not None
        suggestions = (
            stim["suggestions"]
            if want and conf and conf["bucket"] in REFORMULATION["only_for"]
            else []
        )
        store.log(
            req.code,
            q["id"],
            "answer_shown",
            {
                "text": text,
                "via_reformulation": via_reformulation,
                "answer": stim["answer"],
                "answer_error": stim["answer_error"],
                "confidence": conf,
                "passages": stim["passages"],
                "shown_badge": show_badge,
                "shown_suggestions": [s["text"] for s in suggestions],
            },
        )
        if suggestions:
            store.log(
                req.code,
                q["id"],
                "reformulation_shown",
                {"suggestions": [s["text"] for s in suggestions], "details": suggestions},
            )
        body = {
            "answer": stim["answer"],
            "answer_error": stim["answer_error"],
            "passages": stim["passages"],
            "via_reformulation": via_reformulation,
        }
        if show_badge:
            body["confidence"] = conf
        if suggestions:
            body["reformulations"] = suggestions
        return body

    @app.post("/study/event")
    def event(req: EventReq):
        p = participant(req.code)
        if req.type not in EVENT_TYPES:
            raise HTTPException(422, f"unknown event type {req.type!r}")
        if req.type not in NO_QUESTION:
            question(req.question_id)
        if len(json.dumps(req.payload)) > MAX_PAYLOAD_CHARS:
            raise HTTPException(413, "payload too large")
        pl = req.payload
        if req.type == "stated_confidence":
            v = pl.get("value")
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 7:
                raise HTTPException(422, "stated confidence must be an integer from 1 to 7")
        if req.type == "final_answer":
            if not isinstance(pl.get("accepted"), bool):
                raise HTTPException(422, "accepted must be true or false")
            text = str(pl.get("text", "")).strip()
            if not pl["accepted"] and not text:
                raise HTTPException(422, "type your answer when you do not accept the assistant's")
            if len(text) > MAX_ANSWER_CHARS:
                raise HTTPException(413, "answer too long")
            pl = {"accepted": pl["accepted"], "text": "" if pl["accepted"] else text}
        if req.type == "survey":
            try:
                validate_survey(pl.get("values", {}), p["condition"], items)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            pl = {"values": pl["values"]}
        if req.type == "reformulation_clicked" and pl.get("text") not in shown_suggestions(
            req.code, req.question_id or ""
        ):
            raise HTTPException(403, "that reformulation was not shown to this participant")
        ts = time.time()
        store.log(req.code, req.question_id, req.type, pl, ts=ts)
        return {"ok": True, "ts": ts}

    @app.get("/study/documents")
    def study_documents():
        return stimuli.documents()

    @app.get("/study/documents/{doc_id}")
    def study_document(doc_id: str):
        d = stimuli.document(doc_id)
        if d is None:
            raise HTTPException(404, "unknown document")
        return d

    register_admin(app, settings, store, questions, items)
    app.state.study_store = store
