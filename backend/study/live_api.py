"""Two-condition study with voluntary live assistant use and server-owned progression."""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..ingestion import content_hash
from .live_materials import config_hash, load_materials
from .store import StudyStore

CONDITIONS = ("baseline", "badge")
COOKIE = "rag_study"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Consent(Input):
    consent: Literal[True]


class Identified(Input):
    request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,80}$")


class Ask(Identified):
    query: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    phase: Literal["practice", "task"]


class Answer(Input):
    text: str = Field(default="", max_length=4000)
    confidence: int | None = Field(default=None, ge=1, le=7, strict=True)


class Draft(Input):
    phase: Literal["practice", "task"]
    answers: dict[str, Answer]


class Submit(Identified):
    phase: Literal["instructions", "practice", "task", "survey", "interview"]
    answers: dict[str, Answer] = Field(default_factory=dict)
    survey: dict[str, str | int | None] = Field(default_factory=dict)


class Judgment(Identified):
    item_id: str
    judgment: Literal["correct", "incorrect", "unsure"]
    confidence: int = Field(ge=1, le=7, strict=True)


class View(Identified):
    kind: Literal["document", "citation", "evaluation"]
    target: str = Field(max_length=100)


def register_live_study(app, settings, get_pipeline, lock):
    materials = load_materials(settings.study_task)
    if not settings.study_salt:
        raise ValueError("Set STUDY_SALT to a stable secret before starting a live study")
    fingerprint = config_hash(materials, settings)
    store = StudyStore(settings.study_db)
    if any(p["config_hash"] != fingerprint for p in store.participants()):
        raise ValueError(
            "Study configuration changed. Use the original materials or a new STUDY_DB."
        )
    app.state.live_store = store
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    study_documents = {}
    for document in materials.documents:
        study_documents.setdefault(content_hash(document.text.encode())[:16], document)
    expected_docs = set(study_documents)
    ready = False

    def pipeline():
        nonlocal ready
        p = get_pipeline()
        if not ready:
            existing = {d["doc_id"] for d in p.store.documents()}
            if existing - expected_docs:
                raise HTTPException(503, "Study data directory contains a different corpus.")
            for document in materials.documents:
                p.ingest(document.name, document.text.encode())
            ready = True
        return p

    @app.middleware("http")
    async def study_cookie(request, call_next):
        if not request.url.path.startswith("/study"):
            return await call_next(request)
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Open the study on its own page."}, status_code=403)
        token = request.cookies.get(COOKIE, "")
        fresh = re.fullmatch(r"[a-f0-9]{64}", token) is None
        request.state.study_token = secrets.token_hex(32) if fresh else token
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        if fresh:
            response.set_cookie(
                COOKIE,
                request.state.study_token,
                path="/study",
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
                max_age=30 * 86400,
            )
        return response

    def code(request):
        return hashlib.sha256(request.state.study_token.encode()).hexdigest()

    def participant(request):
        p = store.participant(code(request))
        if p is None:
            raise HTTPException(403, "Read the information and consent before starting.")
        if p["config_hash"] != fingerprint:
            raise HTTPException(409, "This session belongs to another study configuration.")
        return p

    def events(p):
        return store.events(p["code"])

    def phase_info(ev):
        transitions = [e for e in ev if e["type"] == "live_phase"]
        if not transitions:
            return "instructions", None
        last = transitions[-1]
        phase = last["payload"]["phase"]
        if phase == "evaluation":
            judgments = {e["question_id"] for e in ev if e["type"] == "live_judgment"}
            if len(judgments) == len(materials.evaluations):
                phase = "survey"
        return phase, last

    def require_phase(p, allowed):
        ev = events(p)
        phase, latest = phase_info(ev)
        if phase not in allowed:
            raise HTTPException(409, "Your session has moved on. Refresh to resume.")
        return ev, phase, latest

    def deadline(ev):
        starts = [e for e in ev if e["type"] == "live_phase" and e["payload"]["phase"] == "task"]
        return starts[0]["ts"] + settings.study_time_budget_min * 60 if starts else None

    def public_answer(payload, p):
        result = {
            key: payload[key]
            for key in (
                "request_id",
                "query",
                "answer",
                "answer_error",
                "passages",
                "phase",
            )
        }
        if p["condition"] == "badge":
            result["score"] = payload["score"]
        return result

    def survey_items(p):
        return [i for i in materials.survey if p["condition"] in i.conditions]

    def snapshot(p):
        ev = events(p)
        phase, _ = phase_info(ev)
        result = {
            "phase": phase,
            "participant_id": "P-" + p["code"][:8],
            "server_time": time.time(),
            "deadline": deadline(ev),
            "preview": materials.preview,
            "title": materials.title,
            "instructions": materials.instructions,
            "task": materials.task,
            "practice": materials.practice.model_dump(exclude={"answer_key"}),
            "items": [i.model_dump(exclude={"answer_key"}) for i in materials.items],
            "transcript": [
                public_answer(e["payload"], p)
                for e in ev
                if e["type"] == "live_answer" and e["payload"]["phase"] == phase
            ],
            "draft": {},
            "evaluation": None,
            "survey": [i.model_dump(exclude={"conditions"}) for i in survey_items(p)],
            "debrief": materials.debrief if phase in {"complete", "withdrawn"} else "",
        }
        if p["condition"] == "badge":
            result["confidence_note"] = (
                "The system confidence bar is estimated from agreement between two document "
                "searches. Use the sources to judge the answer's accuracy."
            )
        drafts = [e for e in ev if e["type"] == "live_draft" and e["payload"]["phase"] == phase]
        if drafts:
            result["draft"] = drafts[-1]["payload"]["answers"]
        if phase == "evaluation":
            done = {e["question_id"] for e in ev if e["type"] == "live_judgment"}
            item = next(i for i in materials.evaluations if i.id not in done)
            result["evaluation"] = item.model_dump(exclude={"answer_key", "ai_correct", "score"})
            result["evaluation"].update(position=len(done) + 1, total=len(materials.evaluations))
            if p["condition"] == "badge":
                result["evaluation"]["score"] = item.score
        return result

    def prior_request(ev, request_id, types):
        return next(
            (e for e in ev if e["type"] in types and e["payload"].get("request_id") == request_id),
            None,
        )

    def validate_answers(phase, answers, complete):
        expected = (
            {materials.practice.id} if phase == "practice" else {i.id for i in materials.items}
        )
        if set(answers) - expected:
            raise HTTPException(422, "Unknown answer field.")
        if complete and (
            set(answers) != expected
            or any(
                not answer.text.strip() or answer.confidence is None for answer in answers.values()
            )
        ):
            raise HTTPException(422, "Answer each item and give a confidence rating.")
        return {key: answer.model_dump() for key, answer in answers.items()}

    @app.get("/study", include_in_schema=False)
    def study_page():
        return FileResponse(frontend / "live-study.html")

    @app.get("/study/app.js", include_in_schema=False)
    def study_script():
        return FileResponse(frontend / "live-study.js")

    @app.get("/study/info")
    def info():
        return {"title": materials.title, "info": materials.info, "preview": materials.preview}

    @app.get("/study/session")
    def session(request: Request):
        with lock:
            p = store.participant(code(request))
            return snapshot(participant(request)) if p else {"phase": "consent"}

    @app.post("/study/session")
    def consent(req: Consent, request: Request):
        with lock:
            token = code(request)
            store.assign(token, CONDITIONS, fingerprint)
            p = participant(request)
            if not events(p):
                store.log(
                    token,
                    None,
                    "live_phase",
                    {
                        "phase": "instructions",
                        "consent": True,
                        "config_hash": fingerprint,
                        "preview": materials.preview,
                    },
                )
            return snapshot(p)

    @app.post("/study/ask")
    def ask(req: Ask, request: Request):
        with lock:
            p = participant(request)
            ev = events(p)
            previous = prior_request(ev, req.request_id, {"live_answer"})
            if previous:
                if previous["payload"]["query"] != req.query:
                    raise HTTPException(409, "This request ID belongs to another query.")
                return public_answer(previous["payload"], p)
            _, phase, _ = require_phase(p, {req.phase})
            if phase == "task" and time.time() >= deadline(ev):
                raise HTTPException(409, "Task time has ended. Submit your current responses.")
            if not prior_request(ev, req.request_id, {"live_query"}):
                store.log(p["code"], None, "live_query", req.model_dump())
            history = []
            for event in ev:
                if event["type"] == "live_answer" and event["payload"]["phase"] == phase:
                    payload = event["payload"]
                    if payload["answer"]:
                        history.extend(
                            [
                                {"role": "user", "content": payload["query"]},
                                {"role": "assistant", "content": payload["answer"]["text"]},
                            ]
                        )
            started = time.time()
            r, answer, error = pipeline().answer(
                req.query, history=history[-6:], doc_ids=list(expected_docs)
            )
            payload = {
                **req.model_dump(),
                "answer": asdict(answer) if answer else None,
                "answer_error": error,
                "score": r.confidence.score if r.confidence else None,
                "passages": [
                    {"name": r.chunks[h.chunk_id].source, "text": r.chunks[h.chunk_id].text}
                    for h in r.fused
                    if h.chunk_id in r.chunks
                ],
                "duration_s": time.time() - started,
                "trace": r.trace,
                "rewritten_query": r.rewritten_query,
            }
            store.log(p["code"], None, "live_answer", payload)
            return public_answer(payload, p)

    @app.post("/study/draft")
    def draft(req: Draft, request: Request):
        with lock:
            p = participant(request)
            require_phase(p, {req.phase})
            answers = validate_answers(req.phase, req.answers, complete=False)
            store.log(p["code"], None, "live_draft", {"phase": req.phase, "answers": answers})
            return {"saved": True}

    @app.post("/study/submit")
    def submit(req: Submit, request: Request):
        with lock:
            p = participant(request)
            if prior_request(events(p), req.request_id, {"live_phase"}):
                return snapshot(p)
            ev, phase, _ = require_phase(p, {req.phase})
            payload = {"request_id": req.request_id, "from_phase": phase}
            if phase == "instructions":
                target = "practice"
            elif phase in {"practice", "task"}:
                expired = phase == "task" and time.time() >= deadline(ev)
                payload.update(
                    answers=validate_answers(phase, req.answers, complete=not expired),
                    expired=expired,
                )
                target = (
                    "task"
                    if phase == "practice"
                    else ("evaluation" if materials.evaluations else "survey")
                )
            elif phase == "interview":
                target = "complete"
            else:
                expected = {i.id: i for i in survey_items(p)}
                if set(req.survey) != set(expected):
                    raise HTTPException(
                        422, "Respond to each survey item, or choose Prefer not to answer."
                    )
                for key, value in req.survey.items():
                    item = expected[key]
                    valid = value is None or (
                        type(value) is int and 1 <= value <= 7
                        if item.type == "scale"
                        else value in item.options
                    )
                    if not valid:
                        raise HTTPException(422, "Invalid survey response.")
                payload["survey"] = req.survey
                target = "interview"
            store.log(p["code"], None, "live_phase", {**payload, "phase": target})
            return snapshot(p)

    @app.post("/study/judge")
    def judge(req: Judgment, request: Request):
        with lock:
            p = participant(request)
            if prior_request(events(p), req.request_id, {"live_judgment"}):
                return snapshot(p)
            require_phase(p, {"evaluation"})
            if req.item_id != snapshot(p)["evaluation"]["id"]:
                raise HTTPException(409, "Refresh to see the current answer-evaluation item.")
            store.log(p["code"], req.item_id, "live_judgment", req.model_dump())
            return snapshot(p)

    @app.get("/study/documents")
    def documents(request: Request):
        with lock:
            p = participant(request)
            require_phase(p, {"practice", "task", "evaluation"})
            return [
                {"doc_id": doc_id, "source": document.name}
                for doc_id, document in study_documents.items()
            ]

    @app.post("/study/view")
    def view(req: View, request: Request):
        with lock:
            p = participant(request)
            ev, phase, _ = require_phase(p, {"practice", "task", "evaluation"})
            if req.kind == "document":
                document = study_documents.get(req.target)
                if document is None:
                    raise HTTPException(404, "Document not found.")
                passages = [document.model_dump()]
            elif req.kind == "citation":
                answer = prior_request(ev, req.target, {"live_answer"})
                if not answer:
                    raise HTTPException(404, "Answer not found in this session.")
                passages = answer["payload"]["passages"]
            else:
                item = snapshot(p)["evaluation"]
                if not item or item["id"] != req.target:
                    raise HTTPException(404, "Evaluation item not available.")
                passages = item["passages"]
            if not prior_request(ev, req.request_id, {"live_view_opened"}):
                store.log(p["code"], None, "live_view_opened", {**req.model_dump(), "phase": phase})
            return {"view_id": req.request_id, "passages": passages}

    @app.post("/study/view/close")
    def close_view(req: Identified, request: Request):
        with lock:
            p = participant(request)
            ev = events(p)
            opened = prior_request(ev, req.request_id, {"live_view_opened"})
            if opened and not prior_request(ev, req.request_id, {"live_view_closed"}):
                store.log(
                    p["code"],
                    None,
                    "live_view_closed",
                    {
                        "request_id": req.request_id,
                        "phase": opened["payload"]["phase"],
                        "seconds": max(0, time.time() - opened["ts"]),
                    },
                )
            return {"ok": True}

    @app.post("/study/withdraw")
    def withdraw(request: Request):
        with lock:
            p = participant(request)
            phase, _ = phase_info(events(p))
            if phase != "withdrawn":
                store.log(p["code"], None, "live_phase", {"phase": "withdrawn"})
            return snapshot(p)

    from .live_admin import register_admin
    from .live_export import register_export

    register_export(app, settings, store, materials, fingerprint, lock)
    register_admin(app, settings, materials, frontend)
