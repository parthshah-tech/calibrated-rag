"""Where a question's answer comes from. LiveStimuli runs the real pipeline; FrozenStimuli replays
answers saved earlier by `freeze`, so the study server needs no models, no index and no API key,
and every participant sees identical text.

    uv run python -m backend.study.freeze"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import FIXED, REFORMULATION, fingerprint
from .questions import questions_hash
from .viewer import reconstruct

FROZEN_VERSION = 1


class LiveStimuli:
    frozen = False

    def __init__(self, get_pipeline, settings) -> None:
        self._get, self._settings = get_pipeline, settings

    def ask(self, qid: str, text: str, want_suggestions: bool = False) -> dict:
        pipe = self._get()
        res, ans, err = pipe.answer(
            text, FIXED["k"], FIXED["mode"], None, FIXED["expand"], FIXED["rerank"], None
        )
        conf = res.confidence
        passages = [
            {"source": res.chunks[h.chunk_id].source, "text": res.chunks[h.chunk_id].text}
            for h in res.fused
            if h.chunk_id in res.chunks
        ]
        suggestions: list[dict] = []
        if want_suggestions and conf is not None and conf.bucket in REFORMULATION["only_for"]:
            suggestions = pipe.suggest_reformulations(
                text, conf.score, REFORMULATION["n"], REFORMULATION["margin"]
            )[: REFORMULATION["max_shown"]]
        return {
            "answer": asdict(ans) if ans else None,
            "answer_error": err,
            "confidence": asdict(conf) if conf else None,
            "passages": passages,
            "suggestions": suggestions,
        }

    def documents(self) -> list[dict]:
        return self._get().store.documents()

    def document(self, doc_id: str) -> dict | None:
        chunks = [c for c in self._get().store.all_chunks() if c.doc_id == doc_id]
        if not chunks:
            return None
        return {"doc_id": doc_id, "source": chunks[0].source, "text": reconstruct(chunks)}

    def fingerprint(self, questions: list[dict]) -> tuple[str, dict]:
        return fingerprint(self._settings, self._get(), questions)


class FrozenStimuli:
    frozen = True

    def __init__(self, path: str | Path, questions: list[dict]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != FROZEN_VERSION:
            raise RuntimeError(f"{path}: unsupported frozen-stimuli version")
        if data["questions_hash"] != questions_hash(questions):
            raise RuntimeError(
                f"{path} was made from a different question set; run freeze again after any "
                "change to the questions"
            )
        self._data = data

    def ask(self, qid: str, text: str, want_suggestions: bool = False) -> dict:
        try:
            return json.loads(json.dumps(self._data["answers"][qid][text]))  # a private copy
        except KeyError:
            raise KeyError(f"no saved answer for question {qid!r} and that wording") from None

    def documents(self) -> list[dict]:
        return [{k: d[k] for k in ("doc_id", "source", "chunks")} for d in self._data["documents"]]

    def document(self, doc_id: str) -> dict | None:
        for d in self._data["documents"]:
            if d["doc_id"] == doc_id:
                return {"doc_id": doc_id, "source": d["source"], "text": d["text"]}
        return None

    def fingerprint(self, questions: list[dict]) -> tuple[str, dict]:
        parts = dict(self._data["config_parts"])
        parts["frozen_at"] = self._data["created_at"]
        return self._data["config_hash"], parts


def build_stimuli(settings, get_pipeline, questions: list[dict]):
    if settings.study_frozen:
        return FrozenStimuli(settings.study_frozen, questions)
    return LiveStimuli(get_pipeline, settings)


def freeze(settings, get_pipeline, questions: list[dict]) -> dict:
    """Run every question (and every suggested rewording) once and return the saved stimuli.
    Fails loudly if any answer could not be written, so a broken run is never frozen."""
    live = LiveStimuli(get_pipeline, settings)
    answers: dict[str, dict] = {}
    problems = []
    for q in questions:
        first = live.ask(q["id"], q["question"], want_suggestions=True)
        answers[q["id"]] = {q["question"]: first}
        for s in first["suggestions"]:
            answers[q["id"]][s["text"]] = live.ask(q["id"], s["text"])
        for text, stim in answers[q["id"]].items():
            if stim["answer"] is None:
                problems.append(f"{q['id']} ({text[:40]!r}): {stim['answer_error']}")
    if problems:
        raise RuntimeError("could not write every answer:\n  " + "\n  ".join(problems))
    h, parts = live.fingerprint(questions)
    docs = []
    for d in live.documents():
        full = live.document(d["doc_id"])
        docs.append({**d, "text": full["text"]})
    return {
        "version": FROZEN_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": h,
        "config_parts": parts,
        "questions_hash": questions_hash(questions),
        "llm_model": settings.llm_model,
        "answers": answers,
        "documents": docs,
    }
