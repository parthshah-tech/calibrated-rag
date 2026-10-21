"""A fingerprint of everything that must stay frozen while participants are being run."""

from __future__ import annotations

import hashlib
import json

from ..confidence import DEFAULT_THRESHOLDS  # noqa: F401  (documented default)
from ..generation import SYSTEM_PROMPT
from .questions import questions_hash

REFORMULATION = {"n": 3, "margin": 0.05, "max_shown": 3, "only_for": ["Low", "Medium"]}
FIXED = {"k": 5, "mode": "hybrid", "rerank": False, "expand": 0, "history": False}


def fingerprint(settings, pipeline, questions: list[dict]) -> tuple[str, dict]:
    docs = sorted(d["doc_id"] for d in pipeline.store.documents())
    parts = {
        "retrieval": {
            k: getattr(settings, k)
            for k in (
                "chunk_strategy",
                "chunk_size",
                "chunk_overlap",
                "embedder",
                "embed_model",
                "candidate_pool",
                "rrf_k",
            )
        },
        "confidence": {
            "metric": settings.confidence_metric,
            "k": settings.confidence_k,
            "low": settings.confidence_low,
            "high": settings.confidence_high,
        },
        "llm_model": settings.llm_model,
        "prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "fixed": FIXED,
        "reformulation": REFORMULATION,
        "documents": docs,
        "questions": questions_hash(questions),
    }
    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], parts
