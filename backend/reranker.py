"""Cross-encoder reranking: score each (query, passage) pair jointly. More accurate than the
bi-encoder used for first-stage retrieval, but one model pass per pair, so it only reorders a
small head of the candidate list."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed; run `uv sync --extra models`"
            ) from e
        self._model = CrossEncoder(model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]


class LazyReranker:
    """Loads the model on first use, so starting the app (or never reranking) costs nothing."""

    def __init__(self, factory: Callable[[], Reranker]) -> None:
        self._factory = factory
        self._inner: Reranker | None = None
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            if self._inner is None:
                self._inner = self._factory()
        return self._inner.score(query, passages)


class LexicalOverlapReranker:
    """Deterministic stand-in for tests: fraction of query words found in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(re.findall(r"\w+", query.lower()))
        out = []
        for p in passages:
            words = set(re.findall(r"\w+", p.lower()))
            out.append(len(q & words) / len(q) if q else 0.0)
        return out


def build_reranker(model: str) -> Reranker:
    return LazyReranker(lambda: CrossEncoderReranker(model))
