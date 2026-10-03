"""BM25 sparse index over rank_bm25.

Deliberately the naive baseline: it rebuilds from scratch on every add (O(N*L) per add),
scores every document per query term (O(q*N)), and ranks with a full sort (O(N log N)).
benchmarks/ measures this; docs/complexity.md lists the planned fixes."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .interfaces import Chunk, Hit

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._ids.append(c.id)
            self._tokens.append(tokenize(c.text))
        self._rebuild()

    def rebuild_from(self, chunks: list[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        self._tokens = [tokenize(c.text) for c in chunks]
        self._rebuild()

    def _rebuild(self) -> None:
        usable = any(self._tokens)
        self._bm25 = BM25Okapi(self._tokens) if usable else None

    def search(self, query: str, k: int) -> list[Hit]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        # descending score, ties broken by insertion order for determinism
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [Hit(self._ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]

    def __len__(self) -> int:
        return len(self._ids)
