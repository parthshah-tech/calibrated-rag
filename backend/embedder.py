"""Embedders. HashEmbedder is deterministic and offline (tests, dev); SentenceTransformer
is the real one. CachedEmbedder adds an LRU cache with hit/miss counters."""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict

from .interfaces import Embedder

_TOKEN = re.compile(r"\w+")


class HashEmbedder:
    """Hashed bag-of-words, L2-normalised. Not semantic: it only captures lexical overlap,
    so it must never be used for reported retrieval numbers."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed; run `uv sync --extra models`"
            ) from e
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class CachedEmbedder:
    def __init__(self, inner: Embedder, max_size: int = 10_000) -> None:
        self.inner, self.max_size = inner, max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, vec in zip(missing, fresh, strict=True):
                self._cache[keys[i]] = vec
        out = []
        for k in keys:
            self._cache.move_to_end(k)
            out.append(self._cache[k])
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return out

    def clear(self) -> None:
        """Drop cached vectors (keeps hit/miss counters). Benchmarks call this so every mode
        pays the full query-encoding cost instead of reusing another mode's work."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def clear_cache(embedder: Embedder) -> None:
    """Clear the embedder's cache if it has one."""
    clear = getattr(embedder, "clear", None)
    if clear is not None:
        clear()


def build_embedder(kind: str, model: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "sentence-transformers":
        return CachedEmbedder(SentenceTransformerEmbedder(model))
    raise ValueError(f"unknown embedder: {kind!r}")
