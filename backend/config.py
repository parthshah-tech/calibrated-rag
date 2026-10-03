"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: str = "./data"
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedder: str = "sentence-transformers"
    embed_model: str = "all-MiniLM-L6-v2"
    confidence_metric: str = "rbo"
    confidence_k: int = 10  # depth of the top-k lists the signal compares; independent of result k
    candidate_pool: int = 50
    rrf_k: int = 60
    study_mode: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ
        return cls(
            data_dir=e.get("DATA_DIR", cls.data_dir),
            chunk_strategy=e.get("CHUNK_STRATEGY", cls.chunk_strategy),
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap),
            embedder=e.get("EMBEDDER", cls.embedder),
            embed_model=e.get("EMBED_MODEL", cls.embed_model),
            confidence_metric=e.get("CONFIDENCE_METRIC", cls.confidence_metric),
            confidence_k=_int("CONFIDENCE_K", cls.confidence_k),
            candidate_pool=_int("CANDIDATE_POOL", cls.candidate_pool),
            rrf_k=_int("RRF_K", cls.rrf_k),
            study_mode=e.get("STUDY_MODE", "0") == "1",
        )
