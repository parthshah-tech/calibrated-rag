"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    confidence_low: float = 0.3  # score below this is Low; fit on your own documents
    confidence_high: float = 0.6  # score at or above this is High
    candidate_pool: int = 50
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 20  # how many top candidates the cross-encoder re-scores
    # LLM (any OpenAI-compatible endpoint). The key is never printed: repr=False.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = field(default="", repr=False)
    llm_reasoning_effort: str | None = None  # e.g. "low" for gpt-oss models; unset otherwise
    # Study mode (see docs/study.md)
    study_mode: bool = False
    study_questions: str = "study/questions.json"
    study_db: str = "./data/study.sqlite"
    study_time_budget_min: int = 35
    study_frozen: str = ""  # path to saved stimuli; when set, no models or API key are needed
    study_admin_token: str = field(default="", repr=False)  # enables /study/admin/* downloads
    study_salt: str = field(default="", repr=False)  # mixed into hashed participant ids

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
            confidence_low=float(e.get("CONFIDENCE_LOW", cls.confidence_low)),
            confidence_high=float(e.get("CONFIDENCE_HIGH", cls.confidence_high)),
            candidate_pool=_int("CANDIDATE_POOL", cls.candidate_pool),
            rrf_k=_int("RRF_K", cls.rrf_k),
            rerank_model=e.get("RERANK_MODEL", cls.rerank_model),
            rerank_pool=_int("RERANK_POOL", cls.rerank_pool),
            llm_base_url=e.get("LLM_BASE_URL", cls.llm_base_url),
            llm_model=e.get("LLM_MODEL", cls.llm_model),
            llm_api_key=e.get("LLM_API_KEY") or e.get("GROQ_API_KEY", ""),
            llm_reasoning_effort=e.get("LLM_REASONING_EFFORT") or None,
            study_mode=e.get("STUDY_MODE", "0") == "1",
            study_questions=e.get("STUDY_QUESTIONS", cls.study_questions),
            study_db=e.get("STUDY_DB", cls.study_db),
            study_time_budget_min=_int("STUDY_TIME_BUDGET_MIN", cls.study_time_budget_min),
            study_frozen=e.get("STUDY_FROZEN", ""),
            study_admin_token=e.get("STUDY_ADMIN_TOKEN", ""),
            study_salt=e.get("STUDY_SALT", ""),
        )
