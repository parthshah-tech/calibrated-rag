"""FastAPI app. Phase 0: upload + hybrid search. Generation, streaming and study mode come later."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .embedder import build_embedder
from .ingestion import SUPPORTED
from .llm import build_llm
from .pipeline import MODES, RAGPipeline
from .reranker import build_reranker
from .vector_store import ChromaVectorStore


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    mode: str = "hybrid"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # used by /ask
    expand: int = Field(default=0, ge=0, le=5)  # extra queries to search, used by /ask
    rerank: bool = False  # re-score the top candidates with the cross-encoder


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or Settings.from_env()
    store = ChromaVectorStore(f"{settings.data_dir}/chroma")
    embedder = build_embedder(settings.embedder, settings.embed_model)
    return RAGPipeline(
        store,
        embedder,
        settings,
        llm=build_llm(settings),
        reranker=build_reranker(settings.rerank_model),
    )


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="RAG Project")
    holder: dict[str, RAGPipeline | None] = {"p": pipeline}

    def get() -> RAGPipeline:
        if holder["p"] is None:  # lazy so importing the app does not load models
            holder["p"] = build_pipeline()
        return holder["p"]

    @app.get("/", include_in_schema=False)
    def index():
        page = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        if not page.exists():
            raise HTTPException(404, "frontend/index.html is missing")
        return FileResponse(page)

    @app.get("/health")
    def health():
        p = holder["p"]
        llm = p.llm is not None if p is not None else bool(Settings.from_env().llm_api_key)
        return {"status": "ok", "llm_configured": llm}

    @app.post("/ingest")
    async def ingest(file: UploadFile):
        name = file.filename or "upload"
        if "." + name.rsplit(".", 1)[-1].lower() not in SUPPORTED:
            raise HTTPException(415, f"unsupported file type; supported: {sorted(SUPPORTED)}")
        return asdict(get().ingest(name, await file.read()))

    def payload(r) -> dict:
        return {
            "query": r.query,
            "mode": r.mode,
            "confidence": asdict(r.confidence) if r.confidence else None,
            "results": [
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "source": r.chunks[h.chunk_id].source,
                    "text": r.chunks[h.chunk_id].text,
                    "start": r.chunks[h.chunk_id].start,
                    "end": r.chunks[h.chunk_id].end,
                }
                for h in r.fused
                if h.chunk_id in r.chunks
            ],
            "rewritten_query": r.rewritten_query,
            "expansions": r.expansions,
            "notes": r.notes,
            "reranked": r.reranked,
            "trace": r.trace,
        }

    def check_mode(req: SearchRequest) -> None:
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")

    @app.post("/search")
    def search(req: SearchRequest):
        check_mode(req)
        return payload(get().retrieve(req.query, req.k, req.mode, rerank=req.rerank))

    @app.post("/ask")
    def ask(req: SearchRequest):
        check_mode(req)
        history = [t.model_dump() for t in req.history]
        r, ans, err = get().answer(req.query, req.k, req.mode, history, req.expand, req.rerank)
        return {**payload(r), "answer": asdict(ans) if ans else None, "answer_error": err}

    return app


def __getattr__(name: str):  # `uvicorn backend.main:app` builds the app on first access
    if name == "app":
        return create_app()
    raise AttributeError(name)
