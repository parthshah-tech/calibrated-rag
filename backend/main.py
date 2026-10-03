"""FastAPI app. Phase 0: upload + hybrid search. Generation, streaming and study mode come later."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .embedder import build_embedder
from .ingestion import SUPPORTED
from .pipeline import MODES, RAGPipeline
from .vector_store import ChromaVectorStore


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    mode: str = "hybrid"


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or Settings.from_env()
    store = ChromaVectorStore(f"{settings.data_dir}/chroma")
    embedder = build_embedder(settings.embedder, settings.embed_model)
    return RAGPipeline(store, embedder, settings)


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
        return {"status": "ok"}

    @app.post("/ingest")
    async def ingest(file: UploadFile):
        name = file.filename or "upload"
        if "." + name.rsplit(".", 1)[-1].lower() not in SUPPORTED:
            raise HTTPException(415, f"unsupported file type; supported: {sorted(SUPPORTED)}")
        return asdict(get().ingest(name, await file.read()))

    @app.post("/search")
    def search(req: SearchRequest):
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        r = get().retrieve(req.query, req.k, req.mode)
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
            "trace": r.trace,
        }

    return app


def __getattr__(name: str):  # `uvicorn backend.main:app` builds the app on first access
    if name == "app":
        return create_app()
    raise AttributeError(name)
