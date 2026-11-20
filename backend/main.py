"""FastAPI app: chat interface, document upload, search, cited answers, and study mode."""

from __future__ import annotations

import re
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import Settings
from .embedder import build_embedder
from .ingestion import SUPPORTED
from .llm import build_llm
from .pipeline import MODES, RAGPipeline
from .reranker import build_reranker
from .vector_store import ChromaVectorStore
from .workspaces import VisitorWorkspaces


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    k: int = Field(default=5, ge=1, le=50)
    mode: str = "hybrid"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # used by /ask
    expand: int = Field(default=0, ge=0, le=5)  # extra queries to search, used by /ask
    rerank: bool = False  # re-score the top candidates with the cross-encoder
    doc_ids: list[str] | None = Field(default=None, max_length=500)  # None means all documents


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


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    holder: dict[str, RAGPipeline | None] = {"p": pipeline}
    lock = threading.RLock()
    visitors = VisitorWorkspaces(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        visitors.close()
        if holder["p"] is not None:
            holder["p"].close()
        if hasattr(app.state, "live_store"):
            app.state.live_store.close()

    app = FastAPI(title="RAG Project", lifespan=lifespan)

    @app.middleware("http")
    async def visitor_cookie(request: Request, call_next):
        isolated = settings.isolate_visitors and not settings.study_mode
        if isolated and request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Use this app from its own page."}, status_code=403)
        token = request.cookies.get("rag_visitor", "")
        fresh = not re.fullmatch(r"[0-9a-f]{64}", token)
        request.state.visitor = secrets.token_hex(32) if fresh else token
        response = await call_next(request)
        if isolated:
            response.headers["Cache-Control"] = "no-store"
            if fresh:
                response.set_cookie(
                    "rag_visitor",
                    request.state.visitor,
                    httponly=True,
                    secure=request.url.scheme == "https",
                    samesite="lax",
                    max_age=60 * 60 * 24 * 30,
                )
        return response

    def get(request: Request | None = None) -> RAGPipeline:
        if settings.isolate_visitors and not settings.study_mode:
            return visitors.get(request.state.visitor)
        if holder["p"] is None:  # lazy so importing the app does not load models
            holder["p"] = build_pipeline(settings)
        return holder["p"]

    def not_in_study() -> None:
        if settings.study_mode:  # participants only use the study endpoints
            raise HTTPException(403, "disabled in study mode")

    @app.get("/", include_in_schema=False)
    def index():
        if settings.study_mode:
            return RedirectResponse("/study")
        page = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        if not page.exists():
            raise HTTPException(404, "frontend/index.html is missing")
        return FileResponse(page)

    @app.get("/health")
    def health():
        p = holder["p"]
        llm = p.llm is not None if p is not None else bool(settings.llm_api_key)
        return {"status": "ok", "llm_configured": llm}

    @app.get("/app.css", include_in_schema=False)
    def app_styles():
        return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "app.css")

    @app.get("/app-config")
    def app_config():
        not_in_study()
        return {
            "isolated": settings.isolate_visitors,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_documents": settings.max_documents,
        }

    @app.get("/documents")
    def documents(request: Request):
        not_in_study()
        with lock:
            return get(request).store.documents()

    @app.get("/documents/{doc_id}")
    def document(doc_id: str, request: Request):
        from .study.viewer import reconstruct

        not_in_study()
        with lock:
            chunks = [c for c in get(request).store.all_chunks() if c.doc_id == doc_id]
            if not chunks:
                raise HTTPException(404, "Document not found.")
            return {"doc_id": doc_id, "source": chunks[0].source, "text": reconstruct(chunks)}

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request):
        not_in_study()
        with lock:
            p = get(request)
            if not p.store.has_doc(doc_id):
                raise HTTPException(404, "Document not found.")
            p.delete_document(doc_id)
        return {"ok": True}

    @app.post("/ingest")
    def ingest(file: UploadFile, request: Request):
        not_in_study()
        name = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1][:255]
        if "." + name.rsplit(".", 1)[-1].lower() not in SUPPORTED:
            raise HTTPException(415, f"unsupported file type; supported: {sorted(SUPPORTED)}")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "File exceeds the upload limit.")
        if not data:
            raise HTTPException(422, "This file is empty.")
        with lock:
            p = get(request)
            from .ingestion import content_hash

            if len(p.store.documents()) >= settings.max_documents and not p.store.has_doc(
                content_hash(data)[:16]
            ):
                raise HTTPException(413, "Document limit reached. Remove a document first.")
            try:
                result = p.ingest(name, data)
            except (ValueError, OSError) as exc:
                raise HTTPException(
                    422, "Could not read this document. Check the file format."
                ) from exc
            if not result.n_chunks and not result.skipped:
                raise HTTPException(
                    422, "No readable text found. Scanned PDFs need OCR before uploading."
                )
            return asdict(result)

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
    def search(req: SearchRequest, request: Request):
        not_in_study()
        check_mode(req)
        with lock:
            r = get(request).retrieve(
                req.query, req.k, req.mode, rerank=req.rerank, doc_ids=req.doc_ids
            )
            return payload(r)

    @app.post("/ask")
    def ask(req: SearchRequest, request: Request):
        not_in_study()
        check_mode(req)
        history = [t.model_dump() for t in req.history]
        with lock:
            r, ans, err = get(request).answer(
                req.query, req.k, req.mode, history, req.expand, req.rerank, req.doc_ids
            )
            return {**payload(r), "answer": asdict(ans) if ans else None, "answer_error": err}

    if settings.study_mode and settings.study_flow == "live":
        from .study.live_api import register_live_study

        register_live_study(app, settings, get, lock)
    elif settings.study_mode:
        from .study.api import register_study
        from .study.questions import load_questions
        from .study.stimuli import build_stimuli

        questions = load_questions(settings.study_questions)
        register_study(app, settings, build_stimuli(settings, get, questions), questions)
    return app


def __getattr__(name: str):  # `uvicorn backend.main:app` builds the app on first access
    if name == "app":
        return create_app()
    raise AttributeError(name)
