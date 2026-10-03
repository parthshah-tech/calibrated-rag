"""Phase 0/1 pipeline: ingest, then retrieve with dense / bm25 / hybrid (RRF) and attach the
pre-fusion confidence signal. Later phases add rewrite, multi-query, rerank, routing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .bm25_index import BM25Index
from .confidence import compute_confidence
from .config import Settings
from .hybrid_retrieval import rrf
from .ingestion import Ingestor
from .interfaces import Embedder, IngestResult, RetrievalResult, VectorStore
from .tracing import Tracer

MODES = ("dense", "bm25", "hybrid")


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings | None = None,
        parallel: bool = True,
    ) -> None:
        self.settings = settings or Settings()
        self.store, self.embedder = store, embedder
        self.bm25 = BM25Index()
        self.bm25.rebuild_from(store.all_chunks())  # restart-safe: sparse index from stored chunks
        self.parallel = parallel
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel else None
        s = self.settings
        self.ingestor = Ingestor(store, embedder, s.chunk_strategy, s.chunk_size, s.chunk_overlap)

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        return self.ingestor.ingest(filename, data, on_added=self.bm25.add)

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        tracer = Tracer()
        pool = self.settings.candidate_pool
        res = RetrievalResult(query=query, mode=mode)

        def run_dense():
            with tracer.span("dense"):
                emb = self.embedder.embed([query])[0]
                return self.store.query(emb, pool)

        def run_bm25():
            with tracer.span("bm25"):
                return self.bm25.search(query, pool)

        want_dense, want_bm25 = mode in ("dense", "hybrid"), mode in ("bm25", "hybrid")
        if want_dense and want_bm25 and self._pool:
            fd, fb = self._pool.submit(run_dense), self._pool.submit(run_bm25)
            res.dense, res.bm25 = fd.result(), fb.result()
        else:
            if want_dense:
                res.dense = run_dense()
            if want_bm25:
                res.bm25 = run_bm25()

        if want_dense and want_bm25:
            with tracer.span("confidence"):
                res.confidence = compute_confidence(
                    [h.chunk_id for h in res.dense],
                    [h.chunk_id for h in res.bm25],
                    metric=self.settings.confidence_metric,
                    k=self.settings.confidence_k,
                )
            with tracer.span("rrf"):
                res.fused = rrf(
                    [[h.chunk_id for h in res.dense], [h.chunk_id for h in res.bm25]],
                    k=self.settings.rrf_k,
                )[:k]
        else:
            res.fused = (res.dense or res.bm25)[:k]

        with tracer.span("fetch_chunks"):
            res.chunks = {c.id: c for c in self.store.get([h.chunk_id for h in res.fused])}
        res.trace = tracer.to_dict()
        return res
