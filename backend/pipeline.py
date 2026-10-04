"""Phase 0/1 pipeline: ingest, then retrieve with dense / bm25 / hybrid (RRF) and attach the
pre-fusion confidence signal. Later phases add rewrite, multi-query, rerank, routing."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .bm25_index import BM25Index
from .confidence import compute_confidence
from .config import Settings
from .generation import Answer, generate_answer
from .history_aware import rewrite_query, trim_history
from .hybrid_retrieval import rrf
from .ingestion import Ingestor
from .interfaces import Embedder, Hit, IngestResult, RetrievalResult, VectorStore
from .llm import LLMClient, LLMError
from .query_expansion import expand_query
from .reranker import Reranker
from .tracing import Tracer

MODES = ("dense", "bm25", "hybrid")


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings | None = None,
        parallel: bool = True,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm = llm
        self.reranker = reranker
        self.store, self.embedder = store, embedder
        self.bm25 = BM25Index()
        self.bm25.rebuild_from(store.all_chunks())  # restart-safe: sparse index from stored chunks
        self.parallel = parallel
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel else None
        s = self.settings
        self.ingestor = Ingestor(store, embedder, s.chunk_strategy, s.chunk_size, s.chunk_overlap)

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        return self.ingestor.ingest(filename, data, on_added=self.bm25.add)

    def _first_pass(self, tracer: Tracer, query: str, dense: bool, bm25: bool, label: str = ""):
        """Dense and BM25 candidate lists for one query (in parallel when both are wanted)."""
        pool = self.settings.candidate_pool

        def run_dense():
            with tracer.span("dense" + label):
                emb = self.embedder.embed([query])[0]
                return self.store.query(emb, pool)

        def run_bm25():
            with tracer.span("bm25" + label):
                return self.bm25.search(query, pool)

        if dense and bm25 and self._pool:
            fd, fb = self._pool.submit(run_dense), self._pool.submit(run_bm25)
            return fd.result(), fb.result()
        return (run_dense() if dense else []), (run_bm25() if bm25 else [])

    @staticmethod
    def _llm_step(tracer: Tracer, res: RetrievalResult, name: str, step) -> None:
        tracer.count("llm_calls")
        tracer.count("llm_cache_hits", int(step.cached))
        if step.error:
            res.notes.append(f"{name} skipped: {step.error}")

    def _rerank(
        self, tracer: Tracer, res: RetrievalResult, query: str, ranked: list[Hit], k: int
    ) -> list[Hit]:
        """Re-score the first `rerank_pool` candidates; the tail keeps its fused order, so recall@k
        beyond the head is unchanged. Falls back to the original ranking with a note on failure."""
        if self.reranker is None:
            res.notes.append("rerank skipped: no reranker configured")
            return ranked
        head, tail = ranked[: self.settings.rerank_pool], ranked[self.settings.rerank_pool :]
        chunks = {c.id: c for c in self.store.get([h.chunk_id for h in head])}
        head = [h for h in head if h.chunk_id in chunks]
        if not head:
            return ranked
        try:
            with tracer.span("rerank", pairs=len(head)):
                scores = self.reranker.score(query, [chunks[h.chunk_id].text for h in head])
        except RuntimeError as e:
            res.notes.append(f"rerank skipped: {e}")
            return ranked
        tracer.count("rerank_pairs", len(head))
        res.pre_rerank, res.reranked = list(head), True
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))  # stable on ties
        return [Hit(head[i].chunk_id, float(scores[i])) for i in order] + tail

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        history: list[dict] | None = None,
        expand: int = 0,
        rerank: bool = False,
    ) -> RetrievalResult:
        """history: earlier turns, used to rewrite a follow-up into a standalone query.
        expand: number of extra LLM-written queries to search too (hybrid mode only).
        Both need an LLM and degrade silently to plain retrieval if it is missing or fails.
        rerank: re-score the top candidates with the cross-encoder (degrades with a note).
        The confidence signal always uses the first-pass lists of the (rewritten) question."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        tracer = Tracer()
        res = RetrievalResult(query=query, mode=mode)
        search_query = query
        if history and self.llm is not None:
            with tracer.span("rewrite"):
                rw = rewrite_query(self.llm, query, history)
            self._llm_step(tracer, res, "rewrite", rw)
            if rw.rewritten:
                search_query, res.rewritten_query = rw.query, rw.query

        want_dense, want_bm25 = mode in ("dense", "hybrid"), mode in ("bm25", "hybrid")
        res.dense, res.bm25 = self._first_pass(tracer, search_query, want_dense, want_bm25)

        if want_dense and want_bm25:
            with tracer.span("confidence"):
                res.confidence = compute_confidence(
                    [h.chunk_id for h in res.dense],
                    [h.chunk_id for h in res.bm25],
                    metric=self.settings.confidence_metric,
                    k=self.settings.confidence_k,
                )
            rankings = [[h.chunk_id for h in res.dense], [h.chunk_id for h in res.bm25]]
            if expand and self.llm is not None:
                with tracer.span("expand"):
                    ex = expand_query(self.llm, search_query, expand)
                self._llm_step(tracer, res, "expansion", ex)
                res.expansions = ex.queries
                for i, q in enumerate(ex.queries, start=1):
                    d, b = self._first_pass(tracer, q, True, True, label=f"#{i}")
                    rankings += [[h.chunk_id for h in d], [h.chunk_id for h in b]]
            with tracer.span("rrf"):
                ranked = rrf(rankings, k=self.settings.rrf_k)
        else:
            ranked = res.dense or res.bm25
        if rerank:
            ranked = self._rerank(tracer, res, search_query, ranked, k)
        res.fused = ranked[:k]

        with tracer.span("fetch_chunks"):
            res.chunks = {c.id: c for c in self.store.get([h.chunk_id for h in res.fused])}
        res.trace = tracer.to_dict()
        return res

    def answer(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        history: list[dict] | None = None,
        expand: int = 0,
        rerank: bool = False,
    ) -> tuple[RetrievalResult, Answer | None, str | None]:
        """Retrieve, then write a grounded answer. Retrieval always survives an LLM failure:
        the third item is an error message when no answer could be produced."""
        res = self.retrieve(query, k, mode, history, expand, rerank)
        if self.llm is None:
            return res, None, "Generation is off: set LLM_API_KEY (or GROQ_API_KEY) in .env."
        chunks = [res.chunks[h.chunk_id] for h in res.fused if h.chunk_id in res.chunks]
        t0 = time.perf_counter()
        try:
            ans = generate_answer(self.llm, query, chunks, history=trim_history(history))
        except LLMError as e:
            return res, None, str(e)
        ms = round((time.perf_counter() - t0) * 1000, 3)
        res.trace["spans"].append({"name": "generate", "ms": ms})
        res.trace["total_ms"] = round(res.trace["total_ms"] + ms, 3)
        counters = res.trace["counters"]
        if ans.model != "none":
            counters["llm_calls"] = counters.get("llm_calls", 0) + 1
            counters["llm_cache_hits"] = counters.get("llm_cache_hits", 0) + int(ans.cached)
        return res, ans, None
