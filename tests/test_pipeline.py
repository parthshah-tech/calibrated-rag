from conftest import DOC_A, DOC_B, DOC_C

from backend.config import Settings
from backend.embedder import HashEmbedder
from backend.pipeline import RAGPipeline
from backend.vector_store import ChromaVectorStore, InMemoryVectorStore


def top_source(res):
    return res.chunks[res.fused[0].chunk_id].source


def test_hybrid_finds_the_right_document(loaded):
    assert top_source(loaded.retrieve("how does reciprocal rank fusion work", 3)) == "a.txt"
    assert top_source(loaded.retrieve("sourdough starter flour", 3)) == "c.txt"


def test_bm25_catches_exact_code_dense_alone_can_miss(loaded):
    res = loaded.retrieve("XJ-4471", 3, mode="bm25")
    assert top_source(res) == "b.txt"


def test_modes_populate_expected_lists(loaded):
    d = loaded.retrieve("fusion", 3, mode="dense")
    b = loaded.retrieve("fusion", 3, mode="bm25")
    h = loaded.retrieve("fusion", 3, mode="hybrid")
    assert d.dense and not d.bm25 and d.confidence is None
    assert b.bm25 and not b.dense and b.confidence is None
    assert h.dense and h.bm25 and h.confidence is not None


def test_confidence_uses_prefusion_lists_and_reports_bucket(loaded):
    res = loaded.retrieve("reciprocal rank fusion rankings", 5)
    assert res.confidence.bucket in {"Low", "Medium", "High"}
    assert 0.0 <= res.confidence.score <= 1.0


def test_trace_has_stage_timings(loaded):
    names = {s["name"] for s in loaded.retrieve("fusion", 3).trace["spans"]}
    assert {"dense", "bm25", "confidence", "rrf", "fetch_chunks"} <= names


def test_parallel_and_sequential_agree(loaded):
    seq = RAGPipeline(loaded.store, loaded.embedder, loaded.settings, parallel=False)
    q = "dense and sparse scores"
    a = [h.chunk_id for h in loaded.retrieve(q, 5).fused]
    b = [h.chunk_id for h in seq.retrieve(q, 5).fused]
    assert a == b


def test_empty_corpus_returns_nothing():
    p = RAGPipeline(InMemoryVectorStore(), HashEmbedder(), Settings())
    r = p.retrieve("anything", 3)
    assert r.fused == [] and r.chunks == {}


def test_invalid_mode(loaded):
    import pytest

    with pytest.raises(ValueError):
        loaded.retrieve("x", 3, mode="nope")


def test_chroma_store_matches_in_memory_and_survives_restart(tmp_path):
    s = Settings(chunk_size=120, chunk_overlap=20)
    mem = RAGPipeline(InMemoryVectorStore(), HashEmbedder(), s)
    chroma = RAGPipeline(ChromaVectorStore(str(tmp_path / "c")), HashEmbedder(), s)
    for p in (mem, chroma):
        for name, text in [("a.txt", DOC_A), ("b.txt", DOC_B), ("c.txt", DOC_C)]:
            p.ingest(name, text.encode())
    q = "dense embeddings paraphrase synonym"
    assert [h.chunk_id for h in mem.retrieve(q, 3, "dense").fused] == [
        h.chunk_id for h in chroma.retrieve(q, 3, "dense").fused
    ]
    # restart: new pipeline over the same directory rebuilds BM25 from stored chunks
    again = RAGPipeline(ChromaVectorStore(str(tmp_path / "c")), HashEmbedder(), s)
    assert again.ingest("a.txt", DOC_A.encode()).skipped
    assert top_source(again.retrieve("XJ-4471", 3, "bm25")) == "b.txt"
    assert chroma.store.count() == again.store.count()


def _two_docs(pipe):
    a = pipe.ingest("a.txt", DOC_A.encode())
    b = pipe.ingest("b.txt", DOC_B.encode())
    return a.doc_id, b.doc_id


def test_documents_lists_each_source_once_with_chunk_counts(pipeline):
    a, b = _two_docs(pipeline)
    docs = {d["doc_id"]: d for d in pipeline.store.documents()}
    assert set(docs) == {a, b} and docs[a]["source"] == "a.txt" and docs[b]["chunks"] >= 1
    assert sum(d["chunks"] for d in docs.values()) == pipeline.store.count()


def test_doc_filter_restricts_dense_bm25_and_hybrid(pipeline):
    a, b = _two_docs(pipeline)
    for mode in ("dense", "bm25", "hybrid"):
        only_b = pipeline.retrieve("fusion rankings keyword", 5, mode, doc_ids=[b])
        assert only_b.chunks and {c.source for c in only_b.chunks.values()} == {"b.txt"}
        none = pipeline.retrieve("fusion rankings keyword", 5, mode, doc_ids=[])
        assert none.fused == []
        every = pipeline.retrieve("fusion rankings keyword", 5, mode, doc_ids=None)
        assert {c.source for c in every.chunks.values()} == {"a.txt", "b.txt"} or mode == "bm25"


def test_doc_filter_also_applies_to_expansion_queries(pipeline):
    from backend.llm import FakeLLM

    a, b = _two_docs(pipeline)
    pipeline.llm = FakeLLM(["sparse retrieval keyword matches"])
    res = pipeline.retrieve("fusion", 5, expand=1, doc_ids=[a])
    assert {c.source for c in res.chunks.values()} == {"a.txt"}


def test_chroma_doc_filter_matches_memory(tmp_path):
    from backend.config import Settings
    from backend.embedder import HashEmbedder
    from backend.pipeline import RAGPipeline
    from backend.vector_store import ChromaVectorStore

    pipe = RAGPipeline(
        ChromaVectorStore(str(tmp_path / "c")),
        HashEmbedder(),
        Settings(chunk_size=120, chunk_overlap=20),
    )
    a, b = _two_docs(pipe)
    res = pipe.retrieve("dense embeddings paraphrase", 5, "dense", doc_ids=[b])
    assert res.chunks and {c.source for c in res.chunks.values()} == {"b.txt"}
    assert pipe.retrieve("anything", 5, "dense", doc_ids=[]).fused == []
    assert {d["source"] for d in pipe.store.documents()} == {"a.txt", "b.txt"}
