import sys

import pytest
from conftest import DOC_A, DOC_B, DOC_C

from backend.config import Settings
from backend.embedder import HashEmbedder
from backend.pipeline import RAGPipeline
from backend.reranker import (
    CrossEncoderReranker,
    LazyReranker,
    LexicalOverlapReranker,
    build_reranker,
)
from backend.vector_store import InMemoryVectorStore


class Scripted:
    """Reranker that scores a passage by which marker word it contains."""

    def __init__(self, prefer: str):
        self.prefer, self.calls = prefer, []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [1.0 if self.prefer in p.lower() else 0.0 for p in passages]


def sources(res):
    return [res.chunks[h.chunk_id].source for h in res.fused]


def test_lexical_stand_in_scores_overlap():
    r = LexicalOverlapReranker()
    assert r.score("red apple", ["a red apple", "green pear", "red"]) == [1.0, 0.0, 0.5]
    assert r.score("", ["x"]) == [0.0]


class ReverseHead:
    """Scores later passages higher, so reranking exactly reverses the head."""

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [float(i) for i in range(len(passages))]


def test_reranker_reorders_the_head_and_keeps_the_tail(loaded):
    base = loaded.retrieve("bread flour sourdough oven", 10)
    loaded.reranker = ReverseHead()
    loaded.settings = Settings(**{**loaded.settings.__dict__, "rerank_pool": 4})
    res = loaded.retrieve("bread flour sourdough oven", 10, rerank=True)
    base_ids = [h.chunk_id for h in base.fused]
    got_ids = [h.chunk_id for h in res.fused]
    assert res.reranked and [h.chunk_id for h in res.pre_rerank] == base_ids[:4]
    assert got_ids[:4] == list(reversed(base_ids[:4]))  # head reordered by reranker scores
    assert got_ids[4:] == base_ids[4:]  # tail keeps its fused order
    assert set(got_ids) == set(base_ids)  # nothing lost or duplicated
    assert [h.score for h in res.fused[:4]] == [3.0, 2.0, 1.0, 0.0]  # reranker scores
    assert res.trace["counters"]["rerank_pairs"] == 4
    assert any(s["name"] == "rerank" and s["pairs"] == 4 for s in res.trace["spans"])
    assert len(loaded.reranker.calls[0][1]) == 4


def test_rerank_scores_replace_fusion_scores_and_ties_keep_order(loaded):
    loaded.reranker = Scripted(prefer="never-appears")  # all zeros: a pure tie
    plain = loaded.retrieve("fusion rankings", 5)
    res = loaded.retrieve("fusion rankings", 5, rerank=True)
    assert [h.chunk_id for h in res.fused] == [h.chunk_id for h in plain.fused]
    assert all(h.score == 0.0 for h in res.fused[: len(res.pre_rerank)])


def test_confidence_is_unchanged_by_reranking(loaded):
    base = loaded.retrieve("fusion rankings", 5).confidence
    loaded.reranker = LexicalOverlapReranker()
    assert loaded.retrieve("fusion rankings", 5, rerank=True).confidence == base


def test_rerank_without_a_reranker_degrades_with_a_note(loaded):
    plain = [h.chunk_id for h in loaded.retrieve("fusion", 5).fused]
    res = loaded.retrieve("fusion", 5, rerank=True)
    assert [h.chunk_id for h in res.fused] == plain and not res.reranked
    assert any("no reranker configured" in n for n in res.notes)


def test_a_failing_reranker_degrades_with_a_note(loaded):
    class Broken:
        def score(self, q, p):
            raise RuntimeError("sentence-transformers is not installed")

    loaded.reranker = Broken()
    plain = [h.chunk_id for h in loaded.retrieve("fusion", 5).fused]
    res = loaded.retrieve("fusion", 5, rerank=True)
    assert [h.chunk_id for h in res.fused] == plain and not res.reranked
    assert any("not installed" in n for n in res.notes)


def test_rerank_works_in_single_retriever_modes(loaded):
    loaded.reranker = LexicalOverlapReranker()
    res = loaded.retrieve("XJ-4471", 3, mode="bm25", rerank=True)
    assert res.reranked and sources(res)[0] == "b.txt"


def test_rerank_uses_the_rewritten_query(loaded):
    from backend.llm import FakeLLM

    loaded.llm = FakeLLM(["XJ-4471 product code"])
    loaded.reranker = Scripted(prefer="xj-4471")
    hist = [{"role": "user", "content": "codes?"}, {"role": "assistant", "content": "XJ-4471"}]
    res = loaded.retrieve("that one?", 3, history=hist, rerank=True)
    assert loaded.reranker.calls[0][0] == "XJ-4471 product code" and res.reranked


def test_lazy_reranker_loads_once_and_only_when_used():
    built = []

    def factory():
        built.append(1)
        return LexicalOverlapReranker()

    lazy = LazyReranker(factory)
    assert built == []  # constructing the app costs nothing
    lazy.score("a", ["a"])
    lazy.score("b", ["b"])
    assert built == [1]


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="uv sync --extra models"):
        CrossEncoderReranker()
    lazy = build_reranker("any")
    with pytest.raises(RuntimeError, match="uv sync --extra models"):
        lazy.score("q", ["p"])


def test_empty_corpus_rerank_is_harmless():
    p = RAGPipeline(
        InMemoryVectorStore(), HashEmbedder(), Settings(), reranker=LexicalOverlapReranker()
    )
    res = p.retrieve("anything", 3, rerank=True)
    assert res.fused == [] and not res.reranked


def test_fixture_docs_are_distinct():
    assert "XJ-4471" in DOC_B and "Sourdough" in DOC_C and "Fusion" in DOC_A


def test_the_pool_is_exactly_what_the_setting_says_even_when_k_is_larger(loaded):
    loaded.reranker = LexicalOverlapReranker()
    loaded.settings = Settings(**{**loaded.settings.__dict__, "rerank_pool": 2})
    res = loaded.retrieve("fusion rankings", 6, rerank=True)  # k=6 > pool=2
    assert len(res.pre_rerank) == 2 and res.trace["counters"]["rerank_pairs"] == 2
    assert len(res.fused) == 6
