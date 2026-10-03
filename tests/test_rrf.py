import pytest

from backend.hybrid_retrieval import rrf


def test_formula_two_lists():
    out = {h.chunk_id: h.score for h in rrf([["a", "b"], ["b", "c"]], k=60)}
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["c"] == pytest.approx(1 / 62)


def test_item_in_both_lists_beats_single_list_leader():
    fused = rrf([["a", "b", "c"], ["x", "b", "y"]], k=60)
    assert fused[0].chunk_id == "b"


def test_deterministic_tie_break():
    # a and x both have score 1/61; tie broken by best rank (equal) then id
    fused = rrf([["a"], ["x"]], k=60)
    assert [h.chunk_id for h in fused] == ["a", "x"]


def test_single_list_preserves_order():
    assert [h.chunk_id for h in rrf([["c", "a", "b"]])] == ["c", "a", "b"]


def test_empty_and_duplicates():
    assert rrf([]) == []
    assert rrf([[], []]) == []
    out = {h.chunk_id: h.score for h in rrf([["a", "a", "b"]], k=60)}
    assert out["a"] == pytest.approx(1 / 61)  # duplicate counted once
    assert out["b"] == pytest.approx(1 / 63)  # rank is raw list position (3), not deduped


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)
