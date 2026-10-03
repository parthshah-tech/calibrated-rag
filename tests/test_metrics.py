import math

import pytest

from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k

RELS = {"b": 1, "c": 1}


def test_recall_hit_mrr():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, RELS, 2) == 0.5
    assert recall_at_k(ranked, RELS, 3) == 1.0
    assert hit_at_k(ranked, RELS, 1) == 0.0
    assert hit_at_k(ranked, RELS, 2) == 1.0
    assert mrr_at_k(ranked, RELS, 10) == 0.5
    assert mrr_at_k(["x", "y"], RELS, 10) == 0.0


def test_ndcg_hand_computed():
    # DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], RELS, 3) == pytest.approx(expected)
    assert ndcg_at_k(["b", "c"], RELS, 10) == pytest.approx(1.0)


def test_ndcg_uses_graded_gain_and_cutoff():
    rels = {"a": 2, "b": 1}
    assert ndcg_at_k(["b", "a"], rels, 2) < 1.0
    assert ndcg_at_k(["a", "b"], rels, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], rels, 1) == 0.0  # relevant doc beyond the cutoff


def test_no_relevant_documents():
    assert recall_at_k(["a"], {}, 5) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 5) == 0.0
    assert hit_at_k(["a"], {"a": 0}, 5) == 0.0
