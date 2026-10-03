import pytest

from backend.confidence import (
    bucket,
    compute_confidence,
    kendall_tau_intersection,
    overlap_at_k,
    rbo,
)

A = ["a", "b", "c", "d", "e"]


def test_overlap_basic():
    assert overlap_at_k(A, A, 5) == 1.0
    assert overlap_at_k(A, ["x", "y", "z", "w", "v"], 5) == 0.0
    assert overlap_at_k(["a", "b"], ["b", "c"], 2) == 0.5


def test_rbo_bounds_and_identity():
    assert rbo(A, A) == pytest.approx(1.0)
    assert rbo(A, ["x", "y", "z", "w", "v"]) == 0.0
    assert 0 < rbo(A, ["a", "x", "c", "y", "e"]) < 1


def test_rbo_weights_top_ranks_more():
    disagree_top = ["x", "b", "c", "d", "e"]
    disagree_bottom = ["a", "b", "c", "d", "x"]
    assert rbo(A, disagree_bottom) > rbo(A, disagree_top)


def test_rbo_symmetric_and_handles_unequal_lengths():
    b = ["c", "a", "q"]
    assert rbo(A, b) == pytest.approx(rbo(b, A))
    assert 0 < rbo(A, b) < 1
    assert rbo([], A) == 0.0


def test_rbo_order_matters_with_same_members():
    assert rbo(A, list(reversed(A))) < 1.0


def test_rbo_validates_p():
    with pytest.raises(ValueError):
        rbo(A, A, p=1.0)


def test_kendall_tau_intersection():
    assert kendall_tau_intersection(A, A) == (1.0, 5)
    assert kendall_tau_intersection(A, list(reversed(A))) == (-1.0, 5)
    assert kendall_tau_intersection(["a", "b"], ["c", "d"]) == (None, 0)
    assert kendall_tau_intersection(["a", "x"], ["a", "y"]) == (None, 1)
    tau, r = kendall_tau_intersection(["a", "b", "c"], ["b", "a", "c"])
    assert r == 3 and tau == pytest.approx(1 / 3)


def test_bucket_edges():
    assert bucket(0.0) == "Low"
    assert bucket(0.3) == "Medium"
    assert bucket(0.6) == "High"
    assert bucket(1.0) == "High"
    with pytest.raises(ValueError):
        bucket(0.5, (0.8, 0.2))


@pytest.mark.parametrize("metric", ["overlap", "rbo", "tau"])
def test_compute_confidence_extremes(metric):
    same = compute_confidence(A, A, metric=metric, k=5)
    none = compute_confidence(A, ["x", "y", "z", "w", "v"], metric=metric, k=5)
    assert same.bucket == "High" and same.score == pytest.approx(1.0)
    assert none.bucket == "Low" and none.score == 0.0 and none.shared == 0


def test_compute_confidence_unknown_metric():
    with pytest.raises(ValueError):
        compute_confidence(A, A, metric="nope")
