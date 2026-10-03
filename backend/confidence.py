"""Retrieval-agreement confidence signal: how much do dense and BM25 agree on what is relevant?

Pure functions over two ranked lists of ids, computed BEFORE fusion on the original query.
Thresholds here are placeholders until eval/validate_signal.py fits them on a dev split."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THRESHOLDS = (0.3, 0.6)  # (low/medium, medium/high), to be fit on a dev split


def overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    """|top_k(a) intersect top_k(b)| / k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(a[:k]) & set(b[:k])) / k


def rbo(a: list[str], b: list[str], p: float = 0.9, k: int | None = None) -> float:
    """Truncated Rank-Biased Overlap, normalised to [0, 1].

    Agreement at depth d is |a[:d] intersect b[:d]| / d; depths are weighted (1-p) * p^(d-1),
    so top ranks matter most. Dividing by (1 - p^k) rescales the truncated sum so identical
    lists score exactly 1. Works for lists of different lengths and different members."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    depth = k if k is not None else max(len(a), len(b))
    if depth == 0 or not a or not b:
        return 0.0
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    overlap = 0  # |a[:d] intersect b[:d]|, maintained incrementally
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(a):
            x = a[d - 1]
            if x not in seen_a:
                seen_a.add(x)
                overlap += x in seen_b
        if d <= len(b):
            y = b[d - 1]
            if y not in seen_b:
                seen_b.add(y)
                overlap += y in seen_a
        total += (p ** (d - 1)) * overlap / d
    return (1 - p) * total / (1 - p**depth)


def kendall_tau_intersection(a: list[str], b: list[str]) -> tuple[float | None, int]:
    """Kendall's tau over items present in both lists. Returns (tau, intersection_size);
    tau is None when fewer than 2 items are shared (undefined)."""
    shared = [x for x in a if x in set(b)]
    r = len(shared)
    if r < 2:
        return None, r
    pos_b = {x: i for i, x in enumerate(b)}
    seq = [pos_b[x] for x in shared]  # ranks in b, listed in a's order
    concordant = discordant = 0
    for i in range(r):
        for j in range(i + 1, r):
            if seq[i] < seq[j]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (r * (r - 1) / 2), r


def bucket(score: float, thresholds: tuple[float, float] = DEFAULT_THRESHOLDS) -> str:
    low, high = thresholds
    if not 0 <= low <= high <= 1:
        raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
    if score >= high:
        return "High"
    if score >= low:
        return "Medium"
    return "Low"


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    bucket: str
    metric: str
    overlap: float
    shared: int  # |dense intersect bm25| within top-k, always reported for transparency


def compute_confidence(
    dense_ids: list[str],
    bm25_ids: list[str],
    metric: str = "rbo",
    k: int = 10,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
) -> ConfidenceResult:
    a, b = dense_ids[:k], bm25_ids[:k]
    ov = overlap_at_k(a, b, k)
    shared = len(set(a) & set(b))
    if metric == "overlap":
        score = ov
    elif metric == "rbo":
        score = rbo(a, b, k=k)
    elif metric == "tau":
        # PROVISIONAL: tau is undefined with <2 shared items and ignores non-shared ones, so it
        # is mapped to [0, 1] and scaled by the shared fraction. To be ablated in validation.
        tau, r = kendall_tau_intersection(a, b)
        score = 0.0 if tau is None else ((tau + 1) / 2) * (r / k)
    else:
        raise ValueError(f"unknown confidence metric: {metric!r}")
    return ConfidenceResult(score, bucket(score, thresholds), metric, ov, shared)
