"""Reciprocal Rank Fusion. Fuses *rankings*, not scores: dense cosine and BM25 scores live on
incomparable scales, so only rank positions are used."""

from __future__ import annotations

from .interfaces import Hit


def rrf(rankings: list[list[str]], k: int = 60) -> list[Hit]:
    """score(d) = sum over rankings of 1 / (k + rank_d), rank starting at 1.

    Ties are broken by best (lowest) rank achieved in any list, then by id, so output is
    deterministic."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, cid in enumerate(ranking, start=1):
            if cid in seen:  # a repeated id inside one list counts once, at its first rank
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
    ordered = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))
    return [Hit(c, scores[c]) for c in ordered]
