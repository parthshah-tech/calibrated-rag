"""Ranking metrics over a ranked list of ids and graded relevance judgments {id: grade}.
Relevant means grade > 0. nDCG uses linear gain and a log2 discount (trec_eval convention)."""

from __future__ import annotations

import math


def _relevant(rels: dict[str, int]) -> set[str]:
    return {d for d, g in rels.items() if g > 0}


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    rel = _relevant(rels)
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def hit_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return 1.0 if _relevant(rels) & set(ranked[:k]) else 0.0


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    rel = _relevant(rels)
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(max(rels.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
