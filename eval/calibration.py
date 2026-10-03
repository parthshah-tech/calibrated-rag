"""Tools for asking: does the confidence score predict retrieval success?

AUROC (threshold-free), per-bucket success rates with Wilson intervals, tertile thresholds
fit on a dev split, and cross-validated isotonic calibration (ECE, Brier skill)."""

from __future__ import annotations

import numpy as np

from backend.confidence import bucket

NAN = float("nan")


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(scores, labels) -> float:
    """P(score of a random success > score of a random failure), ties count half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = _avg_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_ci(scores, labels, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        a = auroc(s[idx], y[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return auroc(s, y), NAN, NAN
    return auroc(s, y), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return NAN, NAN
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def fit_thresholds(scores) -> tuple[float, float]:
    """Equal-frequency (tertile) cutoffs: Low / Medium / High each hold about a third."""
    lo, hi = np.quantile(np.asarray(scores, dtype=float), [1 / 3, 2 / 3])
    return float(lo), float(hi)


def bucket_table(scores, labels, thresholds) -> list[dict]:
    rows = {name: [0, 0] for name in ("Low", "Medium", "High")}
    for s, y in zip(scores, labels, strict=True):
        b = bucket(float(s), thresholds)
        rows[b][0] += 1
        rows[b][1] += int(bool(y))
    out = []
    for name, (n, k) in rows.items():
        lo, hi = wilson(k, n)
        out.append(
            {
                "bucket": name,
                "n": n,
                "successes": k,
                "rate": k / n if n else NAN,
                "lo": lo,
                "hi": hi,
            }
        )
    return out


class IsotonicCalibrator:
    """Pool-adjacent-violators: monotone non-decreasing map from score to success probability."""

    def fit(self, scores, labels) -> IsotonicCalibrator:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        ux, inv = np.unique(s, return_inverse=True)  # pool tied scores first
        sums = np.bincount(inv, weights=y)
        cnt = np.bincount(inv).astype(float)
        blocks: list[list[float]] = []
        for i in range(len(ux)):
            blocks.append([sums[i], cnt[i], ux[i]])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                tot, w, right = blocks.pop()
                blocks[-1][0] += tot
                blocks[-1][1] += w
                blocks[-1][2] = right
        self._xs = np.array([b[2] for b in blocks])
        self._ys = np.array([b[0] / b[1] for b in blocks])
        return self

    def predict(self, scores) -> np.ndarray:
        idx = np.searchsorted(self._xs, np.asarray(scores, dtype=float), side="left")
        return self._ys[np.clip(idx, 0, len(self._xs) - 1)]


def ece(probs, labels, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def cv_isotonic(scores, labels, folds: int = 5, seed: int = 0) -> dict:
    """Held-out calibration quality. brier_skill > 0 means the calibrated score beats
    always predicting the base rate."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    folds = max(2, min(folds, len(s)))
    order = np.random.default_rng(seed).permutation(len(s))
    pred = np.empty(len(s))
    base = np.empty(len(s))
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        pred[test] = IsotonicCalibrator().fit(s[train], y[train]).predict(s[test])
        base[test] = y[train].mean()
    b_cal = float(np.mean((pred - y) ** 2))
    b_base = float(np.mean((base - y) ** 2))
    return {
        "ece": ece(pred, y),
        "brier_cal": b_cal,
        "brier_base": b_base,
        "brier_skill": 1 - b_cal / b_base if b_base > 0 else NAN,
    }


def transfer_calibration(train_scores, train_labels, test_scores, test_labels) -> dict:
    """Fit isotonic calibration on one dataset, evaluate on another. brier_skill is measured
    against predicting the training set's base rate; negative means the transfer hurt."""
    cal = IsotonicCalibrator().fit(train_scores, train_labels)
    pred = cal.predict(test_scores)
    y = np.asarray(test_labels, dtype=float)
    base = float(np.mean(train_labels))
    b_cal = float(np.mean((pred - y) ** 2))
    b_base = float(np.mean((base - y) ** 2))
    return {
        "ece": ece(pred, y),
        "brier_cal": b_cal,
        "brier_base": b_base,
        "brier_skill": 1 - b_cal / b_base if b_base > 0 else NAN,
    }


def auroc_diff_ci(
    scores_a, scores_b, labels, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
):
    """Paired bootstrap of AUROC(a) - AUROC(b) over the same queries."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        da, db = auroc(a[idx], y[idx]), auroc(b[idx], y[idx])
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    point = auroc(a, y) - auroc(b, y)
    if not diffs:
        return point, NAN, NAN
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
