"""Bootstrap statistics. Seeds are fixed so every reported interval is reproducible."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi): percentile bootstrap interval for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    return (
        float(x.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_bootstrap(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison of per-query scores a vs b (same queries, same order).
    `significant` means the interval for mean(a - b) excludes zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("no paired observations")
    mean, lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    return {"mean_diff": mean, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0, "n": d.size}


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")
