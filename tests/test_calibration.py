import numpy as np
import pytest

from eval.calibration import (
    IsotonicCalibrator,
    auroc,
    bucket_table,
    cv_isotonic,
    ece,
    fit_thresholds,
    wilson,
)


def test_auroc_extremes_and_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 60).astype(float)  # many ties
    y = rng.integers(0, 2, 60).astype(bool)
    pos, neg = s[y], s[~y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    assert auroc(s, y) == pytest.approx(wins / (len(pos) * len(neg)))


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    assert all(np.isnan(wilson(0, 0)))
    lo0, hi0 = wilson(0, 10)
    assert lo0 == pytest.approx(0, abs=1e-9) and hi0 > 0


def test_tertile_thresholds_and_bucket_table():
    scores = np.linspace(0, 1, 90)
    th = fit_thresholds(scores)
    assert th[0] < th[1]
    labels = scores > 0.5  # success only for high scores
    rows = {r["bucket"]: r for r in bucket_table(scores, labels, th)}
    assert rows["Low"]["rate"] == 0.0 and rows["High"]["rate"] == 1.0
    assert sum(r["n"] for r in rows.values()) == 90


def test_isotonic_is_monotone_and_fits_a_step():
    cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 0, 1, 0, 1, 1])
    pred = cal.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.95, 0.0])
    assert all(a <= b for a, b in zip(pred[:6], pred[1:6], strict=False))
    assert pred[0] == 0.0 and pred[5] == 1.0
    assert pred[6] == 1.0 and pred[7] == 0.0  # clamped outside the fitted range
    # pooled violators: (0.3 -> 1, 0.4 -> 0) average to 0.5
    assert pred[2] == pytest.approx(0.5) and pred[3] == pytest.approx(0.5)


def test_isotonic_pools_tied_scores():
    cal = IsotonicCalibrator().fit([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0])
    assert cal.predict([0.5])[0] == pytest.approx(0.25)


def test_ece_perfect_and_bad():
    assert ece([0.0, 1.0, 1.0, 0.0], [0, 1, 1, 0]) == 0.0
    assert ece([0.9, 0.9, 0.9, 0.9], [0, 0, 0, 0]) == pytest.approx(0.9)


def test_cv_isotonic_has_skill_on_informative_scores_and_none_on_noise():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 600)
    informative = rng.uniform(0, 1, 600) < s
    noise = rng.uniform(0, 1, 600) < 0.5
    assert cv_isotonic(s, informative)["brier_skill"] > 0.05
    assert cv_isotonic(s, noise)["brier_skill"] < 0.03
