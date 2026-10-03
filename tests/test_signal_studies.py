import json

import numpy as np
import pytest

from eval.calibration import auroc_diff_ci, transfer_calibration
from eval.signal_studies import load_rows, metric_comparison, run, transfer


def rows(dataset, n, seed, shift=0.0, noisy=("overlap", "tau")):
    rng = np.random.default_rng(seed)
    base = np.clip(rng.uniform(0, 1, n) - shift, 0, 1)
    success = rng.uniform(0, 1, n) < np.clip(base + shift, 0, 1)  # success tracks the true signal
    out = []
    for i in range(n):
        rec = {"dataset": dataset, "mode": "hybrid", "qid": f"q{i}", "success": bool(success[i])}
        for m in ("overlap", "rbo", "tau"):
            rec[f"conf_{m}"] = float(rng.uniform(0, 1) if m in noisy else base[i])
        out.append(rec)
    return out


def test_informative_metric_beats_noisy_ones_with_significance():
    c = metric_comparison(rows("a", 600, 0))
    assert c["metrics"]["rbo"]["lo"] > 0.6
    assert c["metrics"]["overlap"]["auroc"] < 0.6
    assert c["vs_rbo"]["overlap"]["significant"] and c["vs_rbo"]["overlap"]["diff"] < 0


def test_auroc_diff_is_zero_for_identical_scores():
    s = np.random.default_rng(0).uniform(0, 1, 200)
    y = np.random.default_rng(1).uniform(0, 1, 200) < s
    d, lo, hi = auroc_diff_ci(s, s, y)
    assert d == 0 and lo == 0 and hi == 0


def test_transfer_exposes_distribution_shift():
    a, b = rows("a", 500, 1), rows("b", 500, 2, shift=0.35)
    t = transfer(a, b)
    # B's scores sit lower, so most B queries land in Low under A's thresholds
    assert t["B_occupancy"]["Low"] > 0.5
    assert sum(t["B_occupancy"].values()) == pytest.approx(1.0)
    assert set(t) >= {"A_to_B_isotonic", "within_B_isotonic_cv", "thresholds_from_A"}


def test_transfer_calibration_hurts_under_shift_but_not_without_it():
    rng = np.random.default_rng(0)
    s_a = rng.uniform(0, 1, 800)
    y_a = rng.uniform(0, 1, 800) < s_a
    s_b = rng.uniform(0, 1, 800)
    same = transfer_calibration(s_a, y_a, s_b, rng.uniform(0, 1, 800) < s_b)
    flipped = transfer_calibration(s_a, y_a, s_b, rng.uniform(0, 1, 800) < (1 - s_b))
    assert same["ece"] < 0.1 < flipped["ece"]
    assert same["brier_skill"] > 0 > flipped["brier_skill"]


def test_run_covers_all_dataset_pairs_and_load_rows_validates(tmp_path):
    allrows = rows("a", 200, 1) + rows("b", 200, 2)
    report = run(allrows)
    assert set(report["transfer"]) == {"a -> b", "b -> a"}
    assert set(report["metric_comparison"]) == {"ALL", "a", "b"}
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"records": allrows}))
    assert len(load_rows(good)) == 400
    bad = tmp_path / "bad.json"
    stripped = [{k: v for k, v in r.items() if k != "conf_tau"} for r in allrows]
    bad.write_text(json.dumps({"records": stripped}))
    with pytest.raises(SystemExit):
        load_rows(bad)
