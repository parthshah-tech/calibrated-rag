"""Two questions about the confidence signal, answered on saved benchmark results.

1. Which agreement metric is best? AUROC for overlap@k, RBO and Kendall's tau (on shared
   items), with paired bootstrap differences against RBO, our default.
2. Do thresholds and calibration transfer across datasets? Fit on dataset A, apply to B.

    uv run python -m eval.signal_studies --results benchmarks/results/beir.json

Needs per-metric scores in the results file (re-run benchmarks.beir_eval if they are missing)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from eval.calibration import (
    auroc_ci,
    auroc_diff_ci,
    bucket_table,
    cv_isotonic,
    fit_thresholds,
    transfer_calibration,
)

METRICS = ("overlap", "rbo", "tau")


def load_rows(path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["records"] if r["mode"] == "hybrid" and r.get("success") is not None]
    if not rows or any(r.get(f"conf_{m}") is None for r in rows for m in METRICS):
        raise SystemExit(
            "results lack per-metric confidence scores: re-run `python -m benchmarks.beir_eval`"
        )
    return rows


def _arr(rows, metric):
    return np.array([r[f"conf_{metric}"] for r in rows], dtype=float)


def metric_comparison(rows) -> dict:
    y = np.array([r["success"] for r in rows], dtype=bool)
    out = {"n": len(rows), "metrics": {}, "vs_rbo": {}}
    for m in METRICS:
        auc, lo, hi = auroc_ci(_arr(rows, m), y)
        out["metrics"][m] = {"auroc": auc, "lo": lo, "hi": hi}
    for m in METRICS:
        if m != "rbo":
            d, lo, hi = auroc_diff_ci(_arr(rows, m), _arr(rows, "rbo"), y)
            out["vs_rbo"][m] = {"diff": d, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}
    return out


def transfer(rows_a, rows_b, metric: str = "rbo") -> dict:
    """Thresholds (tertiles of A) and an isotonic calibrator fit on A, evaluated on B."""
    sa, sb = _arr(rows_a, metric), _arr(rows_b, metric)
    ya = np.array([r["success"] for r in rows_a], dtype=bool)
    yb = np.array([r["success"] for r in rows_b], dtype=bool)
    th = fit_thresholds(sa)
    table = bucket_table(sb, yb, th)
    n = len(sb)
    return {
        "thresholds_from_A": th,
        "B_buckets": table,
        "B_occupancy": {r["bucket"]: r["n"] / n for r in table},
        "A_to_B_isotonic": transfer_calibration(sa, ya, sb, yb),
        "within_B_isotonic_cv": cv_isotonic(sb, yb),
    }


def run(rows) -> dict:
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    report = {"metric_comparison": {"ALL": metric_comparison(rows)}, "transfer": {}}
    for name, g in by_ds.items():
        report["metric_comparison"][name] = metric_comparison(g)
    for a, b in itertools.permutations(by_ds, 2):
        report["transfer"][f"{a} -> {b}"] = transfer(by_ds[a], by_ds[b])
    return report


def format_report(report: dict) -> str:
    lines = []
    for name, c in report["metric_comparison"].items():
        lines.append(f"\n== metric comparison: {name} (n={c['n']}) ==")
        for m, v in c["metrics"].items():
            lines.append(f"  {m:8} AUROC {v['auroc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        for m, v in c["vs_rbo"].items():
            flag = "significant" if v["significant"] else "not significant"
            lines.append(
                f"  {m} minus rbo: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] ({flag})"
            )
    for name, t in report["transfer"].items():
        occ = ", ".join(f"{k} {v:.0%}" for k, v in t["B_occupancy"].items())
        lo, hi = t["thresholds_from_A"]
        a, w = t["A_to_B_isotonic"], t["within_B_isotonic_cv"]
        lines.append(f"\n== transfer {name} ==")
        lines.append(f"  thresholds from A: {lo:.3f} / {hi:.3f}")
        lines.append(f"  share of B queries per bucket under A's thresholds: {occ}")
        for r in t["B_buckets"]:
            lines.append(f"    {r['bucket']:7} n={r['n']:4} success {r['rate']:.2f}")
        lines.append(
            f"  calibration on B, fit on A: ECE {a['ece']:.3f}, Brier skill "
            f"{a['brier_skill']:+.3f} vs A's base rate, "
            f"{a['brier_skill_vs_test_base']:+.3f} vs B's own base rate"
        )
        lines.append(
            f"  calibration on B, fit in B: ECE {w['ece']:.3f}, skill {w['brier_skill']:+.3f}"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = run(load_rows(args.results))
    print(format_report(report))
    out = Path(args.out or Path(args.results).with_name("signal_studies.json"))
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
