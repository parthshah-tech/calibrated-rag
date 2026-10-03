"""Does the confidence signal predict retrieval success?

    uv run python -m eval.validate_signal --results benchmarks/results/beir.json

Uses hybrid-mode records. Thresholds are fit on a random half (dev) and per-bucket success
rates are reported on the other half (held out). AUROC and calibration use all records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibration import auroc_ci, bucket_table, cv_isotonic, fit_thresholds


def validate_group(scores, labels, seed: int = 0) -> dict:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    perm = np.random.default_rng(seed).permutation(len(s))
    dev, test = perm[: len(s) // 2], perm[len(s) // 2 :]
    thresholds = fit_thresholds(s[dev])
    table = bucket_table(s[test], y[test], thresholds)
    rates = [r["rate"] for r in table if r["n"] > 0]
    auc, lo, hi = auroc_ci(s, y)
    return {
        "n": len(s),
        "base_success_rate": float(y.mean()),
        "auroc": {"value": auc, "lo": lo, "hi": hi},
        "thresholds_fit_on_dev": thresholds,
        "heldout_buckets": table,
        "monotonic": all(a <= b for a, b in zip(rates, rates[1:], strict=False)),
        "calibration_cv": cv_isotonic(s, y),
    }


def format_group(name: str, g: dict) -> str:
    a = g["auroc"]
    lines = [
        f"\n== {name}  (n={g['n']}, base success {g['base_success_rate']:.2f}) ==",
        f"AUROC {a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]   (0.5 = no signal)",
        "thresholds (fit on dev): low < {:.3f} <= medium < {:.3f} <= high".format(
            *g["thresholds_fit_on_dev"]
        ),
        "held-out success rate by bucket:",
    ]
    for r in g["heldout_buckets"]:
        lines.append(
            f"  {r['bucket']:7} n={r['n']:4}  {r['rate']:.2f}  [{r['lo']:.2f}, {r['hi']:.2f}]"
        )
    c = g["calibration_cv"]
    lines.append(f"monotonic Low<=Medium<=High: {g['monotonic']}")
    lines.append(f"isotonic (5-fold): ECE {c['ece']:.3f}, Brier skill {c['brier_skill']:+.3f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = json.loads(Path(args.results).read_text())
    rows = [r for r in data["records"] if r["mode"] == "hybrid" and r.get("conf_score") is not None]
    rows = [r for r in rows if r.get("success") is not None]
    groups = {"ALL": rows}
    for r in rows:
        groups.setdefault(r["dataset"], []).append(r)
    report = {}
    for name, g in groups.items():
        if len(g) < 20:
            print(f"\n== {name}: only {len(g)} records, too few to validate ==")
            continue
        report[name] = validate_group([r["conf_score"] for r in g], [r["success"] for r in g])
        print(format_group(name, report[name]))
    out = Path(args.out or Path(args.results).with_name("signal_validation.json"))
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
