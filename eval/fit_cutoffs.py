"""Suggest badge cutoffs from the results of the custom-corpus harness.

    uv run python -m eval.eval_harness --corpus ./eval/corpus --qa ./eval/sample_qa.json
    uv run python -m eval.fit_cutoffs --results eval_results.json

Uses answerable questions in hybrid mode and splits the confidence scores into thirds. It
reports success per bucket on the same questions, so treat the numbers as a sanity check, not
as held-out evidence, and use as many questions as you can."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.calibration import bucket_table, fit_thresholds


def suggest(records: list[dict], strategy: str | None = None) -> dict:
    rows = [
        r
        for r in records
        if r["mode"] == "hybrid"
        and r.get("success") is not None
        and r.get("conf_score") is not None
        and (strategy is None or r.get("strategy") == strategy)
    ]
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} usable questions; write more before fitting cutoffs")
    scores = [r["conf_score"] for r in rows]
    low, high = fit_thresholds(scores)
    return {
        "n": len(rows),
        "low": low,
        "high": high,
        "buckets": bucket_table(scores, [r["success"] for r in rows], (low, high)),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results.json")
    ap.add_argument("--strategy", default="recursive")
    args = ap.parse_args(argv)
    out = suggest(json.loads(Path(args.results).read_text())["records"], args.strategy)
    print(f"questions used: {out['n']}")
    for b in out["buckets"]:
        print(f"  {b['bucket']:7} n={b['n']:3}  retrieval success {b['rate']:.2f}")
    print("\nAdd to .env:")
    print(f"CONFIDENCE_LOW={out['low']:.3f}\nCONFIDENCE_HIGH={out['high']:.3f}")


if __name__ == "__main__":
    main()
