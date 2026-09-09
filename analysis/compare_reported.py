"""Compare our results against the paper's reported numbers. Phase 25.

    python -m analysis.compare_reported
    python -m analysis.compare_reported --metric auc --tolerance 2.0

Reads  research/reported_results.csv   (the paper's numbers -- REFERENCE ONLY)
       results/aggregated/results.csv  (ours)
Writes results/tables/comparison_vs_reported.{md,csv}

Two rules this tool exists to protect:

1. **The paper's numbers are targets, never expected outputs.** Nothing here
   writes a reported value into a results file, and no result is ever adjusted
   toward one.
2. **Only `flag_bundled` rows are compared.** The paper's Table 4 was produced by
   the FLAG authors' own baseline rewrites, so comparing an `official` CARE-GNN
   against it would be a category error (decision D-002).

Statuses:
    MATCH          |delta| <= tolerance
    CLOSE          tolerance < |delta| <= 2x tolerance
    DEVIATION      |delta| > 2x tolerance
    NOT_REPRODUCED we ran it, but the run failed
    UNAVAILABLE    we have not run it
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTED = ROOT / "research" / "reported_results.csv"
OURS = ROOT / "results" / "aggregated" / "results.csv"

# The paper reports percentages; our results are fractions in [0, 1].
SCALE = 100.0

DEFAULT_TOLERANCE = 1.0
"""Percentage points. Chosen to be of the same order as the paper's own reported
standard deviations (Table 4 std ranges roughly 0.01-2.97), so "MATCH" means
"within the paper's own run-to-run noise" rather than an arbitrary bar."""


def load_reported(metric: str, variants: set[str]) -> dict:
    if not REPORTED.exists():
        raise SystemExit(
            f"{REPORTED} missing. Run:\n"
            f"  python -m scripts.analyze.build_reported_results"
        )
    out = {}
    with REPORTED.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["source_table"] != "Table 4":
                continue          # Table 3 is unreproducible, Table 5 is ablation
            if row["metric"] != metric or row["variant"] not in variants:
                continue
            out[(row["dataset"], row["model"], row["variant"])] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]) if row["std"] else 0.0,
                "n_runs": int(row["n_runs"]),
                "verification_status": row["verification_status"],
            }
    return out


def load_ours(metric: str) -> dict:
    if not OURS.exists():
        return {}
    column = f"test_{metric}"
    groups: dict[tuple, list[dict]] = {}
    with OURS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # Only the lineage the paper's numbers came from.
            if row.get("impl_source") != "flag_bundled":
                continue
            key = (row["dataset"], row["model"], row["variant"])
            groups.setdefault(key, []).append(row)

    out = {}
    for key, rows in groups.items():
        completed = [r for r in rows if r.get("status") == "completed"]
        if not completed:
            out[key] = {"status": "NOT_REPRODUCED", "n_runs": len(rows)}
            continue
        values = []
        for row in completed:
            try:
                values.append(float(row[column]) * SCALE)
            except (KeyError, ValueError, TypeError):
                pass
        if not values:
            out[key] = {"status": "NOT_REPRODUCED", "n_runs": len(completed)}
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        out[key] = {
            "mean": mean, "std": std, "n_runs": len(values), "status": "ok",
            "threshold_policy": completed[0].get("threshold_policy", ""),
        }
    return out


def classify(delta: float | None, tolerance: float, our_status: str) -> str:
    if our_status == "missing":
        return "UNAVAILABLE"
    if our_status == "NOT_REPRODUCED":
        return "NOT_REPRODUCED"
    magnitude = abs(delta)
    if magnitude <= tolerance:
        return "MATCH"
    if magnitude <= 2 * tolerance:
        return "CLOSE"
    return "DEVIATION"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="f1_macro",
                        choices=["f1_macro", "auc"])
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="percentage points for MATCH (default 1.0)")
    parser.add_argument("--variants", default="baseline,text,flag,flag_finetuned")
    parser.add_argument("--only-run", action="store_true",
                        help="hide combinations we have not attempted")
    args = parser.parse_args(argv)

    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    reported = load_reported(args.metric, variants)
    ours = load_ours(args.metric)

    rows = []
    for key in sorted(reported):
        dataset, model, variant = key
        ref = reported[key]
        mine = ours.get(key)

        if mine is None:
            rows.append({
                "dataset": dataset, "model": model, "variant": variant,
                "reported": f"{ref['mean']:.2f}±{ref['std']:.2f}",
                "ours": "-", "abs_diff": "", "rel_diff": "",
                "status": "UNAVAILABLE", "n_runs": 0,
            })
            continue
        if mine.get("status") == "NOT_REPRODUCED":
            rows.append({
                "dataset": dataset, "model": model, "variant": variant,
                "reported": f"{ref['mean']:.2f}±{ref['std']:.2f}",
                "ours": "failed", "abs_diff": "", "rel_diff": "",
                "status": "NOT_REPRODUCED", "n_runs": mine["n_runs"],
            })
            continue

        delta = mine["mean"] - ref["mean"]
        rel = delta / ref["mean"] * 100 if ref["mean"] else float("nan")
        rows.append({
            "dataset": dataset, "model": model, "variant": variant,
            "reported": f"{ref['mean']:.2f}±{ref['std']:.2f}",
            "ours": f"{mine['mean']:.2f}±{mine['std']:.2f}",
            "abs_diff": f"{delta:+.2f}", "rel_diff": f"{rel:+.1f}%",
            "status": classify(delta, args.tolerance, "ok"),
            "n_runs": mine["n_runs"],
        })

    if args.only_run:
        rows = [r for r in rows if r["status"] not in ("UNAVAILABLE",)]

    metric_label = {"f1_macro": "F1-macro", "auc": "AUC"}[args.metric]
    header = (
        f"# Reproduction vs reported — {metric_label}\n\n"
        f"Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.\n"
        f"Tolerance: MATCH within ±{args.tolerance:.1f} pp, "
        f"CLOSE within ±{2 * args.tolerance:.1f} pp.\n\n"
        f"Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 "
        f"was produced by the FLAG authors' own baseline rewrites, so an "
        f"`official` baseline is not comparable to it (decision D-002).\n\n"
        f"The reported column is a **reference target**. No result here has been "
        f"adjusted toward it, and reported values are never written into "
        f"`results/`.\n\n"
    )

    lines = [
        "| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['variant']} | "
            f"{row['reported']} | {row['ours']} | {row['abs_diff']} | "
            f"{row['rel_diff']} | {row['n_runs']} | {row['status']} |"
        )
    table = "\n".join(lines)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = "\n".join(
        f"- {status}: {count}"
        for status, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    caveats = (
        "\n\n## Why exact agreement is not achievable\n\n"
        "- The paper's 1:10 downsampling seed is unpublished, so our benchmark "
        "is a different draw of the same construction.\n"
        "- The paper states no train/val/test split; ours is 10/10/80 inherited "
        "from GraphAdapter/GLBench.\n"
        "- The paper states no decision-threshold policy. `argmax` reproduces "
        "the released code; `validation_swept` reproduces BWGNN's setup, which "
        "the paper says it follows. The two differ by several F1 points.\n"
        "- Semantic sampling is reimplemented from Eq. 3-4; no upstream source "
        "exists.\n"
        "- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/"
        "`--inits` here gives a std that is not comparable.\n"
    )

    out_md = ROOT / "results" / "tables" / f"comparison_vs_reported_{args.metric}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        header + table + "\n\n## Summary\n\n" + summary + caveats + "\n",
        encoding="utf-8",
    )

    out_csv = out_md.with_suffix(".csv")
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(header + table)
    print("\n## Summary\n")
    print(summary)
    print(f"\nwrote {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
