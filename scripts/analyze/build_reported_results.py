"""Generate research/reported_results.csv -- the paper's numbers as REFERENCE TARGETS.

These are the values printed in the FLAG paper. They are NOT expected outputs and
must never be written into any results file produced by an actual run. The
comparison tool (analysis/compare_reported) reads this CSV on one side and our
own results/aggregated/results.csv on the other.

Source
------
FLAG: Fraud Detection with LLM-enhanced Graph Neural Network
Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, Chuan Shi
KDD '25, August 3-7 2025, Toronto, ON, Canada. pp. 5150-5160.
DOI: 10.1145/3711896.3737220
Open-access PDF: https://dl.acm.org/doi/pdf/10.1145/3711896.3737220
Author mirror:   http://www.shichuan.org/doc/200.pdf

Tables transcribed:
  Table 3 -- industrial dataset (Huabei / Alipay), single run, no std
  Table 4 -- main results on Reddit and Instagram, mean+-std over 25 runs
  Table 5 -- ablation (SS / LLM / SG), mean+-std

verification_status column:
  TRANSCRIBED  -- read out of the published PDF; not yet re-checked by a second pass
Every row starts as TRANSCRIBED. Promote a row only after a human has re-read it
against the PDF. Do not promote rows automatically.

Usage: python -m scripts.analyze.build_reported_results
"""
from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "reported_results.csv"

PAPER = "FLAG (KDD 2025)"
DOI = "10.1145/3711896.3737220"

# ---------------------------------------------------------------------------
# Table 4 -- main results. Values are percentages.
# Layout per model: variant -> (reddit_f1, reddit_auc, instagram_f1, instagram_auc)
# each as (mean, std).
# ---------------------------------------------------------------------------
TABLE4 = {
    "gcn": {
        "baseline":      ((45.46, 0.01), (50.32, 0.26), (47.88, 0.96), (52.61, 1.80)),
        "text":          ((45.84, 0.36), (57.82, 1.94), (47.29, 0.01), (55.74, 0.73)),
        "flag":          ((48.19, 1.02), (60.18, 0.79), (48.05, 0.69), (56.31, 0.83)),
        "flag_finetuned":((48.72, 1.59), (60.88, 0.68), (49.79, 1.08), (55.45, 1.21)),
    },
    "gat": {
        "baseline":      ((46.66, 0.96), (52.66, 2.25), (49.21, 1.38), (51.53, 1.12)),
        "text":          ((48.26, 2.23), (59.32, 0.29), (48.31, 0.98), (54.69, 0.76)),
        "flag":          ((49.70, 1.76), (60.61, 1.20), (49.13, 1.23), (54.97, 0.75)),
        "flag_finetuned":((50.20, 2.33), (60.57, 1.03), (50.65, 1.61), (55.98, 1.67)),
    },
    "geniepath": {
        "baseline":      ((45.46, 0.01), (52.18, 1.48), (47.31, 0.05), (51.22, 3.16)),
        "text":          ((46.84, 1.89), (56.91, 1.85), (47.29, 0.01), (52.45, 2.31)),
        "flag":          ((48.30, 2.24), (59.43, 0.55), (48.41, 1.51), (55.59, 0.85)),
        "flag_finetuned":((48.69, 2.97), (59.74, 1.68), (48.28, 1.25), (56.24, 2.10)),
    },
    "care_gnn": {
        "baseline":      ((45.46, 0.01), (51.35, 1.62), (47.29, 0.01), (52.07, 1.95)),
        "text":          ((47.66, 1.59), (56.72, 1.38), (48.05, 1.07), (54.92, 0.38)),
        "flag":          ((50.78, 0.95), (58.43, 0.65), (49.64, 1.74), (55.79, 0.58)),
        "flag_finetuned":((51.95, 2.21), (58.74, 1.40), (50.24, 0.46), (56.40, 1.29)),
    },
    "bwgnn": {
        "baseline":      ((45.47, 0.01), (53.82, 2.49), (47.28, 0.01), (51.52, 2.43)),
        "text":          ((48.76, 1.54), (57.56, 1.53), (47.61, 0.72), (54.10, 0.71)),
        "flag":          ((50.93, 2.16), (58.89, 2.50), (48.55, 0.32), (56.33, 0.72)),
        "flag_finetuned":((51.91, 2.38), (59.20, 1.09), (49.35, 1.45), (57.19, 0.28)),
    },
    "dga_gnn": {
        "baseline":      ((45.46, 0.01), (50.10, 0.53), (47.29, 0.01), (50.86, 0.48)),
        "text":          ((45.49, 0.11), (59.59, 1.60), (47.29, 0.01), (56.28, 1.07)),
        "flag":          ((48.77, 2.18), (61.05, 0.71), (48.53, 0.42), (56.73, 0.66)),
        "flag_finetuned":((49.50, 0.83), (61.61, 0.78), (48.95, 0.68), (57.20, 1.03)),
    },
    "pmp": {
        "baseline":      ((46.31, 1.04), (50.16, 0.12), (47.50, 1.56), (50.63, 0.52)),
        "text":          ((47.21, 1.18), (59.79, 0.43), (48.29, 0.01), (56.05, 1.27)),
        "flag":          ((48.91, 1.30), (61.32, 0.66), (48.48, 0.82), (57.10, 0.62)),
        "flag_finetuned":((50.14, 1.48), (61.80, 0.99), (49.76, 1.64), (57.67, 1.09)),
    },
}

# ---------------------------------------------------------------------------
# Table 3 -- industrial dataset (Huabei). NOT REPRODUCIBLE: proprietary Alipay
# data, 13M nodes / 120M edges, single run. Recorded for completeness only.
# model -> (ks, f1_macro, auc)
# ---------------------------------------------------------------------------
TABLE3 = {
    "gcn":       (76.42, 22.76, 94.80),
    "gat":       (76.59, 22.89, 94.87),
    "geniepath": (76.91, 23.28, 94.97),
    "care_gnn":  (76.92, 23.28, 94.98),
    "bwgnn":     (76.85, 23.12, 94.93),
    "dga_gnn":   (76.84, 23.15, 94.92),
    "pmp":       (76.96, 23.32, 95.02),
    "flag":      (77.78, 24.48, 95.41),
    "flag_star": (77.86, 24.72, 95.49),
}

# ---------------------------------------------------------------------------
# Table 5 -- ablation. Component flags are (SS, LLM, SG).
# NOTE: only 5 backbones are ablated; DGA-GNN and PMP are absent from Table 5.
# The full SS+LLM+SG row is not repeated in Table 5 -- it is the '+FLAG'
# (zero-shot) row of Table 4.
# ---------------------------------------------------------------------------
TABLE5 = {
    "gcn": [
        ((0, 0, 1), (47.92, 2.77), (59.60, 1.56), (47.84, 0.79), (55.52, 0.46)),
        ((1, 0, 1), (48.11, 0.66), (60.07, 0.64), (47.97, 0.53), (55.85, 0.38)),
        ((1, 1, 0), (48.19, 0.51), (59.81, 2.11), (47.99, 0.31), (56.03, 0.65)),
    ],
    "gat": [
        ((0, 0, 1), (49.03, 2.69), (59.96, 0.41), (48.76, 1.83), (54.20, 1.64)),
        ((1, 0, 1), (49.16, 0.63), (59.80, 1.64), (48.79, 0.98), (54.79, 0.25)),
        ((1, 1, 0), (49.73, 1.10), (60.27, 1.02), (49.28, 2.02), (54.27, 1.02)),
    ],
    "geniepath": [
        ((0, 0, 1), (47.47, 2.14), (58.35, 0.26), (47.34, 0.79), (54.38, 0.61)),
        ((1, 0, 1), (48.38, 2.52), (58.31, 0.36), (47.54, 0.48), (55.24, 1.94)),
        ((1, 1, 0), (48.13, 1.74), (59.24, 2.99), (47.88, 0.96), (55.61, 1.80)),
    ],
    "care_gnn": [
        ((0, 0, 1), (50.19, 1.10), (57.39, 0.64), (48.47, 1.04), (55.45, 0.64)),
        ((1, 0, 1), (50.48, 2.29), (57.94, 0.60), (48.70, 0.42), (55.47, 0.65)),
        ((1, 1, 0), (50.50, 1.30), (57.90, 0.54), (48.50, 1.48), (55.83, 0.48)),
    ],
    "bwgnn": [
        ((0, 0, 1), (49.23, 2.05), (58.46, 2.36), (48.28, 0.24), (55.54, 0.25)),
        ((1, 0, 1), (50.19, 2.00), (58.76, 1.82), (47.64, 0.79), (56.86, 0.41)),
        ((1, 1, 0), (50.19, 2.89), (58.63, 1.76), (48.29, 0.81), (56.27, 2.15)),
    ],
}

FIELDS = [
    "source_table", "dataset", "model", "variant", "ablation_ss", "ablation_llm",
    "ablation_sg", "metric", "mean", "std", "unit", "n_runs", "paper", "doi",
    "verification_status", "notes",
]


def rows():
    # ---- Table 4 ----
    for model, variants in TABLE4.items():
        for variant, (r_f1, r_auc, i_f1, i_auc) in variants.items():
            for dataset, (f1, auc) in (
                ("reddit", (r_f1, r_auc)),
                ("instagram", (i_f1, i_auc)),
            ):
                for metric, (mean, std) in (("f1_macro", f1), ("auc", auc)):
                    yield {
                        "source_table": "Table 4",
                        "dataset": dataset,
                        "model": model,
                        "variant": variant,
                        "ablation_ss": "", "ablation_llm": "", "ablation_sg": "",
                        "metric": metric,
                        "mean": f"{mean:.2f}", "std": f"{std:.2f}",
                        "unit": "percent", "n_runs": 25,
                        "paper": PAPER, "doi": DOI,
                        "verification_status": "TRANSCRIBED",
                        "notes": "",
                    }

    # ---- Table 3 ----
    for model, (ks, f1, auc) in TABLE3.items():
        for metric, value in (("ks", ks), ("f1_macro", f1), ("auc", auc)):
            note = (
                "proprietary Alipay dataset (13M nodes / 120M edges); "
                "NOT REPRODUCIBLE outside Alipay; single run, no std. "
                "Baselines carry a dagger = trained on raw-text embeddings. "
                "FLAG backbone fixed to GeniePath."
            )
            yield {
                "source_table": "Table 3",
                "dataset": "huabei_industrial",
                "model": model,
                "variant": (
                    "flag" if model == "flag"
                    else "flag_finetuned" if model == "flag_star"
                    else "text"
                ),
                "ablation_ss": "", "ablation_llm": "", "ablation_sg": "",
                "metric": metric,
                "mean": f"{value:.2f}", "std": "",
                "unit": "percent", "n_runs": 1,
                "paper": PAPER, "doi": DOI,
                "verification_status": "TRANSCRIBED",
                "notes": note,
            }

    # ---- Table 5 ----
    for model, entries in TABLE5.items():
        for (ss, llm, sg), r_f1, r_auc, i_f1, i_auc in entries:
            for dataset, (f1, auc) in (
                ("reddit", (r_f1, r_auc)),
                ("instagram", (i_f1, i_auc)),
            ):
                for metric, (mean, std) in (("f1_macro", f1), ("auc", auc)):
                    yield {
                        "source_table": "Table 5",
                        "dataset": dataset,
                        "model": model,
                        "variant": "ablation",
                        "ablation_ss": ss,
                        "ablation_llm": llm,
                        "ablation_sg": sg,
                        "metric": metric,
                        "mean": f"{mean:.2f}", "std": f"{std:.2f}",
                        "unit": "percent", "n_runs": 25,
                        "paper": PAPER, "doi": DOI,
                        "verification_status": "TRANSCRIBED",
                        "notes": (
                            "ablation LLM is NOT fine-tuned; non-LLM rows use raw "
                            "text encoding as node features. The full SS+LLM+SG row "
                            "is Table 4's '+FLAG'. DGA-GNN and PMP are absent from "
                            "Table 5 in the paper."
                        ),
                    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data)

    by_table = {}
    for row in data:
        by_table[row["source_table"]] = by_table.get(row["source_table"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}")
    for table, count in sorted(by_table.items()):
        print(f"  {table}: {count} rows")
    print(f"  total: {len(data)} rows")
    print("\nAll rows are verification_status=TRANSCRIBED.")
    print("Spot-check against the PDF before using as a reproduction verdict.")


if __name__ == "__main__":
    main()
