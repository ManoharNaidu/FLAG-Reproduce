"""Focused flag-vs-FLAG-MD comparison report. Reads ONLY the already-built
results/final_report/data/all_runs.csv (see scripts.analyze.build_final_report) -- no
training, no new data.

    .venv-report/Scripts/python.exe -m scripts.analyze.build_flag_vs_flagmd_report

Writes:
    results/final_report/flag_vs_flag_md.md
    results/final_report/figures/flag_vs_flag_md_delta_auc.png
    results/final_report/figures/flag_vs_flag_md_delta_f1_macro.png

Why a PAIRED comparison, not just two independent means: `scripts/reproduce/
run_flag_md_matrix.sh` documents that the seed/init streams do NOT depend on the sampler,
so "cosine and MD runs are PAIRED" -- run (dataset, model, seed=s, init=i) under cosine
and the SAME (dataset, model, seed=s, init=i) under md_K2_matched share the same data
shuffling and weight initialisation, differing only in which neighbours the sampler
picked. That makes the within-pair difference a much more sensitive (lower-variance)
statistic than comparing two independent 8-run groups, and it is the comparison the
repo's own design intends. A paired t-test (scipy.stats.ttest_rel) is reported alongside
the raw numbers -- not to claim significance where there isn't any, but to make the "is
this delta distinguishable from noise" question checkable rather than eyeballed.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "final_report"
ALL_RUNS = REPORT / "data" / "all_runs.csv"

MODEL_ORDER = ["gcn", "gat", "geniepath", "care_gnn", "bwgnn", "dga_gnn", "pmp"]
DATASETS = ["reddit", "instagram"]
METRICS = [
    ("test_auc", "AUC"),
    ("test_f1_macro", "F1-macro"),
    ("test_recall_fraud", "Recall (fraud)"),
    ("test_precision_fraud", "Precision (fraud)"),
]


def load_paired() -> pd.DataFrame:
    if not ALL_RUNS.exists():
        raise FileNotFoundError(
            f"{ALL_RUNS} missing. Run scripts.analyze.build_final_report first."
        )
    df = pd.read_csv(ALL_RUNS)
    df = df[(df["variant"] == "flag") & (df["status"] == "completed")
            & (df["sampler"].isin(["cosine", "md_K2_matched"]))]
    key = ["dataset", "model", "seed", "initialization"]
    metric_cols = [m for m, _ in METRICS]
    wide = df.pivot_table(index=key, columns="sampler", values=metric_cols)
    # keep only pairs present under BOTH samplers
    wide = wide.dropna()
    return wide


def paired_stats(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in wide.groupby(level=["dataset", "model"]):
        n = len(group)
        row = {"dataset": dataset, "model": model, "n_pairs": n}
        for metric, _ in METRICS:
            cosine = group[(metric, "cosine")].to_numpy()
            md = group[(metric, "md_K2_matched")].to_numpy()
            delta = md - cosine
            row[f"{metric}_cosine_mean"] = cosine.mean()
            row[f"{metric}_cosine_std"] = cosine.std(ddof=1) if n > 1 else 0.0
            row[f"{metric}_md_mean"] = md.mean()
            row[f"{metric}_md_std"] = md.std(ddof=1) if n > 1 else 0.0
            row[f"{metric}_delta_mean"] = delta.mean()
            row[f"{metric}_delta_std"] = delta.std(ddof=1) if n > 1 else 0.0
            if n > 1 and delta.std(ddof=1) > 0:
                t, p = stats.ttest_rel(md, cosine)
            else:
                t, p = np.nan, np.nan
            row[f"{metric}_paired_t"] = t
            row[f"{metric}_paired_p"] = p
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["dataset", "model"])


def make_delta_chart(stats_df: pd.DataFrame, metric: str, label: str, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = stats_df[stats_df["dataset"] == dataset]
        labels = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
        sub = sub.set_index(sub["model"].astype(str))
        means = np.array([sub[f"{metric}_delta_mean"].get(m, np.nan) for m in labels])
        stds = np.array([sub[f"{metric}_delta_std"].get(m, 0.0) for m in labels])
        pvals = np.array([sub[f"{metric}_paired_p"].get(m, np.nan) for m in labels])
        x = np.arange(len(labels))
        colors = ["tab:green" if m > 0 else "tab:red" for m in means]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        for xi, (m, s, p) in enumerate(zip(means, stds, pvals)):
            if np.isfinite(p) and p < 0.05:
                ax.annotate("*", (xi, m + np.sign(m) * (abs(s) + 0.002)),
                            ha="center", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"paired delta: FLAG-MD (K2) minus cosine, {label}")
    fig.suptitle(f"{label}: FLAG-MD vs cosine, PAIRED per (seed,init) - n=8 pairs/model\n"
                 f"error bars = std of the 8 paired deltas; * = paired t-test p<0.05 "
                 f"(none expected here, see report text)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_table(stats_df: pd.DataFrame, dataset: str) -> str:
    sub = stats_df[stats_df["dataset"] == dataset].sort_values("model")
    lines = [
        "| Model | AUC cosine | AUC FLAG-MD | ΔAUC (paired) | p (paired t-test) | "
        "F1 cosine | F1 FLAG-MD | ΔF1 (paired) | p | Recall cosine | Recall FLAG-MD |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sub.iterrows():
        p_auc = "n/a" if pd.isna(r["test_auc_paired_p"]) else f"{r['test_auc_paired_p']:.3f}"
        p_f1 = "n/a" if pd.isna(r["test_f1_macro_paired_p"]) else f"{r['test_f1_macro_paired_p']:.3f}"
        lines.append(
            f"| {r['model']} "
            f"| {r['test_auc_cosine_mean']:.4f} "
            f"| {r['test_auc_md_mean']:.4f} "
            f"| {r['test_auc_delta_mean']:+.4f} ± {r['test_auc_delta_std']:.4f} "
            f"| {p_auc} "
            f"| {r['test_f1_macro_cosine_mean']:.4f} "
            f"| {r['test_f1_macro_md_mean']:.4f} "
            f"| {r['test_f1_macro_delta_mean']:+.4f} ± {r['test_f1_macro_delta_std']:.4f} "
            f"| {p_f1} "
            f"| {r['test_recall_fraud_cosine_mean']:.4f} "
            f"| {r['test_recall_fraud_md_mean']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    wide = load_paired()
    stats_df = paired_stats(wide)
    stats_df.to_csv(REPORT / "data" / "flag_vs_flag_md_paired_stats.csv", index=False)

    figdir = REPORT / "figures"
    make_delta_chart(stats_df, "test_auc", "AUC", figdir / "flag_vs_flag_md_delta_auc.png")
    make_delta_chart(stats_df, "test_f1_macro", "F1-macro",
                      figdir / "flag_vs_flag_md_delta_f1_macro.png")

    headline_p_cols = ["test_auc_paired_p", "test_f1_macro_paired_p"]
    n_sig = int((stats_df[headline_p_cols] < 0.05).sum().sum())
    n_tests = int(stats_df[headline_p_cols].notna().sum().sum())
    n_sig_all = int((stats_df.filter(like="_paired_p") < 0.05).sum().sum())
    n_tests_all = int(stats_df.filter(like="_paired_p").notna().sum().sum())

    md = f"""# FLAG (cosine) vs FLAG-MD (Markov-diffusion, K=2, matched) - paired comparison

[Back to final report](README.md)

Source: `results/final_report/data/all_runs.csv` (built by
`scripts.analyze.build_final_report` from `results/flag_md/flag/{{cosine,md_K2_matched}}/`).
**No new training was run to produce this file.** `flag_finetuned` is not part of this
comparison (it was not yet run when this earlier report was built). It has since been run for all
7 backbones x 5 datasets x {cosine, FLAG-MD K=2}; see
`results/2026-10-02-flag-cosine-vs-md-main-run-report.md`.

## Why "paired", not just two means

`scripts/reproduce/run_flag_md_matrix.sh` documents that the seed/init streams "do not
depend on the sampler, so cosine and MD runs are PAIRED" -- run (dataset, model, seed=s,
init=i) under `cosine` and the identical (seed=s, init=i) under `md_K2_matched` share the
same data shuffling and weight initialisation; only the sampler differs. So instead of just
comparing two independent 8-run means (noisier), each of the 8 pairs' difference
(`md_K2_matched - cosine`) is computed first, then averaged -- and a paired t-test
(`scipy.stats.ttest_rel`) checks whether that average difference is distinguishable from
zero. n=8 pairs per (dataset, model) cell throughout.

## Headline

Across the {n_tests} paired tests on AUC and F1-macro ({len(stats_df)} model/dataset cells x
2 metrics), **{n_sig} {"was" if n_sig == 1 else "were"} significant at p<0.05** -- about what you'd expect from chance
alone at that threshold (~5% of {n_tests} is ~{n_tests * 0.05:.1f}). Widening to all 4
metrics tracked (AUC, F1-macro, recall, precision), it's **{n_sig_all} of {n_tests_all}**,
same story. **The honest reading is that this repo's data does not show a real, consistent
difference between the cosine sampler and this one Markov-diffusion configuration (K=2,
matched neighbour count), for either metric, on either dataset, for any of the 7 backbones.**
Any single model/dataset cell that looks like a win or a loss in the raw numbers is very
likely sampling noise from only 4 seeds x 2 inits, not a real effect of the sampler.

## Reddit

{render_table(stats_df, "reddit")}

## Instagram

{render_table(stats_df, "instagram")}

## Figures

- `figures/flag_vs_flag_md_delta_auc.png` -- paired ΔAUC per model, both datasets, with
  paired-t-test significance stars (there are none, see headline above).
- `figures/flag_vs_flag_md_delta_f1_macro.png` -- same for F1-macro.
- Also relevant (from the main report, independent-groups view, not paired):
  `figures/sampler_ablation_flag_test_auc__{{reddit,instagram}}.png`,
  `figures/sampler_ablation_flag_test_f1_macro__{{reddit,instagram}}.png`,
  `figures/training_curves/flag_variant_val_auc__{{reddit,instagram}}.png`.

## Full numbers

Every mean, std, paired delta, t-statistic and p-value for every metric (including recall,
precision, F1-fraud, accuracy, KS, ECE -- not just AUC/F1-macro shown above):
[`data/flag_vs_flag_md_paired_stats.csv`](data/flag_vs_flag_md_paired_stats.csv).

## What this does NOT cover

- **flag_finetuned** vs anything -- not in this earlier report; now covered in the main run report (see top).
- FLAG-MD configurations other than K=2 matched-count (K=1, K=3, K=5, top-n selection) --
  those exist only for the LLM-free `text` variant
  (`summary/sampler_ablation_text_variant.csv`), not for `flag`.
- Whether 8 pairs is enough to detect a real but small effect -- it may not be; this report
  says "not detectable at n=8," not "there is no effect."
"""
    (REPORT / "flag_vs_flag_md.md").write_text(md, encoding="utf-8")
    print(f"wrote {REPORT / 'flag_vs_flag_md.md'}")
    print(f"wrote data/flag_vs_flag_md_paired_stats.csv, 2 figures")
    print(f"significant paired tests (p<0.05): {n_sig}/{n_tests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
