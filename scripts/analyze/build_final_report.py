"""Build the final results report from EXISTING result JSONs. No training is run here.

    .venv-report/Scripts/python.exe -m scripts.analyze.build_final_report

Sources (nothing else is read):
    results/raw/*.json                         baseline, text -- 1 run (seed0,init0) each,
                                                7 models x 2 datasets, sampler = "none"
    results/flag_md/text/<sampler>/*.json      text variant, 6 samplers, 4 seeds x 2 inits
    results/flag_md/flag/<sampler>/*.json      flag variant, 2 samplers, 4 seeds x 2 inits
    results/flag_md/gpu_profile/**             EXCLUDED -- GPU timing profile runs, not part
                                                of the results grid (2 files/sampler)

Not included, and not fabricated:
    flag_finetuned  -- no run of this variant exists anywhere in this repository. It needs a
                       GPU-generated "residual" LLM-text cache that is not present on this
                       machine (cache/llm/ is empty). See report README "Known gaps".
    DGP             -- not an implemented method in this codebase (only cited as the idea
                       source for FLAG-MD's diffusion operator). Skipped per instruction.
    ROC curves      -- no stored artifact (result JSON or checkpoint) contains raw per-example
                       prediction scores/labels, so a TPR/FPR sweep cannot be reconstructed
                       without re-running inference. Only the scalar AUC (already computed by
                       sklearn.roc_auc_score inside the original run) is available, plus the
                       per-epoch train-loss / val-AUC / val-F1 history that WAS stored.

Writes everything under results/final_report/ (see that folder's README.md for the layout).
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "final_report"
RAW_DIR = ROOT / "results" / "raw"
FLAGMD_DIR = ROOT / "results" / "flag_md"

SCALAR_FIELDS = [
    "experiment_id", "timestamp", "git_commit", "dataset", "model", "variant",
    "impl_source", "model_fidelity", "fidelity_class", "seed", "initialization",
    "device", "framework", "python_version", "platform",
    "dataset_version", "llm_model", "llm_finetuned",
    "best_epoch", "epochs_run", "stopped_early",
    "threshold", "threshold_policy",
    "validation_auc", "validation_f1_macro",
    "test_auc", "test_f1_macro", "test_precision_fraud", "test_recall_fraud",
    "test_f1_fraud", "test_accuracy", "test_ks", "test_ece",
    "training_time", "inference_time",
    "status", "error_message",
]

MODEL_ORDER = ["gcn", "gat", "geniepath", "care_gnn", "bwgnn", "dga_gnn", "pmp"]
DATASETS = ["reddit", "instagram"]
TEXT_SAMPLERS = ["cosine", "md_K1_matched", "md_K2_matched", "md_K2_top_n",
                  "md_K3_matched", "md_K5_matched"]
FLAG_SAMPLERS = ["cosine", "md_K2_matched"]


def load_one(path: pathlib.Path, sampler: str, source: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    row = {k: d.get(k) for k in SCALAR_FIELDS}
    sc = d.get("sampling_config", {}) or {}
    row["sampler"] = sampler
    row["sampling_strategy"] = sc.get("strategy")
    row["sampling_cache_key"] = sc.get("cache_key")
    row["diffusion_steps"] = sc.get("diffusion_steps")
    row["md_selection"] = sc.get("md_selection")
    row["source"] = source
    row["source_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
    row["epoch_history"] = (d.get("extra") or {}).get("epoch_history") or []
    row["feature_source"] = (d.get("extra") or {}).get("feature_source")
    # A model that predicts the majority class for every test example gives
    # precision_fraud = recall_fraud = 0.0 and an F1-macro that depends only on
    # the class ratio, not on the model (research/degenerate_baselines.md).
    # Flag it explicitly rather than presenting the number unqualified.
    row["is_degenerate_all_majority"] = (
        row.get("test_precision_fraud") == 0.0 and row.get("test_recall_fraud") == 0.0
    )
    return row


def collect_runs() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW_DIR.glob("*.json")):
        rows.append(load_one(p, sampler="none", source="results/raw"))
    for variant, samplers in (("text", TEXT_SAMPLERS), ("flag", FLAG_SAMPLERS)):
        for sampler in samplers:
            d = FLAGMD_DIR / variant / sampler
            for p in sorted(d.glob("*.json")):
                rows.append(load_one(p, sampler=sampler, source=f"results/flag_md/{variant}/{sampler}"))
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    # scripts.train.run has no resume logic: re-running --seeds N --inits M when a
    # (seed, init) cell already has a result adds a SECOND, near-identical file for that
    # cell (same seed -> same RNG state -> ~same outcome up to float nondeterminism), not
    # a new independent observation. Keep one per (dataset, model, variant, sampler, seed,
    # init) so duplicates cannot silently inflate a cell's run count or bias its mean.
    key = ["dataset", "model", "variant", "sampler", "seed", "initialization"]
    before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=key, keep="last")
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} duplicate (dataset,model,variant,sampler,seed,init) "
              f"file(s); kept the most recent timestamp for each")
    return df


def explode_epochs(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, r in df.iterrows():
        for e in r["epoch_history"]:
            recs.append({
                "dataset": r["dataset"], "model": r["model"], "variant": r["variant"],
                "sampler": r["sampler"], "seed": r["seed"], "initialization": r["initialization"],
                "experiment_id": r["experiment_id"],
                "epoch": e.get("epoch"), "train_loss": e.get("train_loss"),
                "val_auc": e.get("val_auc"), "val_f1_macro": e.get("val_f1_macro"),
            })
    return pd.DataFrame(recs)


METRICS = ["test_auc", "test_f1_macro", "test_precision_fraud", "test_recall_fraud",
           "test_f1_fraud", "test_accuracy", "test_ks", "test_ece"]


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    ok = df[df["status"] == "completed"]
    agg = ok.groupby(group_cols, observed=True)[METRICS].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    # std is NaN for n=1; make that explicit rather than leaving a blank cell.
    for m in METRICS:
        agg[f"{m}_std"] = agg[f"{m}_std"].fillna(0.0)
    degen = ok.groupby(group_cols, observed=True)["is_degenerate_all_majority"].agg(
        ["sum", "count"]
    ).reset_index()
    degen = degen.rename(columns={"sum": "n_degenerate_all_majority", "count": "n_runs_checked"})
    agg = agg.merge(degen, on=group_cols, how="left")
    return agg


def bar_with_err(ax, labels, groups: dict[str, tuple[np.ndarray, np.ndarray]], ylabel, title,
                  degenerate: dict[str, np.ndarray] | None = None):
    """groups: {group_name: (means_array, stds_array)} aligned to `labels`.

    `degenerate[name]` (optional): boolean array aligned to `labels`, True where EVERY run in
    that cell predicted zero fraud cases (research/degenerate_baselines.md). Marked with a red
    'D' above the bar so a reader cannot mistake a class-ratio artefact for a learned result.
    """
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    x = np.arange(len(labels))
    for i, (name, (means, stds)) in enumerate(groups.items()):
        xpos = x + i * width - 0.4 + width / 2
        ax.bar(xpos, means, width, yerr=stds, capsize=3, label=name)
        if degenerate is not None and name in degenerate:
            for xp, m, s, deg in zip(xpos, means, stds, degenerate[name]):
                if deg and np.isfinite(m):
                    ax.annotate("D", (xp, m + (s or 0)), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=8, color="red",
                                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def make_headline_figures(summary_core: pd.DataFrame, figdir: pathlib.Path) -> None:
    for metric, label in [("test_auc", "AUC"), ("test_f1_macro", "F1-macro"),
                           ("test_recall_fraud", "Recall (fraud class)")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            sub = summary_core[summary_core["dataset"] == dataset]
            labels = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
            groups = {}
            degenerate = {}
            for variant, sampler in [("baseline", "none"), ("text", "none"), ("flag", "cosine")]:
                s2 = sub[(sub["variant"] == variant) & (sub["sampler"] == sampler)]
                s2 = s2.set_index(s2["model"].astype(str))
                means = np.array([s2[f"{metric}_mean"].get(m, np.nan) for m in labels])
                stds = np.array([s2[f"{metric}_std"].get(m, 0.0) for m in labels])
                name = variant if sampler == "none" else f"{variant} ({sampler})"
                groups[name] = (means, stds)
                degenerate[name] = np.array([
                    s2["n_degenerate_all_majority"].get(m, 0) == s2["n_runs_checked"].get(m, -1)
                    for m in labels
                ])
            bar_with_err(ax, labels, groups, label, f"{dataset}", degenerate=degenerate)
        fig.suptitle(f"{label} by model and variant (error bars = std across seeds; "
                     f"baseline/text n=1, flag n=8; red 'D' = every run in that bar predicted "
                     f"zero fraud cases - degenerate_baselines.md)")
        fig.tight_layout()
        fig.savefig(figdir / f"headline_{metric}.png", dpi=150)
        plt.close(fig)


def make_sampler_ablation_figures(summary_core: pd.DataFrame, figdir: pathlib.Path) -> None:
    for variant, samplers in [("text", TEXT_SAMPLERS), ("flag", FLAG_SAMPLERS)]:
        for metric, label in [("test_auc", "AUC"), ("test_f1_macro", "F1-macro")]:
            for dataset in DATASETS:
                sub = summary_core[(summary_core["dataset"] == dataset)
                                    & (summary_core["variant"] == variant)]
                labels = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
                groups = {}
                degenerate = {}
                for sampler in samplers:
                    s2 = sub[sub["sampler"] == sampler]
                    s2 = s2.set_index(s2["model"].astype(str))
                    means = np.array([s2[f"{metric}_mean"].get(m, np.nan) for m in labels])
                    stds = np.array([s2[f"{metric}_std"].get(m, 0.0) for m in labels])
                    groups[sampler] = (means, stds)
                    degenerate[sampler] = np.array([
                        s2["n_degenerate_all_majority"].get(m, 0) == s2["n_runs_checked"].get(m, -1)
                        for m in labels
                    ])
                fig, ax = plt.subplots(figsize=(9, 5))
                bar_with_err(ax, labels, groups, label,
                             f"{variant} variant, sampler ablation - {dataset} "
                             f"(n=8 runs/bar: 4 seeds x 2 inits; red 'D' = all 8 runs degenerate)",
                             degenerate=degenerate)
                fig.tight_layout()
                fig.savefig(figdir / f"sampler_ablation_{variant}_{metric}__{dataset}.png", dpi=150)
                plt.close(fig)


def make_training_curve_grids(epochs_df: pd.DataFrame, curvedir: pathlib.Path) -> None:
    # -- raw baseline/text: single run each, one grid per dataset, models x variant ------
    for dataset in DATASETS:
        sub = epochs_df[(epochs_df["dataset"] == dataset) & (epochs_df["sampler"] == "none")]
        models = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
        fig, axes = plt.subplots(len(models), 2, figsize=(10, 2.2 * len(models)), squeeze=False)
        for i, model in enumerate(models):
            for j, variant in enumerate(["baseline", "text"]):
                ax = axes[i][j]
                s2 = sub[(sub["model"].astype(str) == model) & (sub["variant"] == variant)]
                s2 = s2.sort_values("epoch")
                if s2.empty:
                    ax.set_visible(False)
                    continue
                ax.plot(s2["epoch"], s2["train_loss"], "o-", color="tab:red", label="train_loss")
                ax2 = ax.twinx()
                ax2.plot(s2["epoch"], s2["val_auc"], "s--", color="tab:blue", label="val_auc")
                ax2.plot(s2["epoch"], s2["val_f1_macro"], "^--", color="tab:green", label="val_f1_macro")
                ax.set_title(f"{model} / {variant}", fontsize=9)
                ax.tick_params(labelsize=7)
                ax2.tick_params(labelsize=7)
                ax2.set_ylim(0, 1)
                if i == 0 and j == 0:
                    lines = ax.get_lines() + ax2.get_lines()
                    fig.legend(lines, [l.get_label() for l in lines], loc="upper center",
                               ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{dataset}: baseline vs text training curves "
                     f"(single run, seed=0 init=0; red=train loss (left axis), "
                     f"blue/green=val AUC / F1-macro (right axis, 0-1))", y=1.05, fontsize=10)
        fig.tight_layout()
        fig.savefig(curvedir / f"baseline_text__{dataset}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -- flag_md: mean curve over seeds, one grid per (dataset, variant), lines=samplers --
    for variant, samplers in [("text", TEXT_SAMPLERS), ("flag", FLAG_SAMPLERS)]:
        for dataset in DATASETS:
            sub = epochs_df[(epochs_df["dataset"] == dataset) & (epochs_df["variant"] == variant)
                             & (epochs_df["sampler"].isin(samplers))]
            models = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
            fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), squeeze=False)
            axes = axes[0]
            for ax, model in zip(axes, models):
                s2 = sub[sub["model"].astype(str) == model]
                for sampler in samplers:
                    s3 = s2[s2["sampler"] == sampler]
                    if s3.empty:
                        continue
                    grp = s3.groupby("epoch")["val_auc"].agg(["mean", "std"]).reset_index()
                    ax.plot(grp["epoch"], grp["mean"], marker="o", markersize=3, label=sampler)
                    ax.fill_between(grp["epoch"], grp["mean"] - grp["std"].fillna(0),
                                     grp["mean"] + grp["std"].fillna(0), alpha=0.15)
                ax.set_title(model, fontsize=9)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel("val_auc (mean +/- std over 8 runs)")
            fig.suptitle(f"{variant} variant - {dataset}: validation AUC per epoch by sampler",
                         fontsize=10)
            handles, labels_ = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc="upper center", ncol=len(samplers), fontsize=8,
                       bbox_to_anchor=(0.5, 1.12))
            fig.tight_layout()
            fig.savefig(curvedir / f"{variant}_variant_val_auc__{dataset}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), squeeze=False)
            axes = axes[0]
            for ax, model in zip(axes, models):
                s2 = sub[sub["model"].astype(str) == model]
                for sampler in samplers:
                    s3 = s2[s2["sampler"] == sampler]
                    if s3.empty:
                        continue
                    grp = s3.groupby("epoch")["train_loss"].agg(["mean", "std"]).reset_index()
                    ax.plot(grp["epoch"], grp["mean"], marker="o", markersize=3, label=sampler)
                    ax.fill_between(grp["epoch"], grp["mean"] - grp["std"].fillna(0),
                                     grp["mean"] + grp["std"].fillna(0), alpha=0.15)
                ax.set_title(model, fontsize=9)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel("train_loss (mean +/- std over 8 runs)")
            fig.suptitle(f"{variant} variant - {dataset}: training loss per epoch by sampler",
                         fontsize=10)
            handles, labels_ = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc="upper center", ncol=len(samplers), fontsize=8,
                       bbox_to_anchor=(0.5, 1.12))
            fig.tight_layout()
            fig.savefig(curvedir / f"{variant}_variant_train_loss__{dataset}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "summary").mkdir(exist_ok=True)
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    curvedir = figdir / "training_curves"
    curvedir.mkdir(exist_ok=True)

    df = collect_runs()
    print(f"loaded {len(df)} run records "
          f"({(df['status'] == 'completed').sum()} completed, "
          f"{(df['status'] != 'completed').sum()} not-completed)")

    flat = df.drop(columns=["epoch_history"])
    flat.to_csv(OUT / "data" / "all_runs.csv", index=False)

    failed = flat[flat["status"] != "completed"]
    failed.to_csv(OUT / "data" / "failed_or_incomplete_runs.csv", index=False)
    print(f"wrote data/all_runs.csv ({len(flat)} rows), "
          f"data/failed_or_incomplete_runs.csv ({len(failed)} rows)")

    epochs_df = explode_epochs(df)
    epochs_df.to_csv(OUT / "data" / "epoch_history_long.csv", index=False)
    print(f"wrote data/epoch_history_long.csv ({len(epochs_df)} rows)")

    summary_core = summarize(df, ["dataset", "model", "variant", "sampler"])
    summary_core.to_csv(OUT / "summary" / "summary_by_dataset_model_variant_sampler.csv",
                        index=False)

    headline = summary_core[
        ((summary_core["variant"] == "baseline") & (summary_core["sampler"] == "none"))
        | ((summary_core["variant"] == "text") & (summary_core["sampler"] == "none"))
        | ((summary_core["variant"] == "flag") & (summary_core["sampler"] == "cosine"))
    ].sort_values(["dataset", "model", "variant"])
    headline.to_csv(OUT / "summary" / "headline_baseline_text_flag.csv", index=False)

    text_ablation = summary_core[summary_core["variant"] == "text"].sort_values(
        ["dataset", "model", "sampler"])
    text_ablation.to_csv(OUT / "summary" / "sampler_ablation_text_variant.csv", index=False)

    flag_ablation = summary_core[summary_core["variant"] == "flag"].sort_values(
        ["dataset", "model", "sampler"])
    flag_ablation.to_csv(OUT / "summary" / "sampler_ablation_flag_variant.csv", index=False)

    degenerate_runs = flat[flat["is_degenerate_all_majority"]].sort_values(
        ["dataset", "variant", "model", "sampler", "seed", "initialization"])
    degenerate_runs.to_csv(OUT / "data" / "degenerate_all_majority_runs.csv", index=False)
    print(f"degenerate (predicted-zero-fraud) runs: {len(degenerate_runs)} / {len(flat)}")

    (OUT / "summary" / "flag_finetuned_NOT_AVAILABLE.md").write_text(
        "# flag_finetuned (+FLAG*) - NOT AVAILABLE\n\n"
        "No run of this variant exists in results/raw/ or results/flag_md/ at the time this "
        "report was built. It requires a GPU-generated 'residual' LLM-text cache "
        "(cache/llm/*residual*.json, then its Sentence-BERT encoding in cache/embeddings/) "
        "that is not present on this machine: cache/llm/ contains 0 files.\n\n"
        "This report does not substitute, estimate or interpolate a number for this cell.\n\n"
        "To add it: generate the discriminative AND residual LLM text on a GPU "
        "(`python -m scripts.llm.generate_text --kind both`), encode it "
        "(`python -m scripts.preprocess.encode_llm_text --kind both`), then run "
        "`python -m scripts.train.run --variant flag_finetuned ...` and re-run this script.\n",
        encoding="utf-8",
    )

    make_headline_figures(summary_core, figdir)
    make_sampler_ablation_figures(summary_core, figdir)
    make_training_curve_grids(epochs_df, curvedir)
    print(f"wrote figures to {figdir.relative_to(ROOT)} "
          f"({len(list(figdir.rglob('*.png')))} PNG files)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
