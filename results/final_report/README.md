# Final results report

Generated entirely from **existing, already-completed run files** already sitting in this
repository (`results/raw/` and `results/flag_md/`). **No training was run to produce this
report** — that was your explicit choice (see "Scope" below). Built by
[`scripts/analyze/build_final_report.py`](../../scripts/analyze/build_final_report.py); re-run
it any time with:

```bash
.venv-report/Scripts/python.exe -m scripts.analyze.build_final_report
```

(`.venv-report/` is a small venv with only `pandas` + `matplotlib` + `numpy` — created for this
report so the project's pinned training environment, `.venv-cpu`, was not touched. `.venv-cpu`
is currently broken on this machine: it points at a Python 3.11 install that no longer exists.
That did not block this report because no training/inference was needed.)

## What you asked for, and what's actually here

| You asked for | Status |
|---|---|
| baseline | **Included.** 1 run/model/dataset (seed 0, init 0). |
| text | **Included**, two ways: (1) the canonical 1-run cell, no sampling (same protocol as baseline); (2) a 48x-larger sampler-ablation set (6 samplers x 4 seeds x 2 inits) that reuses the `text` variant's features — see "Two different `text` result sets" below. |
| flag | **Included.** 8 runs/model/dataset (4 seeds x 2 inits) under the cosine sampler, plus the same 8 runs under one Markov-diffusion config (`md_K2_matched`). |
| flag_finetuned | **NOT INCLUDED — does not exist anywhere in this repository.** It needs a GPU-generated "residual" LLM-text cache; `cache/llm/` has 0 files on this machine. Not fabricated, not estimated. See [`summary/flag_finetuned_NOT_AVAILABLE.md`](summary/flag_finetuned_NOT_AVAILABLE.md). You said you have GPU access/caches elsewhere — tell me how to reach them (path to copy in, or a remote instance) and I will run it for real and add it here. |
| DGP | **Excluded, per your instruction.** DGP is not an implemented method in this codebase; it is cited only as the idea source for FLAG-MD's diffusion operator. |
| FLAG-MD | **Included** as the sampler-ablation set (see above): cosine vs. 5 Markov-diffusion configurations, for both the `text` and `flag` variants. |
| AUC numbers | **Included** — real, from each run's stored `sklearn.roc_auc_score` result. |
| **AUC curves (ROC, TPR vs FPR)** | **NOT POSSIBLE from stored data.** No result JSON, and no saved checkpoint (`checkpoints/` is empty), contains raw per-example prediction scores/labels. Only the scalar AUC was ever persisted. Producing a real ROC curve needs re-running inference with score-dumping added — that is new compute, which is out of scope for "reuse existing results only." What you have instead: bar charts of the AUC scalar (mean +/- std across seeds), which is real data, just not a curve. |
| F1-macro numbers | **Included.** |
| Recall | **Included** (`recall_fraud` = recall of the fraud/minority class; this is the metric the repo tracks). |
| Loss curves | **Included, and real** — per-epoch `train_loss`, `val_auc`, `val_f1_macro` were stored by the original training runs and are plotted as-is (5 epochs each; that's the run configuration, not a truncation on my part). |
| "everything" | Precision, F1-fraud, accuracy, KS, ECE are also included in the CSVs (all fields the runs recorded), even though not explicitly named. |

## Focused reports

- [`flag_vs_flag_md.md`](flag_vs_flag_md.md) — **flag (cosine) vs FLAG-MD, paired comparison.**
  Built by [`scripts/analyze/build_flag_vs_flagmd_report.py`](../../scripts/analyze/build_flag_vs_flagmd_report.py)
  (reads only `data/all_runs.csv`, no new training). Uses a *paired* t-test per (seed, init) —
  cosine and FLAG-MD runs share the same seed/init streams by the project's own design
  (`run_flag_md_matrix.sh`), so this is more sensitive than comparing two independent means.
  Headline: essentially no distinguishable difference between the two samplers at n=8 pairs.

## Folder layout

```text
results/final_report/
├── README.md                                   <- this file
├── flag_vs_flag_md.md                          focused flag-vs-FLAG-MD paired comparison (see above)
├── data/
│   ├── all_runs.csv                            one row per run, every stored field (924 rows)
│   ├── epoch_history_long.csv                  one row per (run, epoch): train_loss, val_auc, val_f1_macro
│   ├── failed_or_incomplete_runs.csv            rows with status != "completed" (0 rows: everything on disk succeeded)
│   ├── degenerate_all_majority_runs.csv         runs that predicted ZERO fraud cases (see below)
│   └── flag_vs_flag_md_paired_stats.csv        per-model paired deltas + t-test p-values (flag vs FLAG-MD)
├── summary/
│   ├── summary_by_dataset_model_variant_sampler.csv   mean/std/count of every metric, every cell
│   ├── headline_baseline_text_flag.csv          the Table-4-style comparison (baseline / text-no-sampling / flag-cosine)
│   ├── sampler_ablation_text_variant.csv        text variant x 6 samplers
│   ├── sampler_ablation_flag_variant.csv        flag variant x 2 samplers
│   └── flag_finetuned_NOT_AVAILABLE.md          why this cell is empty, and how to fill it
└── figures/
    ├── headline_test_auc.png                    AUC, baseline vs text vs flag(cosine), both datasets
    ├── headline_test_f1_macro.png                same, F1-macro
    ├── headline_test_recall_fraud.png            same, recall (fraud class)
    ├── sampler_ablation_{text,flag}_{test_auc,test_f1_macro}__{reddit,instagram}.png   (8 files)
    ├── flag_vs_flag_md_delta_{auc,f1_macro}.png    paired-delta charts (see flag_vs_flag_md.md)
    └── training_curves/
        ├── baseline_text__{reddit,instagram}.png              per-model, per-epoch curves, single run
        ├── {text,flag}_variant_val_auc__{reddit,instagram}.png     mean val-AUC per epoch, one line per sampler
        └── {text,flag}_variant_train_loss__{reddit,instagram}.png  mean train-loss per epoch, one line per sampler
```

## Two different `text` result sets — read this before comparing numbers

There are **two distinct sources** for the `text` variant, and they measure different things:

1. **`results/raw/*text*` (n=1 per model/dataset).** Sampler = `none` (no cosine sampling at
   all — the registry's actual default for `text`; see `registry.py VariantSpec.default_sampling_strategy`).
   This is the "canonical" `+text` cell, directly comparable to `baseline`.
2. **`results/flag_md/text/<sampler>/*` (n=8 per model/dataset/sampler, 6 samplers).** This is a
   **separate ablation study** built by `scripts/reproduce/run_flag_md_matrix.sh`: it forces the
   `text` variant's LLM-free features through the *sampler comparison* (cosine vs 5
   Markov-diffusion configs) as a cheap proxy for how the sampler affects a subgraph, without
   needing the GPU/LLM stage. **It is not the canonical `+text` cell** — every row in it uses
   sampling the registry would not normally give `text`.

I kept these separate in the CSVs (`headline_baseline_text_flag.csv` vs
`sampler_ablation_text_variant.csv`) rather than mixing them, and the same applies to `flag`:
`flag/cosine` is the canonical `+FLAG` cell (registry default sampler for `flag`); `flag/md_K2_matched`
is the same ablation idea applied to the real FLAG pipeline.

## Headline finding you should see before reading the numbers: degenerate cells

**17 of 924 runs predicted zero fraud cases** (`precision_fraud = recall_fraud = 0.0` — the model
labelled every single test example as "not fraud"). All 17 are in the single-run
`baseline`/`text` cells (never in the 8-run `flag` cells). Their F1-macro is not a measured
model skill — it is the closed-form value for "always predict majority" on that test split's
class ratio (`p0 / (1 + p0)`, `research/degenerate_baselines.md`, already documented in this
repo from the paper audit). Concretely: Reddit CARE-GNN/DGA-GNN/GAT/PMP (`baseline` and `text`)
and Instagram CARE-GNN/DGA-GNN/GAT/PMP/GCN/GeniePath/BWGNN (`text` only) hit this. Full list:
[`data/degenerate_all_majority_runs.csv`](data/degenerate_all_majority_runs.csv).

These are marked with a red **`D`** directly above the bar in `headline_test_f1_macro.png`,
`headline_test_recall_fraud.png`, and the sampler-ablation F1/recall-style charts, so you cannot
mis-read a flat class-ratio number as a learned result. **This is a real property of these single
runs, not an artefact of my analysis** — but note it could also be a **seed** effect: since
`baseline`/`text` here are n=1, I cannot tell you whether a different seed would also collapse.
The `flag` variant's 8-seed cells never fully collapse on any model, which is itself informative,
but is confounded with `flag`'s different features, not just its seed count — I want to be clear
I'm not claiming FLAG "fixes" this from one data point.

## Statistical caveat you must not ignore

- `baseline` and `text` (canonical, no-sampling): **n=1** per (dataset, model). No error bars
  are meaningful; a std of 0.0 in the CSV means literally one observation, not zero variance.
- `text` (sampler ablation) and `flag`: **n=8** (4 seeds x 2 inits) per (dataset, model, sampler).
  Error bars (std) in the figures are real across those 8 runs.
- The paper's protocol is 25 runs (5 seeds x 5 inits) per cell. Nothing here reaches that; you
  chose "reuse existing results only," so no new runs were launched to close that gap.

## What's excluded and why

- `results/flag_md/*/gpu_profile/` (4 files) — GPU timing-profile runs, not part of the results
  grid; excluded from every table and figure.
- `flag_finetuned` — see table above.
- DGP — excluded per your instruction (not implemented here).
- True ROC curves — see table above.

## Reproducing / extending this report

Everything above is derived, not hand-entered. To regenerate after adding more runs (e.g. more
baseline/text seeds, or a real `flag_finetuned` once you provide GPU access), just re-run:

```bash
.venv-report/Scripts/python.exe -m scripts.analyze.build_final_report
```

It re-scans `results/raw/` and `results/flag_md/` from scratch and overwrites everything under
`results/final_report/`.
