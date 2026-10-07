# FLAG (cosine) vs FLAG-MD (Markov-diffusion, K=2, matched) - paired comparison

[Back to final report](README.md)

Source: `results/final_report/data/all_runs.csv` (built by
`scripts.analyze.build_final_report` from `results/flag_md/flag/{cosine,md_K2_matched}/`).
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

Across the 28 paired tests on AUC and F1-macro (14 model/dataset cells x
2 metrics), **1 was significant at p<0.05** -- about what you'd expect from chance
alone at that threshold (~5% of 28 is ~1.4). Widening to all 4
metrics tracked (AUC, F1-macro, recall, precision), it's **2 of 56**,
same story. **The honest reading is that this repo's data does not show a real, consistent
difference between the cosine sampler and this one Markov-diffusion configuration (K=2,
matched neighbour count), for either metric, on either dataset, for any of the 7 backbones.**
Any single model/dataset cell that looks like a win or a loss in the raw numbers is very
likely sampling noise from only 4 seeds x 2 inits, not a real effect of the sampler.

## Reddit

| Model | AUC cosine | AUC FLAG-MD | ΔAUC (paired) | p (paired t-test) | F1 cosine | F1 FLAG-MD | ΔF1 (paired) | p | Recall cosine | Recall FLAG-MD |
|---|---|---|---|---|---|---|---|---|---|---|
| gcn | 0.6143 | 0.6110 | -0.0034 ± 0.0052 | 0.108 | 0.5284 | 0.5269 | -0.0014 ± 0.0067 | 0.562 | 0.1402 | 0.1380 |
| gat | 0.6432 | 0.6408 | -0.0024 ± 0.0093 | 0.484 | 0.5442 | 0.5418 | -0.0024 ± 0.0086 | 0.458 | 0.2294 | 0.2201 |
| geniepath | 0.6144 | 0.6074 | -0.0070 ± 0.0062 | 0.016 | 0.5342 | 0.5334 | -0.0009 ± 0.0073 | 0.750 | 0.1701 | 0.1785 |
| care_gnn | 0.6278 | 0.6286 | +0.0008 ± 0.0022 | 0.341 | 0.5322 | 0.5311 | -0.0010 ± 0.0056 | 0.627 | 0.2067 | 0.1978 |
| bwgnn | 0.6190 | 0.6266 | +0.0076 ± 0.0204 | 0.330 | 0.5428 | 0.5448 | +0.0020 ± 0.0144 | 0.704 | 0.1730 | 0.1867 |
| dga_gnn | 0.6229 | 0.6252 | +0.0023 ± 0.0055 | 0.275 | 0.5312 | 0.5321 | +0.0008 ± 0.0095 | 0.811 | 0.2393 | 0.2091 |
| pmp | 0.6287 | 0.6214 | -0.0073 ± 0.0130 | 0.157 | 0.5348 | 0.5324 | -0.0024 ± 0.0088 | 0.459 | 0.1783 | 0.1929 |

## Instagram

| Model | AUC cosine | AUC FLAG-MD | ΔAUC (paired) | p (paired t-test) | F1 cosine | F1 FLAG-MD | ΔF1 (paired) | p | Recall cosine | Recall FLAG-MD |
|---|---|---|---|---|---|---|---|---|---|---|
| gcn | 0.6013 | 0.6049 | +0.0036 ± 0.0056 | 0.109 | 0.5396 | 0.5402 | +0.0006 ± 0.0028 | 0.543 | 0.1585 | 0.2039 |
| gat | 0.6060 | 0.6033 | -0.0027 ± 0.0230 | 0.749 | 0.5403 | 0.5373 | -0.0030 ± 0.0162 | 0.618 | 0.2275 | 0.2299 |
| geniepath | 0.5687 | 0.5690 | +0.0002 ± 0.0087 | 0.940 | 0.5281 | 0.5273 | -0.0008 ± 0.0095 | 0.821 | 0.1914 | 0.1501 |
| care_gnn | 0.6058 | 0.6037 | -0.0021 ± 0.0180 | 0.756 | 0.5418 | 0.5421 | +0.0003 ± 0.0061 | 0.877 | 0.2147 | 0.2230 |
| bwgnn | 0.5776 | 0.5743 | -0.0033 ± 0.0229 | 0.698 | 0.5300 | 0.5335 | +0.0034 ± 0.0074 | 0.228 | 0.1308 | 0.1510 |
| dga_gnn | 0.6030 | 0.6073 | +0.0043 ± 0.0108 | 0.297 | 0.5385 | 0.5396 | +0.0011 ± 0.0129 | 0.819 | 0.1825 | 0.2342 |
| pmp | 0.5997 | 0.6019 | +0.0022 ± 0.0203 | 0.769 | 0.5422 | 0.5480 | +0.0058 ± 0.0087 | 0.099 | 0.1853 | 0.1892 |

## Figures

- `figures/flag_vs_flag_md_delta_auc.png` -- paired ΔAUC per model, both datasets, with
  paired-t-test significance stars (there are none, see headline above).
- `figures/flag_vs_flag_md_delta_f1_macro.png` -- same for F1-macro.
- Also relevant (from the main report, independent-groups view, not paired):
  `figures/sampler_ablation_flag_test_auc__{reddit,instagram}.png`,
  `figures/sampler_ablation_flag_test_f1_macro__{reddit,instagram}.png`,
  `figures/training_curves/flag_variant_val_auc__{reddit,instagram}.png`.

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
