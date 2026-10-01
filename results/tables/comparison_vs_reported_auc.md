# Reproduction vs reported — AUC

Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.
Tolerance: MATCH within ±1.0 pp, CLOSE within ±2.0 pp.

Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 was produced by the FLAG authors' own baseline rewrites, so an `official` baseline is not comparable to it (decision D-002).

The reported column is a **reference target**. No result here has been adjusted toward it, and reported values are never written into `results/`.

| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| instagram | bwgnn | baseline | 51.52±2.43 | 54.73±0.00 | +3.21 | +6.2% | 1 | DEVIATION |
| instagram | bwgnn | flag | 56.33±0.72 | 58.04±1.13 | +1.71 | +3.0% | 31 | CLOSE |
| instagram | bwgnn | flag_finetuned | 57.19±0.28 | 57.54±2.52 | +0.35 | +0.6% | 25 | MATCH |
| instagram | bwgnn | text | 54.10±0.71 | - |  |  | 0 | UNAVAILABLE |
| instagram | care_gnn | baseline | 52.07±1.95 | 49.91±0.00 | -2.16 | -4.1% | 1 | DEVIATION |
| instagram | care_gnn | flag | 55.79±0.58 | 61.01±0.96 | +5.22 | +9.3% | 25 | DEVIATION |
| instagram | care_gnn | flag_finetuned | 56.40±1.29 | 59.98±2.29 | +3.58 | +6.3% | 25 | DEVIATION |
| instagram | care_gnn | text | 54.92±0.38 | - |  |  | 0 | UNAVAILABLE |
| instagram | dga_gnn | baseline | 50.86±0.48 | 50.97±0.00 | +0.11 | +0.2% | 1 | MATCH |
| instagram | dga_gnn | flag | 56.73±0.66 | 60.55±1.04 | +3.82 | +6.7% | 25 | DEVIATION |
| instagram | dga_gnn | flag_finetuned | 57.20±1.03 | 59.63±2.47 | +2.43 | +4.2% | 25 | DEVIATION |
| instagram | dga_gnn | text | 56.28±1.07 | - |  |  | 0 | UNAVAILABLE |
| instagram | gat | baseline | 51.53±1.12 | 48.73±0.00 | -2.80 | -5.4% | 1 | DEVIATION |
| instagram | gat | flag | 54.97±0.75 | 60.49±1.25 | +5.52 | +10.0% | 25 | DEVIATION |
| instagram | gat | flag_finetuned | 55.98±1.67 | 57.85±3.63 | +1.87 | +3.3% | 25 | CLOSE |
| instagram | gat | text | 54.69±0.76 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | baseline | 52.61±1.80 | 53.41±0.00 | +0.80 | +1.5% | 1 | MATCH |
| instagram | gcn | flag | 56.31±0.83 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | flag_finetuned | 55.45±1.21 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | text | 55.74±0.73 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | baseline | 51.22±3.16 | 51.20±0.00 | -0.02 | -0.0% | 1 | MATCH |
| instagram | geniepath | flag | 55.59±0.85 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | flag_finetuned | 56.24±2.10 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | text | 52.45±2.31 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | baseline | 50.63±0.52 | 56.10±0.00 | +5.47 | +10.8% | 1 | DEVIATION |
| instagram | pmp | flag | 57.10±0.62 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | flag_finetuned | 57.67±1.09 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | text | 56.05±1.27 | - |  |  | 0 | UNAVAILABLE |
| reddit | bwgnn | baseline | 53.82±2.49 | 55.61±0.00 | +1.79 | +3.3% | 1 | CLOSE |
| reddit | bwgnn | flag | 58.89±2.50 | 62.03±1.77 | +3.14 | +5.3% | 27 | DEVIATION |
| reddit | bwgnn | flag_finetuned | 59.20±1.09 | 61.71±2.01 | +2.51 | +4.2% | 25 | DEVIATION |
| reddit | bwgnn | text | 57.56±1.53 | - |  |  | 0 | UNAVAILABLE |
| reddit | care_gnn | baseline | 51.35±1.62 | 50.11±0.00 | -1.24 | -2.4% | 1 | CLOSE |
| reddit | care_gnn | flag | 58.43±0.65 | 62.86±0.67 | +4.43 | +7.6% | 25 | DEVIATION |
| reddit | care_gnn | flag_finetuned | 58.74±1.40 | 62.91±1.01 | +4.17 | +7.1% | 25 | DEVIATION |
| reddit | care_gnn | text | 56.72±1.38 | - |  |  | 0 | UNAVAILABLE |
| reddit | dga_gnn | baseline | 50.10±0.53 | 53.35±0.00 | +3.25 | +6.5% | 1 | DEVIATION |
| reddit | dga_gnn | flag | 61.05±0.71 | 62.44±0.71 | +1.39 | +2.3% | 25 | CLOSE |
| reddit | dga_gnn | flag_finetuned | 61.61±0.78 | 62.07±0.98 | +0.46 | +0.7% | 25 | MATCH |
| reddit | dga_gnn | text | 59.59±1.60 | - |  |  | 0 | UNAVAILABLE |
| reddit | gat | baseline | 52.66±2.25 | 55.59±0.00 | +2.93 | +5.6% | 1 | DEVIATION |
| reddit | gat | flag | 60.61±1.20 | 64.33±0.68 | +3.72 | +6.1% | 25 | DEVIATION |
| reddit | gat | flag_finetuned | 60.57±1.03 | 63.94±0.77 | +3.37 | +5.6% | 25 | DEVIATION |
| reddit | gat | text | 59.32±0.29 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | baseline | 50.32±0.26 | 59.74±0.00 | +9.42 | +18.7% | 2 | DEVIATION |
| reddit | gcn | flag | 60.18±0.79 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | flag_finetuned | 60.88±0.68 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | text | 57.82±1.94 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | baseline | 52.18±1.48 | 58.06±0.00 | +5.88 | +11.3% | 1 | DEVIATION |
| reddit | geniepath | flag | 59.43±0.55 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | flag_finetuned | 59.74±1.68 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | text | 56.91±1.85 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | baseline | 50.16±0.12 | 51.79±0.00 | +1.63 | +3.3% | 1 | CLOSE |
| reddit | pmp | flag | 61.32±0.66 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | flag_finetuned | 61.80±0.99 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | text | 59.79±0.43 | - |  |  | 0 | UNAVAILABLE |

## Summary

- UNAVAILABLE: 26
- DEVIATION: 19
- CLOSE: 6
- MATCH: 5

## Why exact agreement is not achievable

- The paper's 1:10 downsampling seed is unpublished, so our benchmark is a different draw of the same construction.
- The paper states no train/val/test split; ours is 10/10/80 inherited from GraphAdapter/GLBench.
- The paper states no decision-threshold policy. `argmax` reproduces the released code; `validation_swept` reproduces BWGNN's setup, which the paper says it follows. The two differ by several F1 points.
- Semantic sampling is reimplemented from Eq. 3-4; no upstream source exists.
- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/`--inits` here gives a std that is not comparable.

