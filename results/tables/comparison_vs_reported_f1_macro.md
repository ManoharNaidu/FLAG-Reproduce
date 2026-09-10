# Reproduction vs reported — F1-macro

Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.
Tolerance: MATCH within ±1.0 pp, CLOSE within ±2.0 pp.

Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 was produced by the FLAG authors' own baseline rewrites, so an `official` baseline is not comparable to it (decision D-002).

The reported column is a **reference target**. No result here has been adjusted toward it, and reported values are never written into `results/`.

| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| instagram | bwgnn | baseline | 47.28±0.01 | 49.32±0.00 | +2.04 | +4.3% | 1 | DEVIATION |
| instagram | bwgnn | flag | 48.55±0.32 | - |  |  | 0 | UNAVAILABLE |
| instagram | bwgnn | flag_finetuned | 49.35±1.45 | - |  |  | 0 | UNAVAILABLE |
| instagram | bwgnn | text | 47.61±0.72 | 47.62±0.00 | +0.01 | +0.0% | 1 | MATCH |
| instagram | care_gnn | baseline | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | care_gnn | flag | 49.64±1.74 | - |  |  | 0 | UNAVAILABLE |
| instagram | care_gnn | flag_finetuned | 50.24±0.46 | - |  |  | 0 | UNAVAILABLE |
| instagram | care_gnn | text | 48.05±1.07 | 47.62±0.00 | -0.43 | -0.9% | 1 | MATCH |
| instagram | dga_gnn | baseline | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | dga_gnn | flag | 48.53±0.42 | - |  |  | 0 | UNAVAILABLE |
| instagram | dga_gnn | flag_finetuned | 48.95±0.68 | - |  |  | 0 | UNAVAILABLE |
| instagram | dga_gnn | text | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | gat | baseline | 49.21±1.38 | 48.19±0.00 | -1.02 | -2.1% | 1 | CLOSE |
| instagram | gat | flag | 49.13±1.23 | - |  |  | 0 | UNAVAILABLE |
| instagram | gat | flag_finetuned | 50.65±1.61 | - |  |  | 0 | UNAVAILABLE |
| instagram | gat | text | 48.31±0.98 | 47.62±0.00 | -0.69 | -1.4% | 1 | MATCH |
| instagram | gcn | baseline | 47.88±0.96 | 51.91±0.00 | +4.03 | +8.4% | 1 | DEVIATION |
| instagram | gcn | flag | 48.05±0.69 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | flag_finetuned | 49.79±1.08 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | text | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | geniepath | baseline | 47.31±0.05 | 50.85±0.00 | +3.54 | +7.5% | 1 | DEVIATION |
| instagram | geniepath | flag | 48.41±1.51 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | flag_finetuned | 48.28±1.25 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | text | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | pmp | baseline | 47.50±1.56 | 48.05±0.00 | +0.55 | +1.2% | 1 | MATCH |
| instagram | pmp | flag | 48.48±0.82 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | flag_finetuned | 49.76±1.64 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | text | 48.29±0.01 | 47.62±0.00 | -0.67 | -1.4% | 1 | MATCH |
| reddit | bwgnn | baseline | 45.47±0.01 | 52.81±0.00 | +7.34 | +16.1% | 1 | DEVIATION |
| reddit | bwgnn | flag | 50.93±2.16 | - |  |  | 0 | UNAVAILABLE |
| reddit | bwgnn | flag_finetuned | 51.91±2.38 | - |  |  | 0 | UNAVAILABLE |
| reddit | bwgnn | text | 48.76±1.54 | 53.41±0.00 | +4.65 | +9.5% | 1 | DEVIATION |
| reddit | care_gnn | baseline | 45.46±0.01 | 47.62±0.00 | +2.16 | +4.7% | 1 | DEVIATION |
| reddit | care_gnn | flag | 50.78±0.95 | - |  |  | 0 | UNAVAILABLE |
| reddit | care_gnn | flag_finetuned | 51.95±2.21 | - |  |  | 0 | UNAVAILABLE |
| reddit | care_gnn | text | 47.66±1.59 | 47.62±0.00 | -0.04 | -0.1% | 1 | MATCH |
| reddit | dga_gnn | baseline | 45.46±0.01 | 47.62±0.00 | +2.16 | +4.7% | 1 | DEVIATION |
| reddit | dga_gnn | flag | 48.77±2.18 | - |  |  | 0 | UNAVAILABLE |
| reddit | dga_gnn | flag_finetuned | 49.50±0.83 | - |  |  | 0 | UNAVAILABLE |
| reddit | dga_gnn | text | 45.49±0.11 | 47.62±0.00 | +2.13 | +4.7% | 1 | DEVIATION |
| reddit | gat | baseline | 46.66±0.96 | 47.62±0.00 | +0.96 | +2.1% | 1 | MATCH |
| reddit | gat | flag | 49.70±1.76 | - |  |  | 0 | UNAVAILABLE |
| reddit | gat | flag_finetuned | 50.20±2.33 | - |  |  | 0 | UNAVAILABLE |
| reddit | gat | text | 48.26±2.23 | 47.62±0.00 | -0.64 | -1.3% | 1 | MATCH |
| reddit | gcn | baseline | 45.46±0.01 | 49.58±0.00 | +4.12 | +9.1% | 1 | DEVIATION |
| reddit | gcn | flag | 48.19±1.02 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | flag_finetuned | 48.72±1.59 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | text | 45.84±0.36 | 48.32±0.00 | +2.48 | +5.4% | 1 | DEVIATION |
| reddit | geniepath | baseline | 45.46±0.01 | 53.48±0.00 | +8.02 | +17.6% | 1 | DEVIATION |
| reddit | geniepath | flag | 48.30±2.24 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | flag_finetuned | 48.69±2.97 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | text | 46.84±1.89 | 47.91±0.00 | +1.07 | +2.3% | 1 | CLOSE |
| reddit | pmp | baseline | 46.31±1.04 | 47.62±0.00 | +1.31 | +2.8% | 1 | CLOSE |
| reddit | pmp | flag | 48.91±1.30 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | flag_finetuned | 50.14±1.48 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | text | 47.21±1.18 | 47.62±0.00 | +0.41 | +0.9% | 1 | MATCH |

## Summary

- UNAVAILABLE: 28
- MATCH: 14
- DEVIATION: 11
- CLOSE: 3

## Why exact agreement is not achievable

- The paper's 1:10 downsampling seed is unpublished, so our benchmark is a different draw of the same construction.
- The paper states no train/val/test split; ours is 10/10/80 inherited from GraphAdapter/GLBench.
- The paper states no decision-threshold policy. `argmax` reproduces the released code; `validation_swept` reproduces BWGNN's setup, which the paper says it follows. The two differ by several F1 points.
- Semantic sampling is reimplemented from Eq. 3-4; no upstream source exists.
- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/`--inits` here gives a std that is not comparable.

