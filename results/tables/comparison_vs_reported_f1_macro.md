# Reproduction vs reported — F1-macro

Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.
Tolerance: MATCH within ±1.0 pp, CLOSE within ±2.0 pp.

Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 was produced by the FLAG authors' own baseline rewrites, so an `official` baseline is not comparable to it (decision D-002).

The reported column is a **reference target**. No result here has been adjusted toward it, and reported values are never written into `results/`.

| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| instagram | bwgnn | baseline | 47.28±0.01 | 51.90±0.00 | +4.62 | +9.8% | 1 | DEVIATION |
| instagram | bwgnn | flag | 48.55±0.32 | 53.22±0.86 | +4.67 | +9.6% | 31 | DEVIATION |
| instagram | bwgnn | flag_finetuned | 49.35±1.45 | 53.33±1.00 | +3.98 | +8.1% | 25 | DEVIATION |
| instagram | bwgnn | text | 47.61±0.72 | - |  |  | 0 | UNAVAILABLE |
| instagram | care_gnn | baseline | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| instagram | care_gnn | flag | 49.64±1.74 | 54.47±0.61 | +4.83 | +9.7% | 25 | DEVIATION |
| instagram | care_gnn | flag_finetuned | 50.24±0.46 | 54.38±0.48 | +4.14 | +8.2% | 25 | DEVIATION |
| instagram | care_gnn | text | 48.05±1.07 | - |  |  | 0 | UNAVAILABLE |
| instagram | dga_gnn | baseline | 47.29±0.01 | 50.88±0.00 | +3.59 | +7.6% | 1 | DEVIATION |
| instagram | dga_gnn | flag | 48.53±0.42 | 54.01±0.82 | +5.48 | +11.3% | 25 | DEVIATION |
| instagram | dga_gnn | flag_finetuned | 48.95±0.68 | 53.74±0.96 | +4.79 | +9.8% | 25 | DEVIATION |
| instagram | dga_gnn | text | 47.29±0.01 | - |  |  | 0 | UNAVAILABLE |
| instagram | gat | baseline | 49.21±1.38 | 51.56±0.00 | +2.35 | +4.8% | 1 | DEVIATION |
| instagram | gat | flag | 49.13±1.23 | 53.70±0.94 | +4.57 | +9.3% | 25 | DEVIATION |
| instagram | gat | flag_finetuned | 50.65±1.61 | 53.93±0.88 | +3.28 | +6.5% | 25 | DEVIATION |
| instagram | gat | text | 48.31±0.98 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | baseline | 47.88±0.96 | 50.29±0.00 | +2.41 | +5.0% | 1 | DEVIATION |
| instagram | gcn | flag | 48.05±0.69 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | flag_finetuned | 49.79±1.08 | - |  |  | 0 | UNAVAILABLE |
| instagram | gcn | text | 47.29±0.01 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | baseline | 47.31±0.05 | 50.84±0.00 | +3.53 | +7.5% | 1 | DEVIATION |
| instagram | geniepath | flag | 48.41±1.51 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | flag_finetuned | 48.28±1.25 | - |  |  | 0 | UNAVAILABLE |
| instagram | geniepath | text | 47.29±0.01 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | baseline | 47.50±1.56 | 52.58±0.00 | +5.08 | +10.7% | 1 | DEVIATION |
| instagram | pmp | flag | 48.48±0.82 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | flag_finetuned | 49.76±1.64 | - |  |  | 0 | UNAVAILABLE |
| instagram | pmp | text | 48.29±0.01 | - |  |  | 0 | UNAVAILABLE |
| reddit | bwgnn | baseline | 45.47±0.01 | 52.96±0.00 | +7.49 | +16.5% | 1 | DEVIATION |
| reddit | bwgnn | flag | 50.93±2.16 | 54.41±0.69 | +3.48 | +6.8% | 27 | DEVIATION |
| reddit | bwgnn | flag_finetuned | 51.91±2.38 | 54.00±1.42 | +2.09 | +4.0% | 25 | DEVIATION |
| reddit | bwgnn | text | 48.76±1.54 | - |  |  | 0 | UNAVAILABLE |
| reddit | care_gnn | baseline | 45.46±0.01 | 47.62±0.00 | +2.16 | +4.7% | 1 | DEVIATION |
| reddit | care_gnn | flag | 50.78±0.95 | 53.17±0.55 | +2.39 | +4.7% | 25 | DEVIATION |
| reddit | care_gnn | flag_finetuned | 51.95±2.21 | 53.22±0.73 | +1.27 | +2.4% | 25 | CLOSE |
| reddit | care_gnn | text | 47.66±1.59 | - |  |  | 0 | UNAVAILABLE |
| reddit | dga_gnn | baseline | 45.46±0.01 | 47.62±0.00 | +2.16 | +4.7% | 1 | DEVIATION |
| reddit | dga_gnn | flag | 48.77±2.18 | 53.32±0.69 | +4.55 | +9.3% | 25 | DEVIATION |
| reddit | dga_gnn | flag_finetuned | 49.50±0.83 | 53.10±0.52 | +3.60 | +7.3% | 25 | DEVIATION |
| reddit | dga_gnn | text | 45.49±0.11 | - |  |  | 0 | UNAVAILABLE |
| reddit | gat | baseline | 46.66±0.96 | 52.03±0.00 | +5.37 | +11.5% | 1 | DEVIATION |
| reddit | gat | flag | 49.70±1.76 | 54.37±0.55 | +4.67 | +9.4% | 25 | DEVIATION |
| reddit | gat | flag_finetuned | 50.20±2.33 | 53.94±0.64 | +3.74 | +7.5% | 25 | DEVIATION |
| reddit | gat | text | 48.26±2.23 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | baseline | 45.46±0.01 | 52.85±0.00 | +7.39 | +16.3% | 2 | DEVIATION |
| reddit | gcn | flag | 48.19±1.02 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | flag_finetuned | 48.72±1.59 | - |  |  | 0 | UNAVAILABLE |
| reddit | gcn | text | 45.84±0.36 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | baseline | 45.46±0.01 | 51.99±0.00 | +6.53 | +14.4% | 1 | DEVIATION |
| reddit | geniepath | flag | 48.30±2.24 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | flag_finetuned | 48.69±2.97 | - |  |  | 0 | UNAVAILABLE |
| reddit | geniepath | text | 46.84±1.89 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | baseline | 46.31±1.04 | 50.24±0.00 | +3.93 | +8.5% | 1 | DEVIATION |
| reddit | pmp | flag | 48.91±1.30 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | flag_finetuned | 50.14±1.48 | - |  |  | 0 | UNAVAILABLE |
| reddit | pmp | text | 47.21±1.18 | - |  |  | 0 | UNAVAILABLE |

## Summary

- DEVIATION: 28
- UNAVAILABLE: 26
- MATCH: 1
- CLOSE: 1

## Why exact agreement is not achievable

- The paper's 1:10 downsampling seed is unpublished, so our benchmark is a different draw of the same construction.
- The paper states no train/val/test split; ours is 10/10/80 inherited from GraphAdapter/GLBench.
- The paper states no decision-threshold policy. `argmax` reproduces the released code; `validation_swept` reproduces BWGNN's setup, which the paper says it follows. The two differ by several F1 points.
- Semantic sampling is reimplemented from Eq. 3-4; no upstream source exists.
- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/`--inits` here gives a std that is not comparable.

