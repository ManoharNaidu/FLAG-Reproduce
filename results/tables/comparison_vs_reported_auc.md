# Reproduction vs reported — AUC

Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.
Tolerance: MATCH within ±1.0 pp, CLOSE within ±2.0 pp.

Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 was produced by the FLAG authors' own baseline rewrites, so an `official` baseline is not comparable to it (decision D-002).

The reported column is a **reference target**. No result here has been adjusted toward it, and reported values are never written into `results/`.

| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| instagram | bwgnn | baseline | 51.52±2.43 | 51.13±0.00 | -0.39 | -0.7% | 1 | MATCH |
| instagram | bwgnn | text | 54.10±0.71 | 57.70±0.00 | +3.60 | +6.6% | 1 | DEVIATION |
| instagram | care_gnn | baseline | 52.07±1.95 | 49.99±0.00 | -2.08 | -4.0% | 1 | DEVIATION |
| instagram | care_gnn | text | 54.92±0.38 | 60.92±0.00 | +6.00 | +10.9% | 1 | DEVIATION |
| instagram | dga_gnn | baseline | 50.86±0.48 | 51.14±0.00 | +0.28 | +0.6% | 1 | MATCH |
| instagram | dga_gnn | text | 56.28±1.07 | 61.05±0.00 | +4.77 | +8.5% | 1 | DEVIATION |
| instagram | gat | baseline | 51.53±1.12 | 49.68±0.00 | -1.85 | -3.6% | 1 | CLOSE |
| instagram | gat | text | 54.69±0.76 | 62.13±0.00 | +7.44 | +13.6% | 1 | DEVIATION |
| instagram | gcn | baseline | 52.61±1.80 | 53.58±0.00 | +0.97 | +1.8% | 1 | MATCH |
| instagram | gcn | text | 55.74±0.73 | 60.34±0.00 | +4.60 | +8.3% | 1 | DEVIATION |
| instagram | geniepath | baseline | 51.22±3.16 | 49.43±0.00 | -1.79 | -3.5% | 1 | CLOSE |
| instagram | geniepath | text | 52.45±2.31 | 58.89±0.00 | +6.44 | +12.3% | 1 | DEVIATION |
| instagram | pmp | baseline | 50.63±0.52 | 49.75±0.00 | -0.88 | -1.7% | 1 | MATCH |
| instagram | pmp | text | 56.05±1.27 | 59.69±0.00 | +3.64 | +6.5% | 1 | DEVIATION |
| reddit | bwgnn | baseline | 53.82±2.49 | 60.74±0.00 | +6.92 | +12.9% | 1 | DEVIATION |
| reddit | bwgnn | text | 57.56±1.53 | 66.52±0.00 | +8.96 | +15.6% | 1 | DEVIATION |
| reddit | care_gnn | baseline | 51.35±1.62 | 52.14±0.00 | +0.79 | +1.5% | 1 | MATCH |
| reddit | care_gnn | text | 56.72±1.38 | 62.35±0.00 | +5.63 | +9.9% | 1 | DEVIATION |
| reddit | dga_gnn | baseline | 50.10±0.53 | 50.36±0.00 | +0.26 | +0.5% | 1 | MATCH |
| reddit | dga_gnn | text | 59.59±1.60 | 61.28±0.00 | +1.69 | +2.8% | 1 | CLOSE |
| reddit | gat | baseline | 52.66±2.25 | 52.18±0.00 | -0.48 | -0.9% | 1 | MATCH |
| reddit | gat | text | 59.32±0.29 | 64.13±0.00 | +4.81 | +8.1% | 1 | DEVIATION |
| reddit | gcn | baseline | 50.32±0.26 | 58.51±0.00 | +8.19 | +16.3% | 1 | DEVIATION |
| reddit | gcn | text | 57.82±1.94 | 59.69±0.00 | +1.87 | +3.2% | 1 | CLOSE |
| reddit | geniepath | baseline | 52.18±1.48 | 59.35±0.00 | +7.17 | +13.7% | 1 | DEVIATION |
| reddit | geniepath | text | 56.91±1.85 | 59.67±0.00 | +2.76 | +4.9% | 1 | DEVIATION |
| reddit | pmp | baseline | 50.16±0.12 | 56.90±0.00 | +6.74 | +13.4% | 1 | DEVIATION |
| reddit | pmp | text | 59.79±0.43 | 63.20±0.00 | +3.41 | +5.7% | 1 | DEVIATION |

## Summary

- DEVIATION: 17
- MATCH: 7
- CLOSE: 4

## Why exact agreement is not achievable

- The paper's 1:10 downsampling seed is unpublished, so our benchmark is a different draw of the same construction.
- The paper states no train/val/test split; ours is 10/10/80 inherited from GraphAdapter/GLBench.
- The paper states no decision-threshold policy. `argmax` reproduces the released code; `validation_swept` reproduces BWGNN's setup, which the paper says it follows. The two differ by several F1 points.
- Semantic sampling is reimplemented from Eq. 3-4; no upstream source exists.
- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/`--inits` here gives a std that is not comparable.

