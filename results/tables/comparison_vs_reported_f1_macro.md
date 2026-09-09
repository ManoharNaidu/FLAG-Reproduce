# Reproduction vs reported — F1-macro

Paper: FLAG, KDD 2025, DOI 10.1145/3711896.3737220, Table 4.
Tolerance: MATCH within ±1.0 pp, CLOSE within ±2.0 pp.

Only `impl_source=flag_bundled` runs are compared: the paper's Table 4 was produced by the FLAG authors' own baseline rewrites, so an `official` baseline is not comparable to it (decision D-002).

The reported column is a **reference target**. No result here has been adjusted toward it, and reported values are never written into `results/`.

| Dataset | Model | Variant | Reported | Ours | Abs | Rel | Runs | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| instagram | gcn | baseline | 47.88±0.96 | 51.91±0.00 | +4.03 | +8.4% | 1 | DEVIATION |
| instagram | gcn | text | 47.29±0.01 | 47.62±0.00 | +0.33 | +0.7% | 1 | MATCH |
| reddit | gcn | baseline | 45.46±0.01 | 49.58±0.00 | +4.12 | +9.1% | 1 | DEVIATION |
| reddit | gcn | text | 45.84±0.36 | 48.32±0.00 | +2.48 | +5.4% | 1 | DEVIATION |

## Summary

- DEVIATION: 3
- MATCH: 1

## Why exact agreement is not achievable

- The paper's 1:10 downsampling seed is unpublished, so our benchmark is a different draw of the same construction.
- The paper states no train/val/test split; ours is 10/10/80 inherited from GraphAdapter/GLBench.
- The paper states no decision-threshold policy. `argmax` reproduces the released code; `validation_swept` reproduces BWGNN's setup, which the paper says it follows. The two differ by several F1 points.
- Semantic sampling is reimplemented from Eq. 3-4; no upstream source exists.
- The paper reports 25 runs (5 seeds x 5 inits); a smaller `--seeds`/`--inits` here gives a std that is not comparable.

