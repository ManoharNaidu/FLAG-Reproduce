# Many of the paper's baseline cells are the trivial classifier

**Status:** analysis of published numbers, done 2026-09-09. This is an inference
with a clearly stated assumption, not a verified fact about the authors' setup.

## The observation

Predicting one class everywhere gives F1-macro

```
F1_majority = 2 * p0 * 1 / (p0 + 1)      precision = p0, recall = 1
F1_minority = 0                          nothing predicted, nothing recalled
F1-macro    = p0 / (1 + p0)
```

where `p0` is the majority fraction of the **test set**. It depends only on the
class ratio — not on the model, the features, or the seed.

In FLAG's Table 4, several F1-macro cells report the **same value to two decimal
places across different models**, with a standard deviation of **0.01**:

| dataset | value | cells reporting it |
|---|---:|---|
| Reddit | **45.46** | GCN/baseline, GeniePath/baseline, CARE-GNN/baseline, DGA-GNN/baseline |
| Instagram | **47.29** | GCN/+text, GeniePath/+text, CARE-GNN/baseline, DGA-GNN/baseline, DGA-GNN/+text |

(BWGNN/baseline on Reddit reports 45.47 and on Instagram 47.28 — the same values
to rounding.)

Four architecturally different models — a convolution, an LSTM-based adaptive
path model, a similarity-gated aggregator and a GraphSAGE variant — agreeing to
0.01 across 25 runs each is not a coincidence between models. It is one number
that does not depend on the model.

**Corroboration from AUC.** The same Reddit baseline rows report AUC of 50.32,
52.18, 51.35 and 50.10 — at or near chance. A model that has learned nothing
scores chance AUC *and* the trivial F1-macro. The two metrics agree.

## What follows, if these cells are degenerate

Inverting `F1-macro = p0 / (1 + p0)`:

| F1-macro | implied majority fraction | implied ratio |
|---:|---:|---:|
| 45.46 (paper, Reddit) | 83.35% | **5.01 : 1** |
| 47.29 (paper, Instagram) | 89.72% | **8.73 : 1** |
| 47.62 (ours, both) | 90.91% | **10.00 : 1** |

Our benchmark is 10.00:1 in every split by construction (stratified), and our own
degenerate value is 47.62 on both datasets — which is exactly what we measure for
GAT/baseline, GAT/+text, CARE-GNN/baseline and CARE-GNN/+text on Reddit.

The paper says the construction targets "about 1:10" for **both** datasets. If
these cells are degenerate, the implied test-set ratios are **5.0:1 for Reddit and
8.7:1 for Instagram** — not 10:1, and not equal to each other.

## Why this matters for the reproduction

1. **Those cells cannot be reproduced by improving a model.** They measure a
   class ratio. Our 47.62 vs the paper's 45.46 is a **+2.16 "deviation"** in the
   comparison table, but nothing about the model differs — only the test split's
   composition. Reporting that as a modelling discrepancy would be wrong.

2. **It changes what the FLAG improvement is measured against.** If a baseline is
   the trivial classifier, then "+FLAG beats baseline by 3.14 F1" partly measures
   *the baseline having learned nothing*, not the method's contribution over a
   working baseline. The `+text` rows, which do learn (AUC 55-60), are the more
   informative comparison.

3. **It is a candidate explanation for our stronger baselines.** Our Reddit GCN
   baseline reaches AUC 58.51 where the paper's reaches 50.32. Under this reading
   the difference is not that our GCN is better, but that theirs did not train —
   which is consistent with the unresolved question about what their "shallow
   embeddings" are (`dataset_notes.md` section 7: the stored features are 4096-d
   and Llama-2-derived, not shallow).

## What we have NOT established

- **That the authors' test ratio really is 5:1 / 8.7:1.** The inference assumes
  those cells are degenerate. They could instead be non-degenerate predictions
  that happen to coincide across four models to 0.01, which we consider very
  unlikely but have not ruled out.
- **Why Reddit and Instagram would differ** (5.0:1 vs 8.7:1) when both are
  described as "about 1:10". Possibilities include an unstratified split drifting
  from the global ratio, a different downsampling target per dataset, or the
  ratio applying to the full graph rather than to the test split. The paper
  states no split, so none of these can be checked.
- **Anything about the authors' intent.** A baseline collapsing to the majority
  class under `argmax` on a 1:10 problem is an entirely ordinary outcome, and the
  paper reports it transparently rather than hiding it.

## How this is handled in the codebase

- `flagbench.metrics.classification.threshold_metrics` returns
  **`is_degenerate`** for every evaluation (true when the prediction is
  single-valued), so a collapsed run is visible in its result row rather than
  looking like a modest score.
- `degenerate_f1_macro(p0)` and `implied_majority_ratio(f1)` make the arithmetic
  reusable and testable.
- The comparison against reported values (`analysis/compare_reported.py`) keeps
  reporting the raw delta. It does **not** silently exclude these cells — hiding
  an inconvenient comparison would be worse than showing one that needs a
  footnote. This document is that footnote.

## Reproducing the analysis

```bash
python - <<'PY'
from flagbench.metrics.classification import implied_majority_ratio
for v in (0.4546, 0.4729, 0.4762):
    print(f"F1-macro {v*100:.2f} -> {implied_majority_ratio(v):.2f}:1")
PY
```
