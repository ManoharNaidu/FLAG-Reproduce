"""Centralised classification metrics for every model and variant.

Phase 12. One implementation, used by every experiment, so that a difference
between two rows is a difference in the model and never in the measurement.

Metrics
-------
threshold-free : AUC (AUROC), KS
threshold-based: F1-macro, per-class precision/recall/F1, accuracy
calibration    : ECE

The threshold policy matters more than usual here and the paper does not state
one (research/paper_notes.md section 5):

  * FLAG's released code uses `argmax` over two logits -- a fixed 0.5-equivalent
    cut. On a 1:10 imbalanced problem that is a strong, and probably poor, choice.
  * The paper says it follows **BWGNN's** experimental setup, and BWGNN sweeps
    the decision threshold over `np.linspace(0.05, 0.95, 19)` and keeps the best
    macro-F1.

Both are implemented and selectable (decision S-006, default `validation_swept`).

**The test set is never used to choose a threshold.** `fit_threshold` runs on
validation only; `evaluate` then applies that fixed value to test. This is
enforced by the API: `evaluate` takes a threshold, it never searches for one.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# BWGNN's grid, reproduced exactly. See research/repository_provenance.md section 4.
BWGNN_THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)

THRESHOLD_POLICIES = ("validation_swept", "argmax", "fixed")


class MetricError(ValueError):
    """Raised when a metric cannot be computed honestly."""


def _as_arrays(y_true, scores):
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 2:
        if scores.shape[1] != 2:
            raise MetricError(
                f"expected 2-column scores for binary classification, got "
                f"shape {scores.shape}"
            )
        scores = scores[:, 1]
    scores = scores.ravel()
    if y_true.shape != scores.shape:
        raise MetricError(
            f"y_true {y_true.shape} and scores {scores.shape} disagree"
        )
    return y_true, scores


def auc(y_true, scores) -> float:
    """AUROC. Threshold-free.

    Returns NaN when only one class is present -- AUC is undefined there, and a
    fabricated 0.5 would silently pollute an average.
    """
    y_true, scores = _as_arrays(y_true, scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def ks_statistic(y_true, scores) -> float:
    """Kolmogorov-Smirnov statistic: max |F_pos(x) - F_neg(x)|.

    The paper's industrial metric: "widely used in the financial industry ...
    measures the maximum difference between the cumulative distribution
    functions" of positive and negative scores.

    Computed on the empirical CDFs over the pooled score grid, which is exact
    rather than binned.
    """
    y_true, scores = _as_arrays(y_true, scores)
    pos = np.sort(scores[y_true == 1])
    neg = np.sort(scores[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.unique(scores))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def expected_calibration_error(y_true, scores, num_bins: int = 15) -> float:
    """ECE with equal-width confidence bins.

    FLAG's `test.py` reports ECE but `utils.ECELoss` **does not exist upstream**
    (flag_code_audit.md 5.1), so there is nothing to match. ECE is not a paper
    metric either. The binning scheme is therefore ours and is declared:
    15 equal-width bins over max-probability confidence, the common default.
    """
    y_true, scores = _as_arrays(y_true, scores)
    confidence = np.maximum(scores, 1.0 - scores)
    predicted = (scores >= 0.5).astype(int)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.5, 1.0, num_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.sum() / n * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    """Every threshold-dependent metric at one fixed threshold."""
    y_true, scores = _as_arrays(y_true, scores)
    predicted = (scores >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_fraud": float(precision[1]),
        "recall_fraud": float(recall[1]),
        "f1_fraud": float(f1[1]),
        "precision_normal": float(precision[0]),
        "recall_normal": float(recall[0]),
        "f1_normal": float(f1[0]),
        "support_fraud": int(support[1]),
        "support_normal": int(support[0]),
        "predicted_fraud_count": int(predicted.sum()),
    }


def fit_threshold(
    y_true,
    scores,
    policy: str = "validation_swept",
    grid=None,
    fixed_value: float = 0.5,
) -> tuple[float, dict]:
    """Choose a decision threshold. **Call this on VALIDATION data only.**

    Returns `(threshold, record)`. The record documents which policy ran, so a
    result row can never be ambiguous about how its F1 was obtained.
    """
    if policy not in THRESHOLD_POLICIES:
        raise MetricError(
            f"unknown threshold policy {policy!r}; expected one of "
            f"{THRESHOLD_POLICIES}"
        )
    y_true, scores = _as_arrays(y_true, scores)

    if policy == "argmax":
        # Upstream FLAG: `output.argmax(dim=-1)` on two logits == score >= 0.5.
        return 0.5, {
            "threshold_policy": "argmax",
            "threshold": 0.5,
            "source": "FLAG released code (test.py) uses argmax over 2 logits",
        }

    if policy == "fixed":
        return float(fixed_value), {
            "threshold_policy": "fixed",
            "threshold": float(fixed_value),
            "source": "explicitly configured",
        }

    grid = BWGNN_THRESHOLD_GRID if grid is None else np.asarray(grid, dtype=float)
    scored = [
        (float(f1_score(y_true, (scores >= t).astype(int), average="macro",
                        zero_division=0)), float(t))
        for t in grid
    ]
    best_f1, best_t = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return best_t, {
        "threshold_policy": "validation_swept",
        "threshold": best_t,
        "validation_f1_macro_at_threshold": best_f1,
        "grid_size": len(grid),
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "source": (
            "BWGNN sweeps np.linspace(0.05, 0.95, 19) for best macro-F1; the "
            "FLAG paper says it follows BWGNN's setup but states no policy "
            "itself (decision S-006)"
        ),
    }


@dataclass
class EvaluationResult:
    """All metrics for one split, plus how the threshold was obtained."""

    split: str
    num_examples: int
    auc: float
    ks: float
    ece: float
    threshold_metrics: dict
    threshold_record: dict
    label_distribution: dict = field(default_factory=dict)

    def as_record(self, prefix: str = "") -> dict:
        """Flatten for the results schema, e.g. prefix='test_'."""
        record = {
            f"{prefix}auc": self.auc,
            f"{prefix}ks": self.ks,
            f"{prefix}ece": self.ece,
            f"{prefix}num_examples": self.num_examples,
        }
        for key, value in self.threshold_metrics.items():
            record[f"{prefix}{key}"] = value
        return record

    def summary(self) -> str:
        tm = self.threshold_metrics
        return (
            f"{self.split:5s} n={self.num_examples:>6,}  "
            f"AUC {self.auc:.4f}  F1-macro {tm['f1_macro']:.4f}  "
            f"F1-fraud {tm['f1_fraud']:.4f}  "
            f"P {tm['precision_fraud']:.4f}  R {tm['recall_fraud']:.4f}  "
            f"KS {self.ks:.4f}  @thr {tm['threshold']:.2f}"
        )


def evaluate(
    y_true,
    scores,
    threshold: float,
    threshold_record: dict | None = None,
    split: str = "",
    ece_bins: int = 15,
) -> EvaluationResult:
    """Evaluate at a **given** threshold.

    Deliberately takes the threshold rather than searching for one, so that test
    data cannot be used to tune it. Pass the value returned by `fit_threshold`
    on validation.
    """
    y_true_arr, scores_arr = _as_arrays(y_true, scores)
    counts = np.bincount(y_true_arr, minlength=2)
    return EvaluationResult(
        split=split,
        num_examples=int(len(y_true_arr)),
        auc=auc(y_true_arr, scores_arr),
        ks=ks_statistic(y_true_arr, scores_arr),
        ece=expected_calibration_error(y_true_arr, scores_arr, ece_bins),
        threshold_metrics=threshold_metrics(y_true_arr, scores_arr, threshold),
        threshold_record=dict(threshold_record or {}),
        label_distribution={str(i): int(c) for i, c in enumerate(counts)},
    )


def evaluate_val_and_test(
    val_y, val_scores, test_y, test_scores,
    policy: str = "validation_swept",
    grid=None,
    ece_bins: int = 15,
) -> tuple[EvaluationResult, EvaluationResult, dict]:
    """The full protocol: fit the threshold on validation, apply it to test once.

    This is the only function experiment code should call, because it makes the
    correct ordering the path of least resistance.
    """
    threshold, record = fit_threshold(val_y, val_scores, policy=policy, grid=grid)
    val = evaluate(val_y, val_scores, threshold, record, "val", ece_bins)
    test = evaluate(test_y, test_scores, threshold, record, "test", ece_bins)
    return val, test, record


def aggregate(results: list[dict], keys: list[str] | None = None) -> dict:
    """mean +- std over runs, matching the paper's `mean±std` presentation.

    Uses the **sample** standard deviation (ddof=1), as `statistics.stdev` in
    FLAG's own `test.py` does. A single run reports std 0.0 with n_runs=1 rather
    than NaN, and the count is always carried so the reader can tell.
    """
    if not results:
        return {}
    if keys is None:
        keys = sorted(
            k for k in results[0]
            if isinstance(results[0][k], (int, float))
            and not isinstance(results[0][k], bool)
        )
    out = {"n_runs": len(results)}
    for key in keys:
        values = np.array(
            [r[key] for r in results if r.get(key) is not None], dtype=float
        )
        values = values[np.isfinite(values)]
        if len(values) == 0:
            out[f"{key}_mean"], out[f"{key}_std"] = float("nan"), float("nan")
            out[f"{key}_n"] = 0
            continue
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[f"{key}_n"] = int(len(values))
    return out


def format_mean_std(mean: float, std: float, as_percent: bool = True) -> str:
    """Render as the paper does, e.g. '48.19±1.02'."""
    if not np.isfinite(mean):
        return "n/a"
    scale = 100.0 if as_percent else 1.0
    return f"{mean * scale:.2f}±{std * scale:.2f}"
