"""Unit tests for the shared metrics module.

Cases use hand-constructed score/label pairs whose correct metric value is
computable by hand, so the tests do not merely re-run sklearn and agree with it.

The most important property tested here is procedural rather than numerical:
**the test set must never influence the decision threshold.**
"""
from __future__ import annotations

import numpy as np

from flagbench.metrics.classification import (
    BWGNN_THRESHOLD_GRID,
    degenerate_f1_macro,
    implied_majority_ratio,
    MetricError,
    aggregate,
    auc,
    evaluate,
    evaluate_val_and_test,
    expected_calibration_error,
    fit_threshold,
    format_mean_std,
    ks_statistic,
    threshold_metrics,
)


# ------------------------------------------------------------------- AUC
def test_auc_perfect_separation_is_one():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc(y, scores) - 1.0) < 1e-12


def test_auc_inverted_ranking_is_zero():
    assert abs(auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-12


def test_auc_hand_computed_case():
    """2 positives, 2 negatives; 3 of 4 pos/neg pairs correctly ordered."""
    y = [0, 1, 0, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    # pairs (pos,neg): (0.4,0.1)ok (0.4,0.35)ok (0.8,0.1)ok (0.8,0.35)ok -> 4/4
    assert abs(auc(y, scores) - 1.0) < 1e-12
    y2 = [0, 1, 0, 1]
    scores2 = [0.5, 0.4, 0.35, 0.8]
    # pairs: (0.4,0.5)no (0.4,0.35)ok (0.8,0.5)ok (0.8,0.35)ok -> 3/4
    assert abs(auc(y2, scores2) - 0.75) < 1e-12


def test_auc_single_class_returns_nan_not_a_made_up_value():
    """A fabricated 0.5 would silently pollute an average over runs."""
    assert np.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert np.isnan(auc([0, 0, 0], [0.2, 0.5, 0.9]))


def test_auc_accepts_two_column_scores():
    """Models emit 2 logits/probabilities; column 1 is the fraud score."""
    two_col = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert abs(auc([0, 0, 1, 1], two_col) - 1.0) < 1e-12


def test_mismatched_shapes_are_rejected():
    try:
        auc([0, 1, 1], [0.1, 0.9])
    except MetricError:
        return
    raise AssertionError("shape mismatch should raise")


# -------------------------------------------------------------------- KS
def test_ks_perfect_separation_is_one():
    assert abs(ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12


def test_ks_identical_distributions_is_zero():
    y = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert abs(ks_statistic(y, scores)) < 1e-12


def test_ks_hand_computed_partial_overlap():
    """neg = {0.1, 0.3}, pos = {0.2, 0.4}.

    At x=0.1: F_neg=0.5, F_pos=0.0 -> 0.5.  That is the maximum.
    """
    y = [0, 1, 0, 1]
    scores = [0.1, 0.2, 0.3, 0.4]
    assert abs(ks_statistic(y, scores) - 0.5) < 1e-12


def test_ks_single_class_is_nan():
    assert np.isnan(ks_statistic([1, 1], [0.3, 0.7]))


# ------------------------------------------------------- threshold metrics
def test_threshold_metrics_hand_counted_confusion():
    """scores >= 0.5 predicts positive.

    y     = [0, 0, 1, 1]
    score = [0.2, 0.6, 0.4, 0.9]
    pred  = [0,   1,   0,   1]
    TP=1 (idx3), FP=1 (idx1), FN=1 (idx2), TN=1 (idx0)
    precision_fraud = 1/2, recall_fraud = 1/2, f1_fraud = 1/2, accuracy = 1/2
    """
    m = threshold_metrics([0, 0, 1, 1], [0.2, 0.6, 0.4, 0.9], 0.5)
    assert abs(m["precision_fraud"] - 0.5) < 1e-12
    assert abs(m["recall_fraud"] - 0.5) < 1e-12
    assert abs(m["f1_fraud"] - 0.5) < 1e-12
    assert abs(m["accuracy"] - 0.5) < 1e-12
    assert m["support_fraud"] == 2
    assert m["predicted_fraud_count"] == 2


def test_threshold_changes_predictions():
    y = [0, 0, 1, 1]
    scores = [0.2, 0.6, 0.4, 0.9]
    strict = threshold_metrics(y, scores, 0.8)
    assert strict["predicted_fraud_count"] == 1
    loose = threshold_metrics(y, scores, 0.1)
    assert loose["predicted_fraud_count"] == 4
    assert abs(loose["recall_fraud"] - 1.0) < 1e-12


def test_f1_macro_averages_both_classes():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    m = threshold_metrics(y, scores, 0.5)
    assert abs(m["f1_macro"] - 1.0) < 1e-12
    assert abs(m["f1_fraud"] - 1.0) < 1e-12
    assert abs(m["f1_normal"] - 1.0) < 1e-12


def test_all_negative_prediction_gives_zero_fraud_f1_not_an_error():
    """The degenerate case the paper's ~45.46 baseline F1-macro implies."""
    y = [0] * 10 + [1]
    scores = [0.1] * 11
    m = threshold_metrics(y, scores, 0.5)
    assert m["predicted_fraud_count"] == 0
    assert m["f1_fraud"] == 0.0
    assert m["f1_macro"] > 0.0     # the majority class still scores


# ------------------------------------------------------- threshold policy
def test_argmax_policy_is_exactly_half():
    """Upstream FLAG uses argmax over 2 logits, which is score >= 0.5."""
    t, record = fit_threshold([0, 1], [0.4, 0.6], policy="argmax")
    assert t == 0.5
    assert record["threshold_policy"] == "argmax"


def test_swept_policy_uses_the_bwgnn_grid():
    t, record = fit_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert record["threshold_policy"] == "validation_swept"
    assert record["grid_size"] == 19
    assert abs(record["grid_min"] - 0.05) < 1e-12
    assert abs(record["grid_max"] - 0.95) < 1e-12
    assert t in BWGNN_THRESHOLD_GRID


def test_swept_policy_beats_argmax_on_an_imbalanced_case():
    """The whole reason the policy matters: at 1:10, 0.5 can be badly wrong."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 10)
    # Positives score higher, but everything sits below 0.5.
    scores = np.concatenate([
        rng.uniform(0.00, 0.20, 100),
        rng.uniform(0.25, 0.45, 10),
    ])
    swept_t, _ = fit_threshold(y, scores, policy="validation_swept")
    argmax_t, _ = fit_threshold(y, scores, policy="argmax")

    swept_f1 = threshold_metrics(y, scores, swept_t)["f1_macro"]
    argmax_f1 = threshold_metrics(y, scores, argmax_t)["f1_macro"]
    assert argmax_f1 < swept_f1, (
        "a 0.5 cut should badly underperform a swept threshold here; "
        "this is why the policy is configurable"
    )


def test_unknown_policy_is_rejected():
    try:
        fit_threshold([0, 1], [0.2, 0.8], policy="whatever")
    except MetricError as exc:
        assert "unknown threshold policy" in str(exc)
        return
    raise AssertionError("unknown policy should raise")


def test_fixed_policy_returns_what_it_was_given():
    t, record = fit_threshold([0, 1], [0.2, 0.8], policy="fixed", fixed_value=0.33)
    assert abs(t - 0.33) < 1e-12
    assert record["threshold_policy"] == "fixed"


# ------------------------------------------------ the no-test-tuning rule
def test_evaluate_requires_a_threshold_and_does_not_search():
    """`evaluate` has no policy argument, so it cannot tune on the data given."""
    import inspect

    params = inspect.signature(evaluate).parameters
    assert "threshold" in params
    assert "policy" not in params, (
        "evaluate must not be able to fit a threshold -- that would allow "
        "tuning on the test set"
    )


def test_val_and_test_protocol_applies_the_validation_threshold_to_test():
    rng = np.random.default_rng(1)
    val_y = np.array([0] * 50 + [1] * 5)
    val_scores = np.concatenate([rng.uniform(0, 0.3, 50), rng.uniform(0.2, 0.5, 5)])
    test_y = np.array([0] * 50 + [1] * 5)
    test_scores = np.concatenate([rng.uniform(0, 0.3, 50), rng.uniform(0.2, 0.5, 5)])

    val, test, record = evaluate_val_and_test(
        val_y, val_scores, test_y, test_scores
    )
    assert val.threshold_metrics["threshold"] == test.threshold_metrics["threshold"]
    assert test.threshold_metrics["threshold"] == record["threshold"]
    assert val.split == "val" and test.split == "test"


def test_test_threshold_is_not_the_test_optimum_in_general():
    """Proves the threshold really came from validation, not from test."""
    val_y = np.array([0] * 20 + [1] * 20)
    val_scores = np.concatenate([np.full(20, 0.10), np.full(20, 0.30)])
    # Test scores are shifted upward, so its own optimum differs.
    test_y = np.array([0] * 20 + [1] * 20)
    test_scores = np.concatenate([np.full(20, 0.60), np.full(20, 0.90)])

    _, test, record = evaluate_val_and_test(val_y, val_scores, test_y, test_scores)
    applied = test.threshold_metrics["threshold"]
    assert applied == record["threshold"]
    # Validation's optimum lies between 0.10 and 0.30; applying it to test
    # labels everything positive, which is exactly the expected consequence.
    assert applied < 0.6
    assert test.threshold_metrics["predicted_fraud_count"] == 40


# --------------------------------------------------------------- evaluate
def test_evaluate_returns_all_metric_families():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    result = evaluate(y, scores, 0.5, split="test")
    assert abs(result.auc - 1.0) < 1e-12
    assert abs(result.ks - 1.0) < 1e-12
    assert 0.0 <= result.ece <= 1.0
    assert abs(result.threshold_metrics["f1_macro"] - 1.0) < 1e-12
    assert result.label_distribution == {"0": 2, "1": 2}
    assert "test" in result.summary()


def test_as_record_prefixes_every_field():
    result = evaluate([0, 1], [0.2, 0.8], 0.5, split="test")
    record = result.as_record(prefix="test_")
    assert "test_auc" in record and "test_f1_macro" in record
    assert "test_threshold" in record
    assert all(k.startswith("test_") for k in record)


# ---------------------------------------------------------------- ECE
def test_ece_of_a_perfectly_calibrated_confident_model_is_near_zero():
    scores = np.array([0.99] * 100 + [0.01] * 100)
    y = np.array([1] * 100 + [0] * 100)
    assert expected_calibration_error(y, scores) < 0.02


def test_ece_of_a_confidently_wrong_model_is_large():
    scores = np.array([0.99] * 100)
    y = np.zeros(100, dtype=int)
    assert expected_calibration_error(y, scores) > 0.9


# --------------------------------------------------------------- aggregate
def test_aggregate_matches_hand_computed_mean_and_sample_std():
    runs = [{"auc": 0.60}, {"auc": 0.70}, {"auc": 0.80}]
    agg = aggregate(runs, keys=["auc"])
    assert abs(agg["auc_mean"] - 0.70) < 1e-12
    # sample std (ddof=1) of {0.6,0.7,0.8} is 0.1
    assert abs(agg["auc_std"] - 0.1) < 1e-12
    assert agg["n_runs"] == 3
    assert agg["auc_n"] == 3


def test_aggregate_single_run_reports_zero_std_and_the_count():
    agg = aggregate([{"auc": 0.5}], keys=["auc"])
    assert agg["auc_std"] == 0.0
    assert agg["n_runs"] == 1


def test_aggregate_skips_nan_and_reports_how_many_counted():
    runs = [{"auc": 0.6}, {"auc": float("nan")}, {"auc": 0.8}]
    agg = aggregate(runs, keys=["auc"])
    assert abs(agg["auc_mean"] - 0.7) < 1e-12
    assert agg["auc_n"] == 2, "must report that only 2 runs contributed"
    assert agg["n_runs"] == 3


def test_format_mean_std_matches_the_papers_presentation():
    assert format_mean_std(0.4819, 0.0102) == "48.19±1.02"
    assert format_mean_std(0.6018, 0.0079) == "60.18±0.79"
    assert format_mean_std(float("nan"), 0.0) == "n/a"


# ----------------------------------------------- degenerate-classifier maths
def test_degenerate_f1_macro_matches_a_measured_all_majority_prediction():
    """The closed form must agree with actually scoring the trivial classifier."""
    y = np.array([0] * 1000 + [1] * 100)          # exactly 10:1
    measured = threshold_metrics(y, np.zeros(len(y)), 0.5)["f1_macro"]
    closed_form = degenerate_f1_macro(1000 / 1100)
    assert abs(measured - closed_form) < 1e-12
    assert abs(measured - 0.47619) < 1e-4


def test_degenerate_flag_is_set_for_single_valued_predictions():
    y = np.array([0] * 10 + [1])
    assert threshold_metrics(y, np.zeros(11), 0.5)["is_degenerate"] is True
    assert threshold_metrics(y, np.ones(11), 0.5)["is_degenerate"] is True
    mixed = np.array([0.1] * 5 + [0.9] * 6)
    assert threshold_metrics(y, mixed, 0.5)["is_degenerate"] is False


def test_implied_ratio_inverts_the_closed_form():
    for ratio in (5.0, 8.73, 10.0, 20.0):
        p0 = ratio / (1 + ratio)
        f1 = degenerate_f1_macro(p0)
        assert abs(implied_majority_ratio(f1) - ratio) < 1e-6


def test_implied_ratio_of_the_papers_reddit_baseline():
    """45.46 appears in 4 Reddit baseline cells with std 0.01.

    Documented in research/degenerate_baselines.md. If those cells are the
    trivial classifier, the implied test ratio is ~5:1, not the stated ~10:1.
    """
    assert abs(implied_majority_ratio(0.4546) - 5.01) < 0.02
    assert abs(implied_majority_ratio(0.4729) - 8.73) < 0.02
    assert abs(implied_majority_ratio(0.4762) - 10.00) < 0.02


def _main() -> int:
    tests = [
        (n, o) for n, o in sorted(globals().items())
        if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
