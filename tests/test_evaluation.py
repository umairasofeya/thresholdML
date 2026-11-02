import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from threshold_ml.costs import CostModel
from threshold_ml.evaluation import (
    bootstrap_test_metrics,
    calibration_summary,
    decision_metrics,
    stratified_resamples,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    labels = (rng.random(2_000) < 0.05).astype(int)
    scores = np.clip(0.3 * labels + rng.random(2_000) * 0.7, 0, 1)
    amounts = rng.lognormal(3, 1, 2_000)
    return labels, scores, amounts


def test_decision_metrics_agree_with_sklearn(scored):
    labels, scores, amounts = scored
    metrics = decision_metrics(labels, scores >= 0.6, amounts, CostModel())
    assert metrics["precision"] == pytest.approx(precision_score(labels, scores >= 0.6))
    assert metrics["recall"] == pytest.approx(recall_score(labels, scores >= 0.6))
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fp"] == metrics["flagged"]
    assert sum(cm.values()) == len(labels)


def test_stratified_resamples_preserve_class_counts(scored):
    labels = scored[0]
    for index in stratified_resamples(labels, 20, seed=1):
        assert len(index) == len(labels)
        assert labels[index].sum() == labels.sum()


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    scores = rng.random(200_000)
    labels = (rng.random(200_000) < scores).astype(int)
    summary = calibration_summary(labels, scores)
    assert summary["expected_calibration_error"] < 0.01
    assert sum(b["count"] for b in summary["bins"]) == len(scores)


def test_bootstrap_intervals_bracket_estimates_and_pair_decisions(scored):
    labels, scores, amounts = scored
    result = bootstrap_test_metrics(
        labels,
        amounts,
        {"model": scores},
        {"a": scores >= 0.6, "b": scores >= 0.6, "c": scores >= 0.9},
        CostModel(),
        [("a", "b"), ("a", "c")],
        n_resamples=200,
        seed=3,
    )
    ap = result["average_precision"]["model"]
    assert ap["low"] <= ap["estimate"] <= ap["high"]
    for metric in result["decisions"]["a"].values():
        assert metric["low"] <= metric["estimate"] <= metric["high"]
    identical, different = result["paired_cost_difference"]
    # Identical rules on identical (paired) resamples: zero difference in every resample.
    assert identical["low"] == identical["high"] == identical["share_candidate_cheaper"] == 0
    assert different["low"] < different["high"]


def test_bootstrap_is_deterministic(scored):
    labels, scores, amounts = scored

    def run():
        return bootstrap_test_metrics(labels, amounts, {"m": scores}, {"d": scores >= 0.5}, CostModel(), n_resamples=50)

    assert run() == run()
