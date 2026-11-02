import numpy as np
import pytest

from threshold_ml.costs import CostModel, optimize_threshold

COSTS = CostModel()


def test_cost_accounting_by_hand():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    amounts = np.array([10, 20, 100, 200])
    # Flag {0.8, 0.9}: 2 reviews * 2 + 1 false positive * 5 + missed fraud of 100.
    assert COSTS.total(labels, scores, amounts, 0.5) == 109
    assert COSTS.total(labels, scores, amounts, 2) == 300  # flag none: lose all fraud
    assert COSTS.total(labels, scores, amounts, 0) == 18  # flag all: 4 reviews + 2 false positives
    assert COSTS.row_costs(labels, scores >= 0.5, amounts).sum() == 109


@pytest.mark.parametrize("seed", range(10))
def test_threshold_search_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 200)
    scores = rng.integers(0, 10, 200) / 10  # heavy ties
    amounts = rng.uniform(0, 100, 200)
    costs = CostModel(*rng.uniform(0, 10, 3))
    search = optimize_threshold(labels, scores, amounts, costs)
    brute = np.array([costs.total(labels, scores, amounts, t) for t in search.curve["threshold"]])
    assert np.allclose(search.curve["cost"], brute)
    assert search.cost == pytest.approx(brute.min())
    assert search.alerts == (scores >= search.threshold).sum()
    every_cutoff = [costs.total(labels, scores, amounts, t) for t in np.r_[np.unique(scores), 2.0]]
    assert search.cost == pytest.approx(min(every_cutoff))


def test_ties_prefer_fewer_alerts():
    # Flagging the lone legitimate row costs exactly what it saves elsewhere: zero-sum.
    labels, scores, amounts = np.array([1, 0]), np.array([0.9, 0.1]), np.array([7.0, 0.0])
    search = optimize_threshold(labels, scores, amounts, CostModel(review_cost=7, false_positive_cost=0))
    assert search.alerts == 0


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_cost_model_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        CostModel(review_cost=bad)
