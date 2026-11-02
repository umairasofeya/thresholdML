import numpy as np
import pytest

from threshold_ml.calibration import IsotonicCalibrator, PlattCalibrator
from threshold_ml.costs import CostModel
from threshold_ml.policies import AmountAwareRule, GlobalThreshold, cross_fitted_flags, policy_fitters


@pytest.mark.parametrize("seed", range(5))
def test_expected_cost_rule_minimises_each_rows_expected_cost(seed):
    rng = np.random.default_rng(seed)
    costs = CostModel(*rng.uniform(0, 10, 3))
    p, amounts = rng.random(1_000), rng.lognormal(3, 2, 1_000)
    flag_cost = costs.review_cost + (1 - p) * costs.false_positive_cost
    pass_cost = p * costs.loss_rate * amounts
    flags = costs.expected_cost_flags(p, amounts)
    assert np.all(np.where(flags, flag_cost, pass_cost) <= np.minimum(flag_cost, pass_cost) + 1e-9)


def test_break_even_probability_is_the_decision_boundary():
    costs = CostModel()
    amounts = np.array([1.0, 10.0, 100.0, 1_000.0])
    boundary = costs.break_even_probability(amounts)
    assert np.all(np.diff(boundary) < 0)  # larger payments need less evidence
    reachable = boundary < 1  # a 1-unit payment is never worth flagging at these costs
    assert not costs.expected_cost_flags(boundary[reachable] * 0.999, amounts[reachable]).any()
    assert costs.expected_cost_flags(boundary[reachable] * 1.001, amounts[reachable]).all()
    assert list(reachable) == [False, True, True, True]
    assert np.isinf(CostModel(false_positive_cost=0, loss_rate=0).break_even_probability([5.0])).all()


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_recover_true_probabilities_and_keep_order(calibrator):
    rng = np.random.default_rng(0)
    truth = rng.beta(0.3, 20, 200_000)
    labels = (rng.random(len(truth)) < truth).astype(int)
    distorted = truth**0.5  # monotone but badly calibrated
    fitted = calibrator().fit(distorted, labels).predict(distorted)
    assert np.abs(fitted.mean() - labels.mean()) < 1e-3
    assert np.mean(np.abs(fitted - truth)) < np.mean(np.abs(distorted - truth)) / 5
    order = np.argsort(distorted)
    assert np.all(np.diff(fitted[order]) >= -1e-12)


def test_cross_fitting_never_sees_held_out_labels():
    rng = np.random.default_rng(1)
    labels = (rng.random(5_000) < 0.05).astype(int)
    scores, amounts = np.clip(0.4 * labels + rng.random(5_000) * 0.6, 0, 1), rng.lognormal(3, 1, 5_000)
    fit = policy_fitters(CostModel())["amount_aware_platt"]
    baseline = cross_fitted_flags(fit, labels, scores, amounts, n_blocks=5)
    held_out = np.array_split(np.arange(5_000), 5)[2]
    tampered = labels.copy()
    tampered[held_out] = 1 - tampered[held_out]
    flags = cross_fitted_flags(fit, tampered, scores, amounts, n_blocks=5)
    assert np.array_equal(flags[held_out], baseline[held_out])
    assert not np.array_equal(flags, baseline)  # other blocks did learn from the change


def test_policy_families():
    fitters = policy_fitters(CostModel())
    assert set(fitters) == {"global_threshold", "default_0_5", "amount_aware_platt", "amount_aware_isotonic"}
    labels, scores, amounts = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), np.ones(4) * 100
    assert fitters["default_0_5"](labels, scores, amounts) == GlobalThreshold(0.5)
    assert isinstance(fitters["amount_aware_isotonic"](labels, scores, amounts), AmountAwareRule)
