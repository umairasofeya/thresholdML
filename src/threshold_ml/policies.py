"""Decision policies: frozen rules mapping (score, amount) to flag / don't flag.

Each policy family has a ``fit`` function that learns its parameters from labelled
validation data. Because every family goes through the same ``fit`` → ``flag`` interface,
they can all be cross-fitted identically, which is what makes comparing them fair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .calibration import CALIBRATORS, Calibrator
from .costs import CostModel, optimize_threshold


@dataclass(frozen=True)
class GlobalThreshold:
    """Flag when ``score >= threshold``, regardless of amount."""

    threshold: float

    def flag(self, scores: ArrayLike, amounts: ArrayLike) -> NDArray[np.bool_]:
        return np.asarray(scores) >= self.threshold

    def describe(self) -> dict:
        return {"type": "global_threshold", "threshold": self.threshold}


@dataclass(frozen=True)
class AmountAwareRule:
    """Flag when the expected loss from letting a payment through exceeds the expected cost of reviewing it.

    With calibrated fraud probability ``p`` and amount ``a``, not flagging costs
    ``p · loss_rate · a`` in expectation and flagging costs ``review + (1 − p) · fp``.
    Flagging is optimal exactly when ``p`` exceeds the break-even probability
    ``(review + fp) / (loss_rate · a + fp)``, which falls as the amount rises. Given
    calibrated probabilities this rule is Bayes-optimal under the cost model, and it has no
    threshold to tune.
    """

    method: str
    calibrator: Calibrator
    costs: CostModel

    def probabilities(self, scores: ArrayLike) -> NDArray[np.float64]:
        return self.calibrator.predict(scores)

    def flag(self, scores: ArrayLike, amounts: ArrayLike) -> NDArray[np.bool_]:
        return self.costs.expected_cost_flags(self.probabilities(scores), amounts)

    def describe(self) -> dict:
        return {"type": "amount_aware", "calibration": self.method}


Policy = GlobalThreshold | AmountAwareRule
PolicyFitter = Callable[[NDArray, NDArray, NDArray], Policy]


def policy_fitters(costs: CostModel) -> dict[str, PolicyFitter]:
    """Every candidate policy family, keyed by name."""

    def global_threshold(labels: NDArray, scores: NDArray, amounts: NDArray) -> Policy:
        return GlobalThreshold(optimize_threshold(labels, scores, amounts, costs).threshold)

    def default_threshold(labels: NDArray, scores: NDArray, amounts: NDArray) -> Policy:
        return GlobalThreshold(0.5)

    def amount_aware(method: str) -> PolicyFitter:
        def fit(labels: NDArray, scores: NDArray, amounts: NDArray) -> Policy:
            return AmountAwareRule(method, CALIBRATORS[method]().fit(scores, labels), costs)

        return fit

    return {
        "global_threshold": global_threshold,
        "default_0_5": default_threshold,
        **{f"amount_aware_{method}": amount_aware(method) for method in CALIBRATORS},
    }


def cross_fitted_flags(
    fit: PolicyFitter, labels: ArrayLike, scores: ArrayLike, amounts: ArrayLike, n_blocks: int = 5
) -> NDArray[np.bool_]:
    """Out-of-block decisions over a time-ordered validation set.

    A policy fitted on validation and then scored on the same rows looks better than it
    is, and the more flexible the policy, the more optimistic its score. To compare policy
    families honestly, split validation into ``n_blocks`` contiguous time blocks, fit on all
    blocks but one, decide on the held-out block, and rotate. Every family gets the same
    treatment, so remaining bias is shared rather than favouring one.
    """
    labels, scores, amounts = np.asarray(labels), np.asarray(scores), np.asarray(amounts, dtype=float)
    blocks = np.array_split(np.arange(len(labels)), n_blocks)
    flags = np.zeros(len(labels), dtype=bool)
    for held_out in blocks:
        train = np.setdiff1d(np.arange(len(labels)), held_out, assume_unique=True)
        if len(np.unique(labels[train])) < 2:
            raise ValueError("Every cross-fitting training set needs both classes; use fewer blocks.")
        policy = fit(labels[train], scores[train], amounts[train])
        flags[held_out] = policy.flag(scores[held_out], amounts[held_out])
    return flags
