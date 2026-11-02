"""Hypothetical decision-cost model and exact cost-minimising threshold search."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CostModel:
    """Per-transaction costs, all in one arbitrary currency unit.

    * Every flagged transaction costs ``review_cost`` to investigate.
    * A flagged legitimate transaction additionally costs ``false_positive_cost``
      (customer friction, declined purchase).
    * A missed fraud loses ``loss_rate`` times its amount.
    * A flagged fraud is assumed to be fully prevented.
    """

    review_cost: float = 2.0
    false_positive_cost: float = 5.0
    loss_rate: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")

    def row_costs(self, labels: ArrayLike, flagged: ArrayLike, amounts: ArrayLike) -> NDArray[np.float64]:
        """Cost incurred by each transaction; sums to the total. Lets resampling reuse one pass."""
        labels, flagged, amounts = np.asarray(labels), np.asarray(flagged, dtype=bool), np.asarray(amounts, dtype=float)
        flagged_cost = self.review_cost + self.false_positive_cost * (labels == 0)
        missed_cost = self.loss_rate * amounts * (labels == 1)
        return np.where(flagged, flagged_cost, missed_cost)

    def total(self, labels: ArrayLike, scores: ArrayLike, amounts: ArrayLike, threshold: float) -> float:
        """Total cost when every transaction with ``score >= threshold`` is flagged."""
        return float(self.row_costs(labels, np.asarray(scores) >= threshold, amounts).sum())

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSearch:
    threshold: float
    cost: float
    alerts: int
    curve: pd.DataFrame


def optimize_threshold(labels: ArrayLike, scores: ArrayLike, amounts: ArrayLike, costs: CostModel) -> ThresholdSearch:
    """Find the threshold that minimises total cost, exactly, in O(n log n).

    Sort scores descending; lowering the threshold past each distinct score flags that
    whole tie group at once, changing the cost by the group's summed deltas. Evaluating
    the cumulative sum at the end of each tie group gives the cost of every achievable
    decision rule, plus the flag-none rule. ``argmin`` returns the first minimum, which is
    the highest threshold, so ties resolve towards fewer alerts.
    """
    labels, scores, amounts = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(amounts, dtype=float)
    if not len(scores):
        raise ValueError("Cannot optimise a threshold over zero scores.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels, sorted_amounts = scores[order], labels[order], amounts[order]
    deltas = (
        costs.review_cost
        + costs.false_positive_cost * (sorted_labels == 0)
        - costs.loss_rate * sorted_amounts * (sorted_labels == 1)
    )
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    flag_none_cost = costs.loss_rate * amounts[labels == 1].sum()
    curve = pd.DataFrame(
        {
            "threshold": np.r_[np.nextafter(scores.max(), np.inf), sorted_scores[group_ends]],
            "alerts": np.r_[0, group_ends + 1],
            "cost": np.r_[flag_none_cost, flag_none_cost + np.cumsum(deltas)[group_ends]],
        }
    )
    best = int(np.argmin(curve["cost"].to_numpy()))
    row = curve.iloc[best]
    return ThresholdSearch(threshold=float(row.threshold), cost=float(row.cost), alerts=int(row.alerts), curve=curve)
