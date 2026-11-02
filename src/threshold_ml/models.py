"""Candidate estimators. Hyperparameters are fixed up front; there is deliberately no search."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int) -> dict[str, ClassifierMixin]:
    """A linear baseline and a shallow boosted ensemble.

    No class weighting or resampling: the cost-optimal threshold absorbs class imbalance,
    and leaving scores unweighted keeps them interpretable as (uncalibrated) risk.
    Internal early stopping is disabled because it would carve a *random* validation
    subset out of the chronological training window.
    """
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1_000, random_state=seed),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=seed,
        ),
    }
