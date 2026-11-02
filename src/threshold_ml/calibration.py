"""Post-hoc probability calibration of frozen model scores.

Gradient boosting and unweighted logistic regression rank transactions well, but their
raw scores need not be probabilities. The amount-aware decision rule multiplies a
probability by a currency amount, so here calibration matters for the decision itself,
not just for presentation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPSILON = 1e-12


class Calibrator(Protocol):
    def fit(self, scores: ArrayLike, labels: ArrayLike) -> Calibrator: ...
    def predict(self, scores: ArrayLike) -> NDArray[np.float64]: ...


def _logit(scores: ArrayLike) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(scores, dtype=float), _EPSILON, 1 - _EPSILON)
    return np.log(clipped / (1 - clipped))[:, None]


class PlattCalibrator:
    """Two-parameter logistic map on the score's log-odds.

    Fitting on log-odds rather than on the raw score (as scikit-learn's ``sigmoid`` does
    for ``predict_proba`` input) keeps resolution among the many scores near zero, which
    is where almost all legitimate traffic sits. Monotonic, so ranking is unchanged.
    """

    def fit(self, scores: ArrayLike, labels: ArrayLike) -> PlattCalibrator:
        self._model = LogisticRegression(C=1e4).fit(_logit(scores), np.asarray(labels))
        return self

    def predict(self, scores: ArrayLike) -> NDArray[np.float64]:
        return self._model.predict_proba(_logit(scores))[:, 1]


class IsotonicCalibrator:
    """Non-parametric monotone step function: more flexible than Platt, noisier with few frauds."""

    def fit(self, scores: ArrayLike, labels: ArrayLike) -> IsotonicCalibrator:
        self._model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
        )
        return self

    def predict(self, scores: ArrayLike) -> NDArray[np.float64]:
        return self._model.predict(np.asarray(scores, dtype=float))


CALIBRATORS: dict[str, type[PlattCalibrator] | type[IsotonicCalibrator]] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}
