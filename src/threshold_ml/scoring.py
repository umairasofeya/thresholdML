"""Self-describing model bundles for scoring new transactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin

from . import __version__
from .costs import CostModel
from .data import validate
from .features import FEATURE_NAMES, build_features


@dataclass
class ModelBundle:
    """A fitted estimator together with everything needed to reproduce its decisions."""

    name: str
    model: ClassifierMixin
    threshold: float
    costs: CostModel
    feature_names: tuple[str, ...] = FEATURE_NAMES
    provenance: dict[str, Any] = field(default_factory=dict)
    package_version: str = __version__

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate raw rows, then return them with ``score`` and ``flagged`` columns appended.

        Labels are not required. Rows are returned in time order, matching training.
        """
        clean = validate(frame, require_label=False)
        features = build_features(clean)
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Feature layout differs from the one this model was trained on.")
        scores = self.model.predict_proba(features)[:, 1]
        return clean.assign(score=scores, flagged=scores >= self.threshold)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        """Load a bundle. joblib unpickles arbitrary code: only load files you trust."""
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} is not a {cls.__name__}.")
        return bundle
