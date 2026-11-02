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
from .policies import AmountAwareRule, Policy


@dataclass
class ModelBundle:
    """A fitted estimator, its frozen decision policies, and everything needed to reproduce them."""

    name: str
    model: ClassifierMixin
    policies: dict[str, Policy]
    preferred_policy: str
    costs: CostModel
    feature_names: tuple[str, ...] = FEATURE_NAMES
    provenance: dict[str, Any] = field(default_factory=dict)
    package_version: str = __version__

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate raw rows and append the raw score, calibrated probabilities and every policy's decision.

        ``flagged`` is the decision of the preferred policy, chosen on cross-fitted validation
        cost. Labels are not required. Rows are returned in time order, matching training.
        """
        clean = validate(frame, require_label=False)
        features = build_features(clean)
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Feature layout differs from the one this model was trained on.")
        scores = self.model.predict_proba(features)[:, 1]
        amounts = clean["Amount"].to_numpy()
        columns: dict[str, Any] = {"score": scores}
        for policy in self.policies.values():
            if isinstance(policy, AmountAwareRule):
                columns[f"probability_{policy.method}"] = policy.probabilities(scores)
        for policy_name, policy in self.policies.items():
            columns[f"flagged_{policy_name}"] = policy.flag(scores, amounts)
        columns["flagged"] = columns[f"flagged_{self.preferred_policy}"]
        return clean.assign(**columns)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        """Load a bundle. joblib unpickles arbitrary code: only load files you trust."""
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} is not a {cls.__name__}.")
        return bundle
