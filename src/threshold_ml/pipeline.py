"""End-to-end experiment: load → split → fit → choose decision policies on validation → freeze → evaluate."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from . import __version__
from .costs import CostModel, optimize_threshold
from .data import LABEL_COLUMN, Source, load_transactions, validate
from .evaluation import bootstrap_test_metrics, calibration_summary, decision_metrics
from .features import build_features
from .models import build_models
from .plots import evaluation_figure
from .policies import AmountAwareRule, cross_fitted_flags, policy_fitters
from .report import render_report
from .scoring import ModelBundle
from .splits import chronological_split

log = logging.getLogger(__name__)

BASELINE_POLICY = "global_threshold"


@dataclass(frozen=True)
class ExperimentConfig:
    source: Source = "synthetic"
    csv_path: str | None = None
    rows: int = 20_000
    seed: int = 42
    output: Path = Path("results")
    costs: CostModel = field(default_factory=CostModel)
    n_bootstrap: int = 1_000
    cross_fit_blocks: int = 5


def _split_summary(part: pd.DataFrame) -> dict:
    return {
        "rows": len(part),
        "frauds": int(part[LABEL_COLUMN].sum()),
        "time_min": float(part["Time"].min()),
        "time_max": float(part["Time"].max()),
    }


def _selection_key(validation: dict) -> tuple[float, int]:
    """Lowest cross-fitted validation cost; ties go to fewer alerts."""
    return validation["cost"], validation["flagged"]


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the full protocol and write results.json, REPORT.md, evaluation.png and per-model artefacts.

    The test partition is read only after every model, calibrator and policy is frozen.
    """
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    costs = config.costs

    raw, provenance = load_transactions(config.source, config.csv_path, config.rows, config.seed)
    frame = validate(raw)
    split = chronological_split(frame)
    log.info("Split sizes: %s", {name: len(part) for name, part in split._asdict().items()})

    # Phase 1: fit on train; fit and compare decision policies on validation. No test data is read here.
    validation = split.validation
    val_labels, val_amounts = validation[LABEL_COLUMN].to_numpy(), validation["Amount"].to_numpy()
    validation_features = build_features(validation)
    fitters = policy_fitters(costs)
    bundles: dict[str, ModelBundle] = {}
    curves: dict[str, pd.DataFrame] = {}
    validation_ap: dict[str, float] = {}
    validation_results: dict[str, dict[str, dict]] = {}
    for name, model in build_models(config.seed).items():
        log.info("Fitting %s", name)
        model.fit(build_features(split.train), split.train[LABEL_COLUMN])
        scores = model.predict_proba(validation_features)[:, 1]
        validation_ap[name] = float(average_precision_score(val_labels, scores))
        curves[name] = optimize_threshold(val_labels, scores, val_amounts, costs).curve
        policies, validation_results[name] = {}, {}
        for policy_name, fit in fitters.items():
            policies[policy_name] = fit(val_labels, scores, val_amounts)
            in_sample = policies[policy_name].flag(scores, val_amounts)
            held_out = cross_fitted_flags(fit, val_labels, scores, val_amounts, config.cross_fit_blocks)
            validation_results[name][policy_name] = {
                "in_sample": decision_metrics(val_labels, in_sample, val_amounts, costs),
                "cross_fitted": decision_metrics(val_labels, held_out, val_amounts, costs),
            }
        preferred = min(policies, key=lambda p: _selection_key(validation_results[name][p]["cross_fitted"]))
        bundles[name] = ModelBundle(name, model, policies, preferred, costs, provenance=provenance)
    selected_model = min(
        bundles,
        key=lambda m: _selection_key(validation_results[m][bundles[m].preferred_policy]["cross_fitted"]),
    )
    selected_policy = bundles[selected_model].preferred_policy

    # Phase 2: evaluate every frozen policy on test.
    test, test_labels, test_amounts = split.test, split.test[LABEL_COLUMN].to_numpy(), split.test["Amount"].to_numpy()
    test_features = build_features(test)
    test_scores = {name: bundle.model.predict_proba(test_features)[:, 1] for name, bundle in bundles.items()}
    decisions = {
        f"{name}/{policy_name}": policy.flag(test_scores[name], test_amounts)
        for name, bundle in bundles.items()
        for policy_name, policy in bundle.policies.items()
    }
    probabilities = {
        name: {
            policy.method: policy.probabilities(test_scores[name])
            for policy in bundle.policies.values()
            if isinstance(policy, AmountAwareRule)
        }
        for name, bundle in bundles.items()
    }
    results: dict = {
        "data": provenance,
        "rows": len(frame),
        "seed": config.seed,
        "cost_assumptions": costs.as_dict(),
        "environment": {
            "threshold_ml": __version__,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "splits": {name: _split_summary(part) for name, part in split._asdict().items()},
        "selected_on_validation": {
            "model": selected_model,
            "policy": selected_policy,
            "criterion": f"cross-fitted validation cost ({config.cross_fit_blocks} contiguous time blocks)",
        },
        "models": {},
        "test_baselines": {
            "flag_none": float(costs.row_costs(test_labels, np.zeros(len(test), bool), test_amounts).sum()),
            "flag_all": float(costs.row_costs(test_labels, np.ones(len(test), bool), test_amounts).sum()),
        },
    }
    for name, bundle in bundles.items():
        results["models"][name] = {
            "average_precision": {
                "validation": validation_ap[name],
                "test": float(average_precision_score(test_labels, test_scores[name])),
            },
            "calibration": {
                "raw": calibration_summary(test_labels, test_scores[name]),
                **{method: calibration_summary(test_labels, p) for method, p in probabilities[name].items()},
            },
            "preferred_policy": bundle.preferred_policy,
            "policies": {
                policy_name: {
                    "params": policy.describe(),
                    "validation": validation_results[name][policy_name],
                    "test": decision_metrics(test_labels, decisions[f"{name}/{policy_name}"], test_amounts, costs),
                }
                for policy_name, policy in bundle.policies.items()
            },
        }

    model_names = list(bundles)
    comparisons = [
        (f"{name}/{BASELINE_POLICY}", f"{name}/{policy_name}")
        for name in model_names
        for policy_name in fitters
        if policy_name != BASELINE_POLICY
    ] + [(f"{model_names[0]}/{policy_name}", f"{model_names[1]}/{policy_name}") for policy_name in fitters]
    log.info("Bootstrapping test metrics (%d resamples)", config.n_bootstrap)
    results["bootstrap"] = bootstrap_test_metrics(
        test_labels,
        test_amounts,
        test_scores,
        decisions,
        costs,
        comparisons,
        n_resamples=config.n_bootstrap,
        seed=config.seed,
    )

    for name, bundle in bundles.items():
        bundle.save(output / f"{name}.joblib")
        curves[name].to_csv(output / f"{name}_thresholds.csv", index=False)
        pd.DataFrame(
            {
                "Time": test["Time"],
                "label": test_labels,
                "amount": test_amounts,
                "score": test_scores[name],
                **{f"probability_{method}": p for method, p in probabilities[name].items()},
                **{f"flagged_{policy_name}": decisions[f"{name}/{policy_name}"] for policy_name in bundle.policies},
            }
        ).to_csv(output / f"{name}_test_predictions.csv", index=False)
    evaluation_figure(results, test_labels, test_amounts, test_scores, probabilities, curves, output / "evaluation.png")
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "REPORT.md").write_text(render_report(results))
    return results
