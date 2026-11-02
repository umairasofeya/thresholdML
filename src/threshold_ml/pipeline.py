"""End-to-end experiment: load → validate → split → fit → select thresholds → freeze → evaluate."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .costs import CostModel, optimize_threshold
from .data import LABEL_COLUMN, Source, load_transactions, validate
from .evaluation import bootstrap_test_metrics, calibration_summary, decision_metrics
from .features import build_features
from .models import build_models
from .plots import evaluation_figure
from .report import render_report
from .scoring import ModelBundle
from .splits import chronological_split

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    source: Source = "synthetic"
    csv_path: str | None = None
    rows: int = 20_000
    seed: int = 42
    output: Path = Path("results")
    costs: CostModel = field(default_factory=CostModel)
    n_bootstrap: int = 1_000


def _split_summary(part: pd.DataFrame) -> dict:
    return {
        "rows": len(part),
        "frauds": int(part[LABEL_COLUMN].sum()),
        "time_min": float(part["Time"].min()),
        "time_max": float(part["Time"].max()),
    }


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the full protocol and write results.json, REPORT.md, evaluation.png and per-model artefacts.

    The test partition is touched only after every model and threshold is frozen.
    """
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    costs = config.costs

    raw, provenance = load_transactions(config.source, config.csv_path, config.rows, config.seed)
    frame = validate(raw)
    split = chronological_split(frame)
    log.info("Split sizes: %s", {name: len(part) for name, part in split._asdict().items()})

    # Phase 1: fit on train, choose thresholds on validation. No test data is read here.
    frozen: dict[str, ModelBundle] = {}
    curves: dict[str, pd.DataFrame] = {}
    validation_metrics: dict[str, dict] = {}
    validation_features = build_features(split.validation)
    for name, model in build_models(config.seed).items():
        log.info("Fitting %s", name)
        model.fit(build_features(split.train), split.train[LABEL_COLUMN])
        scores = model.predict_proba(validation_features)[:, 1]
        search = optimize_threshold(split.validation[LABEL_COLUMN], scores, split.validation["Amount"], costs)
        frozen[name] = ModelBundle(name, model, search.threshold, costs, provenance=provenance)
        curves[name] = search.curve
        validation_metrics[name] = decision_metrics(
            split.validation[LABEL_COLUMN], scores, split.validation["Amount"], search.threshold, costs
        )
    selected = min(frozen, key=lambda name: (validation_metrics[name]["cost"], validation_metrics[name]["flagged"]))

    # Phase 2: evaluate the frozen rules on test.
    test, test_labels, test_amounts = split.test, split.test[LABEL_COLUMN].to_numpy(), split.test["Amount"].to_numpy()
    test_features = build_features(test)
    test_scores = {name: bundle.model.predict_proba(test_features)[:, 1] for name, bundle in frozen.items()}
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
        "selected_on_validation": selected,
        "models": {},
        "test_baselines": {
            "flag_none": costs.total(test_labels, np.zeros(len(test)), test_amounts, 1.0),
            "flag_all": costs.total(test_labels, np.ones(len(test)), test_amounts, 0.0),
        },
    }
    for name, bundle in frozen.items():
        scores = test_scores[name]
        results["models"][name] = {
            "validation": validation_metrics[name],
            "test": decision_metrics(test_labels, scores, test_amounts, bundle.threshold, costs),
            "test_at_0_5": decision_metrics(test_labels, scores, test_amounts, 0.5, costs),
            "calibration": calibration_summary(test_labels, scores),
        }
    log.info("Bootstrapping test metrics (%d resamples)", config.n_bootstrap)
    results["bootstrap"] = bootstrap_test_metrics(
        test_labels,
        test_amounts,
        test_scores,
        {name: bundle.threshold for name, bundle in frozen.items()},
        costs,
        n_resamples=config.n_bootstrap,
        seed=config.seed,
    )

    for name, bundle in frozen.items():
        bundle.save(output / f"{name}.joblib")
        curves[name].to_csv(output / f"{name}_thresholds.csv", index=False)
        pd.DataFrame(
            {
                "Time": test["Time"],
                "label": test_labels,
                "amount": test_amounts,
                "score": test_scores[name],
                "flagged": test_scores[name] >= bundle.threshold,
            }
        ).to_csv(output / f"{name}_test_predictions.csv", index=False)
    evaluation_figure(results, test_labels, test_scores, curves, output / "evaluation.png")
    serialisable = {**results, "bootstrap": {k: v for k, v in results["bootstrap"].items() if k != "cost_samples"}}
    (output / "results.json").write_text(json.dumps(serialisable, indent=2) + "\n")
    (output / "REPORT.md").write_text(render_report(results))
    return serialisable
