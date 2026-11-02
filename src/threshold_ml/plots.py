"""Four-panel evaluation figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

PALETTE = {"logistic_regression": "#3b6ea8", "gradient_boosting": "#d1495b"}
LABELS = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def evaluation_figure(results: dict, test_labels: np.ndarray, test_scores: dict, curves: dict, path: Path) -> None:
    """PR curves with operating points, validation cost curves, reliability and bootstrap cost."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    (pr_axis, cost_axis), (calibration_axis, bootstrap_axis) = axes
    models = results["models"]

    for name, scores in test_scores.items():
        color, test = PALETTE.get(name), models[name]["test"]
        precision, recall, _ = precision_recall_curve(test_labels, scores)
        pr_axis.step(
            recall, precision, where="post", color=color, label=f"{_label(name)} (AP {test['average_precision']:.3f})"
        )
        pr_axis.scatter(test["recall"], test["precision"], color=color, edgecolor="black", zorder=5, s=50)
    pr_axis.axhline(test_labels.mean(), color="gray", linestyle="--", linewidth=1, label="Prevalence")
    pr_axis.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.01),
        ylim=(0, 1.02),
        title="Test precision–recall (dots: frozen operating points)",
    )

    for name, curve in curves.items():
        color, validation = PALETTE.get(name), models[name]["validation"]
        cost_axis.plot(curve["alerts"], curve["cost"], color=color, label=_label(name))
        cost_axis.scatter(validation["flagged"], validation["cost"], color=color, edgecolor="black", zorder=5, s=50)
    cost_axis.set_xscale("symlog", linthresh=10)
    cost_axis.set_yscale("log")
    cost_axis.set(
        xlabel="Validation alerts raised",
        ylabel="Hypothetical cost",
        title="Validation cost by alert volume (dots: selected)",
    )

    calibration_axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect calibration")
    for name in test_scores:
        calibration = models[name]["calibration"]
        bins = calibration["bins"]
        calibration_axis.plot(
            [b["mean_score"] for b in bins],
            [b["fraud_rate"] for b in bins],
            marker="o",
            color=PALETTE.get(name),
            label=f"{_label(name)} (Brier {calibration['brier_score']:.2e})",
        )
    calibration_axis.set(
        xlabel="Mean predicted score",
        ylabel="Observed fraud rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Test reliability (10 equal-width bins)",
    )

    bootstrap = results["bootstrap"]
    for name, samples in bootstrap["cost_samples"].items():
        color = PALETTE.get(name)
        bootstrap_axis.hist(samples, bins=40, alpha=0.5, color=color, label=_label(name))
        bootstrap_axis.axvline(models[name]["test"]["cost"], color=color, linewidth=2)
    bootstrap_axis.set(
        xlabel="Hypothetical test cost",
        ylabel="Resamples",
        title=f"Bootstrap test cost ({bootstrap['n_resamples']:,} stratified resamples)",
    )

    for axis in axes.flat:
        axis.legend(fontsize=8, frameon=True)
    figure.suptitle(f"Threshold ML evaluation — data source: {results['data']['source']}", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
