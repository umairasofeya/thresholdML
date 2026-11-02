"""Six-panel evaluation figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .costs import CostModel

PALETTE = {"logistic_regression": "#3b6ea8", "gradient_boosting": "#d1495b"}
CALIBRATION_STYLE = {"raw": "-", "platt": "--", "isotonic": ":"}
MODEL_LABELS = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}
POLICY_LABELS = {
    "global_threshold": "global threshold",
    "default_0_5": "default 0.5",
    "amount_aware_platt": "amount-aware (Platt)",
    "amount_aware_isotonic": "amount-aware (isotonic)",
}


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " "))


def decision_label(key: str) -> str:
    model, policy = key.split("/")
    return f"{model_label(model)}, {POLICY_LABELS.get(policy, policy)}"


def _best_amount_aware(model: dict) -> str:
    candidates = [name for name in model["policies"] if name.startswith("amount_aware_")]
    return min(candidates, key=lambda name: model["policies"][name]["validation"]["cross_fitted"]["cost"])


def _forest(axis, labels: list[str], intervals: list[dict], colors: list[str], highlight: int | None = None) -> None:
    positions = np.arange(len(labels))[::-1]
    for y, interval, color in zip(positions, intervals, colors, strict=True):
        axis.plot([interval["low"], interval["high"]], [y, y], color=color, linewidth=2)
        axis.scatter(interval["estimate"], y, color=color, edgecolor="black", zorder=5, s=40)
    axis.set_yticks(positions, labels, fontsize=8)
    if highlight is not None:
        axis.get_yticklabels()[highlight].set_fontweight("bold")


def evaluation_figure(
    results: dict,
    test_labels: np.ndarray,
    test_amounts: np.ndarray,
    test_scores: dict[str, np.ndarray],
    probabilities: dict[str, dict[str, np.ndarray]],
    curves: dict,
    path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    (pr_axis, cost_axis, calibration_axis), (rule_axis, forest_axis, difference_axis) = axes
    models, bootstrap = results["models"], results["bootstrap"]
    selected = results["selected_on_validation"]
    costs = CostModel(**results["cost_assumptions"])

    # 1. Ranking quality, with the global-threshold operating point.
    for name, scores in test_scores.items():
        color, operating = PALETTE.get(name), models[name]["policies"]["global_threshold"]["test"]
        precision, recall, _ = precision_recall_curve(test_labels, scores)
        ap = models[name]["average_precision"]["test"]
        pr_axis.step(recall, precision, where="post", color=color, label=f"{model_label(name)} (AP {ap:.3f})")
        pr_axis.scatter(operating["recall"], operating["precision"], color=color, edgecolor="black", zorder=5, s=50)
    pr_axis.axhline(test_labels.mean(), color="gray", linestyle="--", linewidth=1, label="Prevalence")
    pr_axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02))
    pr_axis.set_title("Test precision–recall (dots: global-threshold rule)")

    # 2. The exact validation threshold search.
    for name, curve in curves.items():
        chosen = models[name]["policies"]["global_threshold"]["validation"]["in_sample"]
        cost_axis.plot(curve["alerts"], curve["cost"], color=PALETTE.get(name), label=model_label(name))
        cost_axis.scatter(chosen["flagged"], chosen["cost"], color=PALETTE.get(name), edgecolor="black", zorder=5)
    cost_axis.set_xscale("symlog", linthresh=10)
    cost_axis.set_yscale("log")
    cost_axis.set(xlabel="Validation alerts raised", ylabel="Hypothetical cost")
    cost_axis.set_title("Global threshold search on validation (dots: chosen)")

    # 3. Calibration before and after, for the selected model.
    model_name = selected["model"]
    calibration_axis.plot([0, 1], [0, 1], color="gray", linewidth=1, label="Perfect calibration")
    for method, summary in models[model_name]["calibration"].items():
        bins = summary["bins"]
        calibration_axis.plot(
            [b["mean_score"] for b in bins],
            [b["fraud_rate"] for b in bins],
            CALIBRATION_STYLE.get(method, "-"),
            marker="o",
            markersize=4,
            color=PALETTE.get(model_name),
            label=f"{method} (Brier {summary['brier_score']:.2e}, ECE {summary['expected_calibration_error']:.1e})",
        )
    calibration_axis.set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", xlim=(0, 1), ylim=(0, 1))
    calibration_axis.set_title(f"Test reliability: {model_label(model_name)}")

    # 4. What the amount-aware rule does, in probability × amount space.
    method_policy = _best_amount_aware(models[model_name])
    method = models[model_name]["policies"][method_policy]["params"]["calibration"]
    p = probabilities[model_name][method]
    caught = costs.expected_cost_flags(p, test_amounts)
    amounts = np.maximum(test_amounts, 0.1)
    legit = np.flatnonzero(test_labels == 0)
    sample = np.random.default_rng(0).choice(legit, min(4_000, len(legit)), replace=False)
    rule_axis.scatter(amounts[sample], np.maximum(p[sample], 1e-6), s=4, color="lightgray", label="Legitimate (sample)")
    fraud = test_labels == 1
    rule_axis.scatter(amounts[fraud & caught], p[fraud & caught], s=28, color="#2a9d8f", label="Fraud, flagged")
    rule_axis.scatter(amounts[fraud & ~caught], np.maximum(p[fraud & ~caught], 1e-6), s=28, marker="x",
                      color="black", label="Fraud, missed")  # fmt: skip
    grid = np.logspace(-1, np.log10(max(amounts.max(), 10)), 300)
    rule_axis.plot(grid, np.minimum(costs.break_even_probability(grid), 1), color="#e76f51", linewidth=2,
                   label="Break-even probability")  # fmt: skip
    rule_axis.set(
        xscale="log", yscale="log", ylim=(1e-6, 1.5), xlabel="Amount", ylabel=f"Calibrated P(fraud), {method}"
    )
    rule_axis.set_title("Amount-aware rule: flag above the curve")

    # 5. Test cost of every frozen decision rule.
    keys = sorted(bootstrap["decisions"], key=lambda k: bootstrap["decisions"][k]["cost"]["estimate"])
    selected_key = f"{selected['model']}/{selected['policy']}"
    _forest(
        forest_axis,
        [decision_label(k) for k in keys],
        [bootstrap["decisions"][k]["cost"] for k in keys],
        [PALETTE.get(k.split("/")[0], "gray") for k in keys],
        highlight=keys.index(selected_key),
    )
    forest_axis.axvline(results["test_baselines"]["flag_none"], color="gray", linestyle="--", label="Flag none")
    forest_axis.set(xlabel="Hypothetical test cost")
    forest_axis.set_title(f"Test cost, {bootstrap['confidence']:.0%} bootstrap CI (bold: selected on validation)")

    # 6. Paired differences: what each change actually buys.
    differences = bootstrap["paired_cost_difference"]
    _forest(
        difference_axis,
        [f"{decision_label(d['candidate'])}\n  vs {decision_label(d['baseline'])}" for d in differences],
        differences,
        [PALETTE.get(d["candidate"].split("/")[0], "gray") for d in differences],
    )
    difference_axis.axvline(0, color="black", linewidth=1)
    difference_axis.tick_params(axis="y", labelsize=6.5)
    difference_axis.set(xlabel="Paired test cost difference (negative: candidate cheaper)")
    difference_axis.set_title("Paired bootstrap comparisons")

    for axis in (pr_axis, cost_axis, calibration_axis, forest_axis):
        axis.legend(fontsize=8, frameon=True)
    rule_axis.legend(fontsize=8, frameon=True, loc="lower left")
    figure.suptitle(f"Threshold ML evaluation — data source: {results['data']['source']}", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
