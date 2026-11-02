"""Point metrics, calibration diagnostics and stratified bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from .costs import CostModel

Interval = dict[str, float]


def _precision_recall(labels: NDArray, flagged: NDArray) -> tuple[float, float]:
    true_positives = int((flagged & (labels == 1)).sum())
    alerts, positives = int(flagged.sum()), int((labels == 1).sum())
    return (true_positives / alerts if alerts else 0.0), (true_positives / positives if positives else 0.0)


def decision_metrics(
    labels: ArrayLike, scores: ArrayLike, amounts: ArrayLike, threshold: float, costs: CostModel
) -> dict:
    """Ranking quality (AP) plus the outcome of the frozen decision rule ``score >= threshold``."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    flagged = scores >= threshold
    precision, recall = _precision_recall(labels, flagged)
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "flagged": int(flagged.sum()),
        "cost": costs.total(labels, scores, amounts, threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def calibration_summary(labels: ArrayLike, scores: ArrayLike, bins: int = 10) -> dict:
    """Brier score and expected calibration error over equal-width score bins.

    Thresholds are chosen on cost, so calibration isn't required for the decision rule,
    but it matters if scores are ever read as probabilities.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    mean_score = np.bincount(index, weights=scores, minlength=bins) / np.maximum(counts, 1)
    fraud_rate = np.bincount(index, weights=labels, minlength=bins) / np.maximum(counts, 1)
    occupied = counts > 0
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "expected_calibration_error": float(np.sum(counts * np.abs(mean_score - fraud_rate)) / len(scores)),
        "bins": [
            {"mean_score": float(s), "fraud_rate": float(r), "count": int(c)}
            for s, r, c in zip(mean_score[occupied], fraud_rate[occupied], counts[occupied], strict=True)
        ],
    }


def stratified_resamples(labels: ArrayLike, n_resamples: int, seed: int) -> Iterator[NDArray[np.intp]]:
    """Yield bootstrap index arrays that resample frauds and legitimate rows separately.

    With only tens of test frauds, an unstratified bootstrap would sometimes draw far
    fewer positives (or none), inflating variance for reasons unrelated to the model.
    """
    labels = np.asarray(labels)
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield np.concatenate([rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))])


def _interval(estimate: float, samples: NDArray, confidence: float) -> Interval:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"estimate": float(estimate), "low": float(low), "high": float(high)}


def bootstrap_test_metrics(
    labels: ArrayLike,
    amounts: ArrayLike,
    model_scores: dict[str, ArrayLike],
    thresholds: dict[str, float],
    costs: CostModel,
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Percentile intervals for each model's test metrics, plus paired cost differences.

    Every model is evaluated on the *same* resamples, so differences between models are
    paired: shared sampling noise (an unusually expensive missed fraud, say) cancels out.
    Thresholds stay frozen; this measures test-set sampling uncertainty only, not the
    variability of training or threshold selection.
    """
    labels, amounts = np.asarray(labels), np.asarray(amounts, dtype=float)
    names = list(model_scores)
    scores = {name: np.asarray(model_scores[name]) for name in names}
    flagged = {name: scores[name] >= thresholds[name] for name in names}
    row_costs = {name: costs.row_costs(labels, flagged[name], amounts) for name in names}
    samples: dict[str, dict[str, list[float]]] = {
        name: {"average_precision": [], "precision": [], "recall": [], "cost": []} for name in names
    }
    for index in stratified_resamples(labels, n_resamples, seed):
        sample_labels = labels[index]
        for name in names:
            precision, recall = _precision_recall(sample_labels, flagged[name][index])
            samples[name]["average_precision"].append(average_precision_score(sample_labels, scores[name][index]))
            samples[name]["precision"].append(precision)
            samples[name]["recall"].append(recall)
            samples[name]["cost"].append(row_costs[name][index].sum())

    point = {name: decision_metrics(labels, scores[name], amounts, thresholds[name], costs) for name in names}
    result: dict = {
        "n_resamples": n_resamples,
        "confidence": confidence,
        "method": "stratified percentile bootstrap over test rows, thresholds frozen",
        "models": {
            name: {
                metric: _interval(point[name][metric], np.asarray(values), confidence)
                for metric, values in samples[name].items()
            }
            for name in names
        },
        "cost_samples": {name: samples[name]["cost"] for name in names},
        "paired_cost_difference": {},
    }
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            difference = np.asarray(samples[second]["cost"]) - np.asarray(samples[first]["cost"])
            result["paired_cost_difference"][f"{second}_minus_{first}"] = {
                **_interval(point[second]["cost"] - point[first]["cost"], difference, confidence),
                f"share_{second}_cheaper": float((difference < 0).mean()),
            }
    return result
