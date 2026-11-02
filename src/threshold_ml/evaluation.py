"""Decision metrics, calibration diagnostics and paired stratified bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from .costs import CostModel

Interval = dict[str, float]


def _precision_recall(labels: NDArray, flagged: NDArray) -> tuple[float, float]:
    true_positives = int((flagged & (labels == 1)).sum())
    alerts, positives = int(flagged.sum()), int((labels == 1).sum())
    return (true_positives / alerts if alerts else 0.0), (true_positives / positives if positives else 0.0)


def decision_metrics(labels: ArrayLike, flagged: ArrayLike, amounts: ArrayLike, costs: CostModel) -> dict:
    """Outcome of a frozen set of flag / don't-flag decisions."""
    labels, flagged = np.asarray(labels), np.asarray(flagged, dtype=bool)
    precision, recall = _precision_recall(labels, flagged)
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel().tolist()
    return {
        "precision": precision,
        "recall": recall,
        "flagged": int(flagged.sum()),
        "cost": float(costs.row_costs(labels, flagged, amounts).sum()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def calibration_summary(labels: ArrayLike, probabilities: ArrayLike, bins: int = 10) -> dict:
    """Brier score and expected calibration error over equal-width probability bins."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    mean_score = np.bincount(index, weights=probabilities, minlength=bins) / np.maximum(counts, 1)
    fraud_rate = np.bincount(index, weights=labels, minlength=bins) / np.maximum(counts, 1)
    occupied = counts > 0
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(np.sum(counts * np.abs(mean_score - fraud_rate)) / len(probabilities)),
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
    decisions: dict[str, ArrayLike],
    costs: CostModel,
    comparisons: Sequence[tuple[str, str]] = (),
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Percentile intervals for ranking quality per model and outcomes per decision rule.

    Every model and rule is evaluated on the *same* resamples, so each ``(baseline,
    candidate)`` pair in ``comparisons`` gets a paired difference: shared sampling noise
    (an unusually expensive missed fraud, say) cancels out. Decisions stay frozen; this
    measures test-set sampling uncertainty only, not training or selection variability.
    """
    labels, amounts = np.asarray(labels), np.asarray(amounts, dtype=float)
    scores = {name: np.asarray(values) for name, values in model_scores.items()}
    flags = {name: np.asarray(values, dtype=bool) for name, values in decisions.items()}
    row_costs = {name: costs.row_costs(labels, flagged, amounts) for name, flagged in flags.items()}
    ap_samples: dict[str, list[float]] = {name: [] for name in scores}
    decision_samples: dict[str, dict[str, list[float]]] = {
        name: {"precision": [], "recall": [], "cost": []} for name in flags
    }
    for index in stratified_resamples(labels, n_resamples, seed):
        sample_labels = labels[index]
        for name, values in scores.items():
            ap_samples[name].append(average_precision_score(sample_labels, values[index]))
        for name, flagged in flags.items():
            precision, recall = _precision_recall(sample_labels, flagged[index])
            decision_samples[name]["precision"].append(precision)
            decision_samples[name]["recall"].append(recall)
            decision_samples[name]["cost"].append(row_costs[name][index].sum())

    point = {name: decision_metrics(labels, flagged, amounts, costs) for name, flagged in flags.items()}
    result: dict = {
        "n_resamples": n_resamples,
        "confidence": confidence,
        "method": "stratified percentile bootstrap over test rows, decisions frozen",
        "average_precision": {
            name: _interval(average_precision_score(labels, scores[name]), np.asarray(samples), confidence)
            for name, samples in ap_samples.items()
        },
        "decisions": {
            name: {
                metric: _interval(point[name][metric], np.asarray(values), confidence)
                for metric, values in decision_samples[name].items()
            }
            for name in flags
        },
        "paired_cost_difference": [],
    }
    for baseline, candidate in comparisons:
        difference = np.asarray(decision_samples[candidate]["cost"]) - np.asarray(decision_samples[baseline]["cost"])
        result["paired_cost_difference"].append(
            {
                "baseline": baseline,
                "candidate": candidate,
                **_interval(point[candidate]["cost"] - point[baseline]["cost"], difference, confidence),
                "share_candidate_cheaper": float((difference < 0).mean()),
            }
        )
    return result
