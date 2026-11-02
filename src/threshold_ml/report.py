"""Markdown summary of a finished run."""

from __future__ import annotations


def _ci(interval: dict, fmt: str = ".3f") -> str:
    return f"{interval['estimate']:{fmt}} [{interval['low']:{fmt}}, {interval['high']:{fmt}}]"


def render_report(results: dict) -> str:
    source = results["data"]["source"]
    bootstrap = results["bootstrap"]
    confidence = f"{bootstrap['confidence']:.0%}"
    splits = results["splits"]
    lines = [
        f"# Threshold ML run — `{source}` data",
        "",
        f"{results['rows']:,} transactions; chronological split train/validation/test = "
        + " / ".join(f"{splits[name]['rows']:,} ({splits[name]['frauds']} frauds)" for name in splits)
        + ".",
        f"Selected on validation cost: **{results['selected_on_validation']}**.",
        "",
        f"## Test metrics at frozen validation thresholds ({confidence} bootstrap intervals)",
        "",
        "| Model | Threshold | Average precision | Precision | Recall | Hypothetical cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in results["models"].items():
        intervals = bootstrap["models"][name]
        lines.append(
            f"| {name} | {model['test']['threshold']:.4g} | {_ci(intervals['average_precision'])} | "
            f"{_ci(intervals['precision'])} | {_ci(intervals['recall'])} | {_ci(intervals['cost'], ',.0f')} |"
        )
    lines += [
        "",
        "## Decision outcomes on test",
        "",
        "| Model | Rule | Alerts | TP | FP | FN | Cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in results["models"].items():
        for rule, key in (("validation threshold", "test"), ("default 0.5", "test_at_0_5")):
            m = model[key]
            cm = m["confusion_matrix"]
            lines.append(
                f"| {name} | {rule} | {m['flagged']:,} | {cm['tp']} | {cm['fp']:,} | {cm['fn']} | {m['cost']:,.2f} |"
            )
    baselines = results["test_baselines"]
    lines += [
        f"| — | flag none | 0 | 0 | 0 | — | {baselines['flag_none']:,.2f} |",
        f"| — | flag all | all | — | — | 0 | {baselines['flag_all']:,.2f} |",
        "",
        "## Model comparison",
        "",
    ]
    for key, diff in bootstrap["paired_cost_difference"].items():
        share_key = next(name for name in diff if name.startswith("share_"))
        cheaper = share_key.removeprefix("share_").removesuffix("_cheaper")
        lines.append(
            f"- `{key}` test cost: {_ci(diff, ',.0f')}. "
            f"{cheaper} is cheaper in {diff[share_key]:.0%} of paired resamples."
        )
    lines += [
        "",
        "## Calibration (test)",
        "",
        "| Model | Brier score | Expected calibration error |",
        "|---|---:|---:|",
    ]
    for name, model in results["models"].items():
        c = model["calibration"]
        lines.append(f"| {name} | {c['brier_score']:.2e} | {c['expected_calibration_error']:.2e} |")
    lines += [
        "",
        "Synthetic data demonstrates software behaviour only."
        if source == "synthetic"
        else "A single chronological holdout; intervals capture test sampling noise, not training variability.",
        "Costs are hypothetical, not realised savings.",
        "",
        "![Evaluation](evaluation.png)",
        "",
    ]
    return "\n".join(lines)
