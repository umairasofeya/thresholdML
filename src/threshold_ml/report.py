"""Markdown summary of a finished run."""

from __future__ import annotations


def _ci(interval: dict, fmt: str = ".3f") -> str:
    return f"{interval['estimate']:{fmt}} [{interval['low']:{fmt}}, {interval['high']:{fmt}}]"


def _policy_params(params: dict) -> str:
    if params["type"] == "global_threshold":
        return f"score ≥ {params['threshold']:.4g}"
    return f"expected cost, {params['calibration']} calibration"


def render_report(results: dict) -> str:
    source = results["data"]["source"]
    bootstrap = results["bootstrap"]
    confidence = f"{bootstrap['confidence']:.0%}"
    splits = results["splits"]
    selected = results["selected_on_validation"]
    lines = [
        f"# Threshold ML run — `{source}` data",
        "",
        f"{results['rows']:,} transactions; chronological split train/validation/test = "
        + " / ".join(f"{splits[name]['rows']:,} ({splits[name]['frauds']} frauds)" for name in splits)
        + ".",
        f"Selected on {selected['criterion']}: **{selected['model']} / {selected['policy']}**.",
        "",
        f"## Ranking quality ({confidence} bootstrap intervals)",
        "",
        "| Model | Validation AP | Test AP |",
        "|---|---:|---:|",
    ]
    for name, model in results["models"].items():
        lines.append(
            f"| {name} | {model['average_precision']['validation']:.3f} | {_ci(bootstrap['average_precision'][name])} |"
        )
    lines += [
        "",
        "## Decision policies",
        "",
        "Validation cost is cross-fitted (each block decided by a policy fitted on the other blocks); "
        "in-sample cost is shown for reference and is optimistic.",
        "",
        "| Model | Policy | Rule | Validation cost (cross-fitted / in-sample) | Test alerts | TP | FP | FN "
        f"| Test cost ({confidence} CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in results["models"].items():
        for policy_name, policy in model["policies"].items():
            validation, test = policy["validation"], policy["test"]
            cm = test["confusion_matrix"]
            marker = " **(selected)**" if (name, policy_name) == (selected["model"], selected["policy"]) else ""
            lines.append(
                f"| {name} | {policy_name}{marker} | {_policy_params(policy['params'])} | "
                f"{validation['cross_fitted']['cost']:,.0f} / {validation['in_sample']['cost']:,.0f} | "
                f"{test['flagged']:,} | {cm['tp']} | {cm['fp']:,} | {cm['fn']} | "
                f"{_ci(bootstrap['decisions'][f'{name}/{policy_name}']['cost'], ',.0f')} |"
            )
    baselines = results["test_baselines"]
    lines += [
        "",
        f"Baselines on test: flag none {baselines['flag_none']:,.2f}; flag all {baselines['flag_all']:,.2f}.",
        "",
        "## Paired comparisons on test",
        "",
        "| Candidate | Baseline | Cost difference | Candidate cheaper in |",
        "|---|---|---:|---:|",
    ]
    for difference in bootstrap["paired_cost_difference"]:
        lines.append(
            f"| {difference['candidate']} | {difference['baseline']} | {_ci(difference, ',.0f')} | "
            f"{difference['share_candidate_cheaper']:.0%} |"
        )
    lines += ["", "## Calibration (test)", "", "| Model | Probabilities | Brier score | ECE |", "|---|---|---:|---:|"]
    for name, model in results["models"].items():
        for method, summary in model["calibration"].items():
            lines.append(
                f"| {name} | {method} | {summary['brier_score']:.2e} | {summary['expected_calibration_error']:.2e} |"
            )
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
