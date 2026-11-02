"""Command-line interface: ``threshold-ml run`` and ``threshold-ml score``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .costs import CostModel
from .pipeline import ExperimentConfig, run_experiment
from .scoring import ModelBundle


def _run(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        source=args.source,
        csv_path=args.csv,
        rows=args.rows,
        seed=args.seed,
        output=Path(args.output),
        costs=CostModel(args.review_cost, args.false_positive_cost, args.loss_rate),
        n_bootstrap=args.bootstrap,
        cross_fit_blocks=args.cross_fit_blocks,
    )
    results = run_experiment(config)
    selected = results["selected_on_validation"]
    model = results["models"][selected["model"]]
    test = model["policies"][selected["policy"]]["test"]
    print(
        json.dumps(
            {
                "source": results["data"]["source"],
                "selected_on_validation": f"{selected['model']} / {selected['policy']}",
                "test_average_precision": round(model["average_precision"]["test"], 4),
                "test_cost": round(test["cost"], 2),
                "report": str(config.output / "REPORT.md"),
            },
            indent=2,
        )
    )


def _score(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.model)
    scored = bundle.score(pd.read_csv(args.input))
    scored.to_csv(args.output, index=False)
    print(
        f"Scored {len(scored):,} rows with {bundle.name} ({bundle.preferred_policy}); "
        f"{int(scored['flagged'].sum()):,} flagged → {args.output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="threshold-ml", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log progress to stderr.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Train, choose decision policies on validation and evaluate on test.")
    run.add_argument(
        "--source",
        choices=["synthetic", "download", "auto", "csv"],
        default="synthetic",
        help="synthetic (offline, default), download (fail on error), auto (fall back to "
        "synthetic, recorded in provenance), or csv (requires --csv).",
    )
    run.add_argument("--csv", help="Path to a local CSV for --source csv.")
    run.add_argument("--rows", type=int, default=20_000, help="Synthetic row count.")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", default="results")
    run.add_argument("--bootstrap", type=int, default=1_000, help="Stratified bootstrap resamples on test.")
    run.add_argument(
        "--cross-fit-blocks", type=int, default=5, help="Contiguous validation blocks for cross-fitted policy costs."
    )
    cost = run.add_argument_group("hypothetical cost model")
    cost.add_argument("--review-cost", type=float, default=2.0, help="Cost of reviewing any alert.")
    cost.add_argument("--false-positive-cost", type=float, default=5.0, help="Extra cost of a false alarm.")
    cost.add_argument("--loss-rate", type=float, default=1.0, help="Fraction of a missed fraud's amount lost.")
    run.set_defaults(handler=_run)

    score = commands.add_parser("score", help="Score new transactions with a saved model bundle.")
    score.add_argument("--model", required=True, help="Path to a trusted .joblib bundle.")
    score.add_argument("--input", required=True, help="CSV with Time, Amount and V1–V28 (Class optional).")
    score.add_argument("--output", required=True, help="Destination CSV.")
    score.set_defaults(handler=_score)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        args.handler(args)
    except (ValueError, OSError) as error:
        sys.exit(f"threshold-ml: error: {error}")
