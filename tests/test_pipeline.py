import json

import numpy as np
import pandas as pd
import pytest

from threshold_ml.cli import main
from threshold_ml.data import synthetic_transactions
from threshold_ml.scoring import ModelBundle


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    output = tmp_path_factory.mktemp("run")
    main(["run", "--source", "synthetic", "--rows", "6000", "--bootstrap", "50", "--output", str(output)])
    return output


def test_run_writes_complete_artifacts(run_dir):
    for name in ["results.json", "REPORT.md", "evaluation.png"]:
        assert (run_dir / name).stat().st_size > 0
    results = json.loads((run_dir / "results.json").read_text())
    assert results["selected_on_validation"] in results["models"]
    assert set(results["bootstrap"]["models"]) == set(results["models"])
    assert sum(split["rows"] for split in results["splits"].values()) == results["rows"] == 6_000


def test_saved_predictions_reproduce_reported_metrics(run_dir):
    """Independent recomputation from the published CSVs, as a reviewer would do it."""
    results = json.loads((run_dir / "results.json").read_text())
    costs = results["cost_assumptions"]
    for name, model in results["models"].items():
        predictions = pd.read_csv(run_dir / f"{name}_test_predictions.csv")
        flagged, labels = predictions["flagged"].to_numpy(), predictions["label"].to_numpy()
        cost = (
            costs["review_cost"] * flagged.sum()
            + costs["false_positive_cost"] * (flagged & (labels == 0)).sum()
            + costs["loss_rate"] * predictions["amount"][~flagged & (labels == 1)].sum()
        )
        assert cost == pytest.approx(model["test"]["cost"])
        assert model["test"]["confusion_matrix"]["tp"] == (flagged & (labels == 1)).sum()


def test_saved_bundle_scores_unlabelled_rows_identically(run_dir, tmp_path):
    bundle = ModelBundle.load(run_dir / "gradient_boosting.joblib")
    predictions = pd.read_csv(run_dir / "gradient_boosting_test_predictions.csv")
    rows = synthetic_transactions(6_000).sort_values("Time", kind="stable").reset_index(drop=True)
    test_rows = rows[rows["Time"] >= predictions["Time"].min()].drop(columns="Class")
    test_rows.to_csv(tmp_path / "new.csv", index=False)
    main(
        [
            "score",
            "--model",
            str(run_dir / "gradient_boosting.joblib"),
            "--input",
            str(tmp_path / "new.csv"),
            "--output",
            str(tmp_path / "scored.csv"),
        ]
    )
    scored = pd.read_csv(tmp_path / "scored.csv")
    assert np.allclose(scored["score"], predictions["score"])
    assert (scored["flagged"] == (scored["score"] >= bundle.threshold)).all()


def test_cli_reports_bad_costs_cleanly(capsys):
    with pytest.raises(SystemExit, match="non-negative"):
        main(["run", "--review-cost", "-1"])
