import numpy as np
import pytest

from threshold_ml.data import DataValidationError, load_transactions, synthetic_transactions, validate
from threshold_ml.features import FEATURE_NAMES, build_features
from threshold_ml.splits import chronological_split


def test_synthetic_data_is_seeded():
    assert synthetic_transactions(5_000).equals(synthetic_transactions(5_000))
    assert not synthetic_transactions(5_000).equals(synthetic_transactions(5_000, seed=43))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("Amount", -1, "non-negative"), ("Time", np.nan, "non-finite"), ("Class", 2, "binary"), ("V3", "x", "numeric")],
)
def test_validate_rejects_bad_rows(column, value, message):
    frame = synthetic_transactions(1_000)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate(frame)


def test_validate_reports_missing_columns():
    with pytest.raises(DataValidationError, match="V28"):
        validate(synthetic_transactions(1_000).drop(columns="V28"))


def test_validate_sorts_by_time_and_allows_unlabelled_rows():
    frame = synthetic_transactions(1_000).sample(frac=1, random_state=0)
    clean = validate(frame.drop(columns="Class"), require_label=False)
    assert clean["Time"].is_monotonic_increasing
    assert "Class" not in clean


def test_split_boundaries_are_strict_and_keep_ties_together():
    frame = validate(synthetic_transactions(10_000))
    frame["Time"] = frame["Time"] // 1_000  # force many rows to share timestamps
    train, validation, test = chronological_split(frame)
    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_split_requires_both_classes_everywhere():
    frame = synthetic_transactions(1_000).assign(Class=0)
    with pytest.raises(ValueError, match="both classes"):
        chronological_split(frame)


def test_features_are_row_local():
    frame = synthetic_transactions(5_000)
    expected = build_features(frame).iloc[:10]
    frame["Class"] = 1 - frame["Class"]
    assert expected.equals(build_features(frame.iloc[:10]))
    assert tuple(expected.columns) == FEATURE_NAMES


def test_auto_source_falls_back_explicitly(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("threshold_ml.data.urlopen", offline)
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        _, provenance = load_transactions("auto", rows=1_000)
    assert provenance["source"] == "synthetic"
    assert provenance["fallback_reason"] == "offline"
    with pytest.raises(OSError):
        load_transactions("download", rows=1_000)


def test_csv_source_records_hash(tmp_path):
    path = tmp_path / "data.csv"
    synthetic_transactions(1_000).to_csv(path, index=False)
    frame, provenance = load_transactions("csv", path)
    assert len(frame) == 1_000
    assert len(provenance["sha256"]) == 64
    with pytest.raises(ValueError, match="--csv"):
        load_transactions("csv")
