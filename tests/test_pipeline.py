import numpy as np
import pytest
from fraud_detection import synthetic_data, validate, chronological_split, features, decision_cost, optimize_threshold, load_data


def test_seed():
    assert synthetic_data(5000).equals(synthetic_data(5000))
    assert not synthetic_data(5000).equals(synthetic_data(5000, 43))


def test_time_boundaries():
    frame = validate(synthetic_data(10000))
    frame.Time = frame.Time // 1000
    train, validation, test = chronological_split(frame)
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_features_ignore_labels_and_future():
    frame = synthetic_data(5000)
    expected = features(frame).iloc[:10]
    frame.Class = 1 - frame.Class
    assert expected.equals(features(frame.iloc[:10]))
    assert 'Class' not in expected.columns


@pytest.mark.parametrize('seed', range(5))
def test_threshold_brute_force(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 100)
    scores = rng.integers(0, 10, 100) / 10
    amounts = rng.uniform(0, 100, 100)
    threshold, curve = optimize_threshold(labels, scores, amounts)
    brute = [decision_cost(labels, scores, amounts, cutoff) for cutoff in curve.threshold]
    assert decision_cost(labels, scores, amounts, threshold) == pytest.approx(min(brute))
    assert np.allclose(curve.cost, brute)


def test_invalid_data():
    frame = synthetic_data(1000)
    frame.loc[0, 'Amount'] = -1
    with pytest.raises(ValueError):
        validate(frame)


def test_one_class_partition():
    frame = synthetic_data(1000)
    frame.Class = 0
    with pytest.raises(ValueError, match='both classes'):
        chronological_split(frame)


def test_cost_accounting():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .8, .2, .9])
    amounts = np.array([10, 20, 100, 200])
    assert decision_cost(labels, scores, amounts, .5) == 109
    assert decision_cost(labels, scores, amounts, 2) == 300
    assert decision_cost(labels, scores, amounts, 0) == 18


def test_explicit_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr('fraud_detection.urlopen', fail)
    with pytest.warns(UserWarning, match='SYNTHETIC'):
        _, provenance = load_data('auto', None, 1000, 42)
    assert provenance['source'] == 'synthetic'
    with pytest.raises(OSError):
        load_data('download', None, 1000, 42)
