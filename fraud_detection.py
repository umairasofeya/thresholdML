"""Cost-sensitive chronological credit card fraud benchmark."""
import argparse
import hashlib
import json
import platform
import warnings
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
FEATURES = [f'V{number}' for number in range(1, 29)]


def synthetic_data(rows=20000, seed=42):
    if rows < 1000:
        raise ValueError('Use at least 1000 synthetic rows.')
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 28))
    times = np.sort(rng.integers(0, 172800, rows))
    amount = np.round(rng.lognormal(3.5, 1.2, rows), 2)
    risk = -5.5 + 1.2 * latent[:, 0] - latent[:, 1] + 1.5 * (latent[:, 2] * latent[:, 3] > 1.5) + .3 * (times > 120000)
    labels = rng.binomial(1, 1 / (1 + np.exp(-risk)))
    return pd.DataFrame(latent, columns=FEATURES).assign(Time=times, Amount=amount, Class=labels)


def validate(frame):
    required = ['Time', 'Amount', 'Class'] + FEATURES
    if not set(required).issubset(frame.columns):
        raise ValueError('Required columns: Time, Amount, Class, V1 through V28.')
    frame = frame[required].copy()
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite input.')
    if not set(frame.Class.unique()).issubset({0, 1}):
        raise ValueError('Class must be binary.')
    if (frame.Amount < 0).any() or (frame.Time < 0).any():
        raise ValueError('Amount and Time must be nonnegative.')
    return frame.sort_values('Time', kind='stable').reset_index(drop=True)


def load_data(source, path, rows, seed):
    if source == 'synthetic':
        return synthetic_data(rows, seed), {'source': 'synthetic', 'seed': seed}
    if source == 'csv':
        payload = Path(path).read_bytes()
        return pd.read_csv(BytesIO(payload)), {'source': 'csv', 'sha256': hashlib.sha256(payload).hexdigest()}
    try:
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
        return validate(pd.read_csv(BytesIO(payload))), {'source': 'public_download', 'url': URL, 'sha256': hashlib.sha256(payload).hexdigest()}
    except Exception as error:
        if source != 'auto':
            raise
        warnings.warn(f'Download failed; using SYNTHETIC data: {error}')
        return synthetic_data(rows, seed), {'source': 'synthetic', 'seed': seed, 'fallback_reason': str(error)}


def chronological_split(frame):
    times = np.sort(frame.Time.unique())
    if len(times) < 10:
        raise ValueError('At least 10 unique timestamps required.')
    first, second = times[int(len(times) * .6)], times[int(len(times) * .8)]
    partitions = [frame[frame.Time < first], frame[(frame.Time >= first) & (frame.Time < second)], frame[frame.Time >= second]]
    if any(part.Class.nunique() != 2 for part in partitions):
        raise ValueError('Every time partition needs both classes.')
    return partitions


def features(frame):
    result = frame[FEATURES].copy()
    result['log_amount'] = np.log1p(frame.Amount)
    phase = 2 * np.pi * (frame.Time % 86400) / 86400
    result['day_sin'] = np.sin(phase)
    result['day_cos'] = np.cos(phase)
    result['latent_norm'] = np.sqrt((frame[FEATURES] ** 2).sum(axis=1))
    return result


def decision_cost(labels, scores, amounts, threshold, review_cost=2., false_positive_cost=5., loss_rate=1.):
    flagged = np.asarray(scores) >= threshold
    labels = np.asarray(labels)
    return float(review_cost * flagged.sum() + false_positive_cost * ((labels == 0) & flagged).sum() + loss_rate * np.asarray(amounts)[(labels == 1) & ~flagged].sum())


def optimize_threshold(labels, scores, amounts, review_cost=2., false_positive_cost=5., loss_rate=1.):
    """Exact O(n log n) sweep, grouping ties and including flag-none."""
    labels, scores, amounts = map(np.asarray, (labels, scores, amounts))
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    deltas = review_cost + false_positive_cost * (sorted_labels == 0) - loss_rate * amounts[order] * sorted_labels
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), sorted_scores[ends]]
    costs = np.r_[loss_rate * amounts[labels == 1].sum(), loss_rate * amounts[labels == 1].sum() + np.cumsum(deltas)[ends]]
    return float(thresholds[np.argmin(costs)]), pd.DataFrame({'threshold': thresholds, 'cost': costs})


def evaluate(frame, scores, threshold, costs):
    predicted = scores >= threshold
    return {'threshold': threshold, 'average_precision': float(average_precision_score(frame.Class, scores)),
            'precision': float(precision_score(frame.Class, predicted, zero_division=0)),
            'recall': float(recall_score(frame.Class, predicted, zero_division=0)),
            'flagged': int(predicted.sum()), 'cost': decision_cost(frame.Class, scores, frame.Amount, threshold, **costs),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(frame.Class, predicted, labels=[0, 1]).ravel().tolist()}


def run(args):
    costs = dict(review_cost=args.review_cost, false_positive_cost=args.false_positive_cost, loss_rate=args.loss_rate)
    if not np.isfinite(list(costs.values())).all() or min(costs.values()) < 0:
        raise ValueError('Costs must be finite and nonnegative.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, provenance = load_data(args.source, args.csv, args.rows, args.seed)
    frame = validate(frame)
    train, validation, test = chronological_split(frame)
    models = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed)),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.08, l2_regularization=5., early_stopping=False, random_state=args.seed)}
    report = {'data': provenance, 'rows': len(frame), 'seed': args.seed, 'cost_assumptions': costs,
              'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
              'splits': {name: {'rows': len(part), 'frauds': int(part.Class.sum()), 'time_min': float(part.Time.min()), 'time_max': float(part.Time.max())} for name, part in zip(['train', 'validation', 'test'], [train, validation, test])}, 'models': {}}
    plt.style.use('seaborn-v0_8-whitegrid')
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, model in models.items():
        model.fit(features(train), train.Class)
        validation_scores = model.predict_proba(features(validation))[:, 1]
        threshold, curve = optimize_threshold(validation.Class, validation_scores, validation.Amount, **costs)
        curve.to_csv(output / f'{name}_thresholds.csv', index=False)
        scores = model.predict_proba(features(test))[:, 1]
        report['models'][name] = {'validation': evaluate(validation, validation_scores, threshold, costs), 'test': evaluate(test, scores, threshold, costs), 'test_at_0_5': evaluate(test, scores, .5, costs)}
        joblib.dump({'model': model, 'threshold': threshold, 'feature_names': list(features(train).columns), 'costs': costs, 'provenance': provenance}, output / f'{name}.joblib')
        pd.DataFrame({'Time': test.Time, 'label': test.Class, 'amount': test.Amount, 'score': scores, 'flagged': scores >= threshold}).to_csv(output / f'{name}_test_predictions.csv', index=False)
        precision, recall, _ = precision_recall_curve(test.Class, scores)
        axes[0].plot(recall, precision, label=f"{name} (AP {report['models'][name]['test']['average_precision']:.3f})")
        axes[1].plot(curve.threshold, curve.cost, label=name)
    report['selected_on_validation'] = min(report['models'], key=lambda name: report['models'][name]['validation']['cost'])
    report['test_baselines'] = {'flag_none': decision_cost(test.Class, np.zeros(len(test)), test.Amount, 1, **costs), 'flag_all': decision_cost(test.Class, np.ones(len(test)), test.Amount, 0, **costs)}
    axes[0].axhline(test.Class.mean(), color='gray', linestyle='--', label='Test prevalence')
    axes[0].set(xlabel='Recall', ylabel='Precision', title=f"Test precision–recall | {provenance['source']}")
    axes[1].set(xlabel='Threshold', ylabel='Hypothetical cost (currency units)', title='Validation threshold search')
    axes[1].set_yscale('symlog', linthresh=1)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'evaluation.png', dpi=170)
    plt.close(figure)
    (output / 'results.json').write_text(json.dumps(report, indent=2))
    summary = [f"# Measured run: {provenance['source']}", '', f"Rows: {len(frame):,}. Selected using validation cost: **{report['selected_on_validation']}**.", '', '| Model | Test AP | Precision | Recall | Hypothetical test cost |', '|---|---:|---:|---:|---:|']
    for name, result in report['models'].items():
        metrics = result['test']
        summary.append(f"| {name} | {metrics['average_precision']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['cost']:.2f} |")
    summary += ['', 'Synthetic results demonstrate software behavior only.' if provenance['source'] == 'synthetic' else 'Measured on a single chronological dataset holdout, not a production performance guarantee.', 'Costs are hypothetical, not realized savings.', '', '![Evaluation](evaluation.png)']
    (output / 'REPORT.md').write_text('\n'.join(summary) + '\n')
    print(json.dumps({'source': provenance['source'], 'selected_on_validation': report['selected_on_validation'], 'results': str(output / 'results.json')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['synthetic', 'download', 'auto', 'csv'], default='synthetic')
    parser.add_argument('--csv', help='Local CSV for --source csv')
    parser.add_argument('--rows', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results')
    parser.add_argument('--review-cost', type=float, default=2.)
    parser.add_argument('--false-positive-cost', type=float, default=5.)
    parser.add_argument('--loss-rate', type=float, default=1.)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
