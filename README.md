# Threshold ML

Threshold ML is a reproducible Python project for cost-sensitive credit card fraud detection, comparing logistic regression and histogram gradient boosting on chronological transaction splits, with engineered features and validation-selected financial decision thresholds.

## Run locally

Python 3.11+ recommended. From the `threshold-ml` folder:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python fraud_detection.py --source synthetic --rows 20000 --output results
```

Windows activation: `.venv\Scripts\activate`. Open `results/REPORT.md` and `results/evaluation.png`. The ZIP includes actual measured examples, trained model bundles, prediction CSVs and threshold curves under `examples/public/` and `examples/synthetic/`.

## Public dataset

The benchmark uses the public anonymized credit card fraud CSV referenced in the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) and the [original Kaggle listing](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Review the source terms before redistributing raw data. No raw public CSV is included here.

```sh
python fraud_detection.py --source download --output results/public
# Try download, with an explicit, labeled synthetic fallback on failure:
python fraud_detection.py --source auto --output results/auto
# Use a downloaded copy:
python fraud_detection.py --source csv --csv data/creditcard.csv --output results/public
```

If Python has local certificate issues, use your system's verified HTTPS downloader:

```sh
mkdir -p data
curl --fail --location --retry 2 https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv --output data/creditcard.csv
python fraud_detection.py --source csv --csv data/creditcard.csv --output results/public
```

Defaults use offline synthetic data. Explicit download mode fails loudly. Only auto mode falls back and records the reason. Local/download inputs receive SHA-256 hashes; retain the CSV and compare hashes for exact input reproduction. `--seed` defaults to 42. `requirements-lock.txt` records the verified environment; platform differences can affect floating point results.

## Measured real-data results

Full dataset: **284,807 transactions, 492 frauds**. Chronological test set: **60,140 transactions, 77 frauds**. See [results.json](examples/public/results.json) for full metrics, boundaries, versions and provenance.

| Model | Test average precision | Precision | Recall | Hypothetical cost |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7437 | 24.44% | 85.71% | 3,922.20 |
| Histogram gradient boosting | 0.7878 | 86.36% | 74.03% | 3,851.44 |

Gradient boosting was selected by validation cost. At its frozen threshold of 0.389157 it flagged 66 test transactions: 57 true positives and 9 false positives, with 20 missed frauds. Its default 0.5 threshold had slightly **lower test cost (3,837.44)** than the validation-selected threshold. No test-driven retuning was performed. Flag-none cost was 8,396.81 under the same hypothetical assumptions. These are experimental costs, not realized savings.

![Real-data evaluation](examples/public/evaluation.png)

## Design and leakage controls

1. Validate Time, Amount, Class and V1–V28; reject invalid labels, missing/nonfinite values and negative times/amounts.
2. Stable-sort by Time; assign 60/20/20 of **unique timestamps** to train/validation/test, keeping tied timestamps together. Row proportions therefore differ. Require both classes in each partition.
3. Engineer only row-local features: V1–V28, log(1+Amount), sine/cosine of elapsed daily phase, and latent-vector magnitude. Daily phase is an assumed periodic signal, not verified local clock time. No invented customer histories are used.
4. Fit scaling and both estimators on training only. Logistic regression uses StandardScaler and max_iter=1000. Boosting uses 120 iterations, 15 leaves, learning rate .08 and L2=5. Random internal early stopping is disabled. There is no oversampling, class weighting or hyperparameter search.
5. Exactly sweep distinct validation scores, grouping ties and including flag-none. Minimize validation cost per model; choose the model by validation cost. Ties prefer fewer alerts. Freeze models and thresholds before evaluating test data; do not retrain on validation.
6. Report average precision (AP), precision, recall, confusion counts, alert count and cost, including default-threshold and flag-all/none comparisons. AP is the step-weighted precision–recall summary, not trapezoidal PR area.

`results.json` contains all metrics and assumptions. Prediction CSVs permit independent recomputation. Model bundles contain estimator, feature order, threshold and costs. Score validated new rows using `features(frame)` and the estimator's `predict_proba`, then compare to the saved threshold. Only load joblib files from trusted sources.

## Hypothetical financial model

All costs use a common currency unit: 2 per flagged transaction, an additional 5 for each flagged legitimate transaction, and 100% of Amount for each missed fraud. Thus cost = `2*(TP+FP) + 5*FP + sum(Amount for FN)`. Flagged fraud is assumed fully prevented. This simplifies recoveries, delays, imperfect interventions, capacity limits and customer lifetime effects.

```sh
python fraud_detection.py --review-cost 3 --false-positive-cost 10 --loss-rate 0.7
```

Thresholds use a global score rule; amount enters validation costs but does not directly determine the action threshold. Scores are not claimed to be calibrated probabilities. Validation optimization does not guarantee lower future cost.

## Limits

This pipeline's preprocessing/tuning cannot see future test rows. The public dataset's upstream PCA history is outside our control, so this cannot guarantee absence of leakage in source feature creation. Customer independence cannot be established without identifiers. The historical dataset is short and the test has only 77 frauds; a single holdout does not establish statistical superiority or production readiness. Real deployment needs longer out-of-time tests, delayed-label handling, uncertainty estimates, calibration, drift monitoring and a realistic review policy. Synthetic results demonstrate software behavior only. Do not repeatedly tune against the test set.

## Files

- `fraud_detection.py`: runnable acquisition-to-report pipeline.
- `tests/`: reproducibility, time boundaries, row-local features, exact threshold optimization and fallback checks.
- `.github/workflows/tests.yml`: automated tests and offline smoke run.
- `examples/public/`: real-data results, models, charts and acquisition record.
- `examples/synthetic/`: separately labeled offline demo.
- `CV.md`: measured CV description and interview context.
- `VERIFICATION.md`: actual checks performed.

Upload this folder's contents to GitHub. Model binaries and new result/data folders are ignored by default; example CSVs, charts and metrics are included intentionally. No credentials or paid service is needed.
