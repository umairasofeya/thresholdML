# Threshold ML

**Cost-sensitive credit card fraud detection, evaluated the way a deployment would actually face it.**
Models train on the past, choose a decision threshold on a later period by minimising an explicit cost model, freeze that rule, and only then see the most recent period. Every reported number comes with a bootstrap interval.

![Evaluation on the public dataset](examples/public/evaluation.png)

## Results on 284,807 real transactions

Chronological test set: the final 60,140 transactions, containing 77 frauds. Thresholds were frozen on validation before any test data was scored. 95% stratified bootstrap intervals are shown in brackets.

| Model | Average precision | Precision | Recall | Hypothetical cost |
|---|---:|---:|---:|---:|
| Logistic regression | 0.744 [0.647, 0.827] | 24.4% [21.5, 27.7] | 85.7% [76.6, 92.2] | 3,922 [1,913, 6,948] |
| **Gradient boosting** (selected) | 0.788 [0.698, 0.869] | 86.4% [78.3, 93.8] | 74.0% [63.6, 83.1] | 3,851 [1,145, 7,621] |
| Flag nothing | — | — | 0% | 8,397 |

**What this shows:**

- **Cost-aware thresholds halve losses** compared with flagging nothing: 3,851 vs 8,397. The selected rule raised 66 alerts (57 frauds caught, 9 false alarms, 20 missed).
- **The two models are not distinguishable on cost.** Their paired cost difference is −71 [−1,376, +2,224], and boosting is cheaper in only 59% of paired resamples. They reach similar costs very differently: logistic regression flags 4× more transactions to catch more fraud. With 77 test frauds, the ranking is not statistically settled, and the bottom-right panel above shows why.
- **Threshold choice matters more than model choice for logistic regression.** Its validation-chosen threshold (0.026) beats the default 0.5 by 774 cost units. For boosting, 0.5 happened to do slightly better on test (3,837 vs 3,851). That gap is well inside the noise, and the threshold was not retuned on test.

Full numbers are in [`examples/public/REPORT.md`](examples/public/REPORT.md) and [`results.json`](examples/public/results.json). The dataset hash is recorded in [`ACQUISITION.md`](examples/public/ACQUISITION.md).

## Quick start

```sh
make install         # venv + locked dependencies + editable package
make test            # 36 tests, a few seconds
make demo            # offline synthetic run → results/synthetic/
make reproduce       # download + hash-check the public data, regenerate examples/
```

Or directly:

```sh
pip install -e ".[dev]"
threshold-ml run --source csv --csv data/creditcard.csv --output results/public
threshold-ml run --review-cost 3 --false-positive-cost 10 --loss-rate 0.7   # alternative cost assumptions
threshold-ml score --model results/public/gradient_boosting.joblib --input new.csv --output scored.csv
```

`--source` accepts `synthetic` (the offline default), `csv`, `download` (fails loudly) or `auto`. `auto` falls back to synthetic data and records the fallback reason in the provenance block.

## Method

| Stage | Choice | Why |
|---|---|---|
| Split | 60/20/20 over **unique timestamps**, in time order | No future rows in training; tied timestamps never straddle a boundary |
| Features | V1–V28, `log1p(Amount)`, sin/cos of daily phase, latent-vector norm | Strictly row-local, so no label or cross-row leakage is possible |
| Models | Standardised logistic regression; histogram gradient boosting (120 trees, 15 leaves, lr 0.08, L2 5) | Fixed in advance, with no hyperparameter search to overfit validation |
| Imbalance | No resampling or class weights | The cost-optimal threshold absorbs the imbalance directly |
| Threshold | Exact O(n log n) sweep over every distinct validation score, including flag-none; ties favour fewer alerts | Gives the true cost minimum, not a grid approximation ([`costs.py`](src/threshold_ml/costs.py)) |
| Selection | Lowest validation cost, then fewest alerts | Test is never used for any choice |
| Uncertainty | 1,000 stratified bootstrap resamples of test, models evaluated on the same resamples | Paired comparisons cancel shared noise; stratifying keeps the count of 77 frauds fixed |
| Calibration | Brier score and ECE alongside the reliability curve | Shows whether scores can be read as probabilities |

### Cost model

All costs are hypothetical, in one currency unit:

```
cost = review_cost × alerts + false_positive_cost × FP + loss_rate × Σ amount(missed frauds)
```

The defaults are 2, 5 and 1.0. A flagged fraud is assumed fully prevented. Because missed-fraud cost scales with the amount, the optimiser trades review workload against dollar exposure rather than raw counts.

## Verification

The test suite in [`tests/`](tests) checks:

- the threshold optimiser against brute force over random tie-heavy inputs and random cost parameters
- strict temporal boundaries, with ties forced together
- that features are invariant to labels and to other rows
- schema validation for every failure mode
- that stratified resampling preserves class counts, and that identical models give zero paired difference
- that a near-perfectly calibrated simulation yields ECE ≈ 0
- end to end: metrics recomputed independently from the published prediction CSVs, and saved bundles that re-score unlabelled rows to identical outputs

CI runs lint, format checks, the tests and a smoke run on Python 3.11–3.13.

## Limitations

- **One short, anonymised dataset.** It covers two days, and V1–V28 are upstream PCA outputs whose fitting we can't audit for leakage.
- **No customer identifiers.** Per-card behavioural features, usually the strongest fraud signal, aren't possible.
- **The intervals cover test sampling noise only.** They do not cover training or threshold-selection variability; a rolling-origin backtest would be the next step.
- **The cost model is deliberately simple.** It ignores partial recoveries, review capacity limits, label delay and customer lifetime effects.
- **Synthetic runs exercise the software, not the science.**

## Layout

```
src/threshold_ml/
  data.py         schema validation, synthetic generator, provenance-tracked loading
  splits.py       chronological partitioning
  features.py     row-local feature engineering
  costs.py        CostModel and exact threshold optimiser
  models.py       fixed candidate estimators
  evaluation.py   metrics, calibration, paired stratified bootstrap
  pipeline.py     two-phase experiment (fit/select → freeze → evaluate)
  scoring.py      self-describing ModelBundle for inference
  plots.py, report.py, cli.py
examples/public/     real-data run: report, metrics, figure, predictions, threshold curves
examples/synthetic/  offline demo run
```

Model bundles (`*.joblib`) are regenerated by `make reproduce` and are not committed. Only load joblib files you trust.
