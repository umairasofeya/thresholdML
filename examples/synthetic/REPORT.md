# Threshold ML run — `synthetic` data

20,000 transactions; chronological split train/validation/test = 11,989 (176 frauds) / 4,001 (63 frauds) / 4,010 (92 frauds).
Selected on validation cost: **logistic_regression**.

## Test metrics at frozen validation thresholds (95% bootstrap intervals)

| Model | Threshold | Average precision | Precision | Recall | Hypothetical cost |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.06763 | 0.232 [0.162, 0.330] | 0.178 [0.140, 0.216] | 0.457 [0.359, 0.555] | 4,496 [3,472, 5,872] |
| gradient_boosting | 0.04965 | 0.112 [0.086, 0.152] | 0.145 [0.112, 0.180] | 0.380 [0.293, 0.478] | 4,882 [3,855, 6,212] |

## Decision outcomes on test

| Model | Rule | Alerts | TP | FP | FN | Cost |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | validation threshold | 236 | 42 | 194 | 50 | 4,495.78 |
| logistic_regression | default 0.5 | 2 | 1 | 1 | 91 | 5,435.23 |
| gradient_boosting | validation threshold | 242 | 35 | 207 | 57 | 4,882.02 |
| gradient_boosting | default 0.5 | 1 | 0 | 1 | 92 | 5,489.65 |
| — | flag none | 0 | 0 | 0 | — | 5,482.65 |
| — | flag all | all | — | — | 0 | 27,610.00 |

## Model comparison

- `gradient_boosting_minus_logistic_regression` test cost: 386 [-15, 833]. gradient_boosting is cheaper in 3% of paired resamples.

## Calibration (test)

| Model | Brier score | Expected calibration error |
|---|---:|---:|
| logistic_regression | 1.99e-02 | 5.39e-03 |
| gradient_boosting | 2.18e-02 | 1.29e-02 |

Synthetic data demonstrates software behaviour only.
Costs are hypothetical, not realised savings.

![Evaluation](evaluation.png)
