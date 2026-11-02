# Threshold ML run — `synthetic` data

20,000 transactions; chronological split train/validation/test = 11,989 (176 frauds) / 4,001 (63 frauds) / 4,010 (92 frauds).
Selected on cross-fitted validation cost (5 contiguous time blocks): **gradient_boosting / amount_aware_isotonic**.

## Ranking quality (95% bootstrap intervals)

| Model | Validation AP | Test AP |
|---|---:|---:|
| logistic_regression | 0.099 | 0.232 [0.162, 0.330] |
| gradient_boosting | 0.076 | 0.112 [0.086, 0.152] |

## Decision policies

Validation cost is cross-fitted (each block decided by a policy fitted on the other blocks); in-sample cost is shown for reference and is optimistic.

| Model | Policy | Rule | Validation cost (cross-fitted / in-sample) | Test alerts | TP | FP | FN | Test cost (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | global_threshold | score ≥ 0.06763 | 4,673 / 4,085 | 236 | 42 | 194 | 50 | 4,496 [3,472, 5,872] |
| logistic_regression | default_0_5 | score ≥ 0.5 | 4,554 / 4,554 | 2 | 1 | 1 | 91 | 5,435 [4,224, 6,827] |
| logistic_regression | amount_aware_platt | expected cost, platt calibration | 3,604 / 3,418 | 152 | 28 | 124 | 64 | 3,454 [2,779, 4,200] |
| logistic_regression | amount_aware_isotonic | expected cost, isotonic calibration | 3,626 / 3,087 | 154 | 25 | 129 | 67 | 4,007 [3,099, 5,032] |
| gradient_boosting | global_threshold | score ≥ 0.04965 | 4,678 / 4,106 | 242 | 35 | 207 | 57 | 4,882 [3,855, 6,212] |
| gradient_boosting | default_0_5 | score ≥ 0.5 | 4,547 / 4,547 | 1 | 0 | 1 | 92 | 5,490 [4,296, 6,884] |
| gradient_boosting | amount_aware_platt | expected cost, platt calibration | 3,548 / 3,758 | 135 | 11 | 124 | 81 | 4,869 [3,880, 6,083] |
| gradient_boosting | amount_aware_isotonic **(selected)** | expected cost, isotonic calibration | 3,456 / 3,339 | 158 | 14 | 144 | 78 | 4,699 [3,736, 5,866] |

Baselines on test: flag none 5,482.65; flag all 27,610.00.

## Paired comparisons on test

| Candidate | Baseline | Cost difference | Candidate cheaper in |
|---|---|---:|---:|
| logistic_regression/default_0_5 | logistic_regression/global_threshold | 939 [74, 1,862] | 1% |
| logistic_regression/amount_aware_platt | logistic_regression/global_threshold | -1,042 [-2,309, -185] | 100% |
| logistic_regression/amount_aware_isotonic | logistic_regression/global_threshold | -489 [-1,381, 49] | 95% |
| gradient_boosting/default_0_5 | gradient_boosting/global_threshold | 608 [-261, 1,543] | 10% |
| gradient_boosting/amount_aware_platt | gradient_boosting/global_threshold | -13 [-541, 465] | 52% |
| gradient_boosting/amount_aware_isotonic | gradient_boosting/global_threshold | -183 [-804, 339] | 72% |
| gradient_boosting/global_threshold | logistic_regression/global_threshold | 386 [-15, 833] | 3% |
| gradient_boosting/default_0_5 | logistic_regression/default_0_5 | 54 [-14, 191] | 11% |
| gradient_boosting/amount_aware_platt | logistic_regression/amount_aware_platt | 1,415 [536, 2,562] | 0% |
| gradient_boosting/amount_aware_isotonic | logistic_regression/amount_aware_isotonic | 692 [-164, 1,745] | 6% |

## Calibration (test)

| Model | Probabilities | Brier score | ECE |
|---|---|---:|---:|
| logistic_regression | raw | 1.99e-02 | 5.39e-03 |
| logistic_regression | platt | 2.00e-02 | 3.68e-03 |
| logistic_regression | isotonic | 2.04e-02 | 3.34e-03 |
| gradient_boosting | raw | 2.18e-02 | 1.29e-02 |
| gradient_boosting | platt | 2.14e-02 | 5.96e-03 |
| gradient_boosting | isotonic | 2.14e-02 | 5.32e-03 |

Synthetic data demonstrates software behaviour only.
Costs are hypothetical, not realised savings.

![Evaluation](evaluation.png)
