# Threshold ML run — `csv` data

284,807 transactions; chronological split train/validation/test = 162,992 (356 frauds) / 61,675 (59 frauds) / 60,140 (77 frauds).
Selected on cross-fitted validation cost (5 contiguous time blocks): **logistic_regression / amount_aware_platt**.

## Ranking quality (95% bootstrap intervals)

| Model | Validation AP | Test AP |
|---|---:|---:|
| logistic_regression | 0.659 | 0.744 [0.647, 0.827] |
| gradient_boosting | 0.750 | 0.788 [0.698, 0.869] |

## Decision policies

Validation cost is cross-fitted (each block decided by a policy fitted on the other blocks); in-sample cost is shown for reference and is optimistic.

| Model | Policy | Rule | Validation cost (cross-fitted / in-sample) | Test alerts | TP | FP | FN | Test cost (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | global_threshold | score ≥ 0.02574 | 5,437 / 4,814 | 270 | 66 | 204 | 11 | 3,922 [1,913, 6,948] |
| logistic_regression | default_0_5 | score ≥ 0.5 | 5,856 / 5,856 | 65 | 48 | 17 | 29 | 4,696 [1,874, 8,407] |
| logistic_regression | amount_aware_platt **(selected)** | expected cost, platt calibration | 4,247 / 4,240 | 89 | 26 | 63 | 51 | 3,041 [987, 6,081] |
| logistic_regression | amount_aware_isotonic | expected cost, isotonic calibration | 4,296 / 4,170 | 92 | 26 | 66 | 51 | 2,993 [980, 5,971] |
| gradient_boosting | global_threshold | score ≥ 0.3892 | 4,950 / 4,663 | 66 | 57 | 9 | 20 | 3,851 [1,145, 7,621] |
| gradient_boosting | default_0_5 | score ≥ 0.5 | 4,957 / 4,957 | 64 | 57 | 7 | 20 | 3,837 [1,115, 7,594] |
| gradient_boosting | amount_aware_platt | expected cost, platt calibration | 4,807 / 4,870 | 67 | 28 | 39 | 49 | 2,991 [913, 6,038] |
| gradient_boosting | amount_aware_isotonic | expected cost, isotonic calibration | 5,150 / 4,856 | 54 | 27 | 27 | 50 | 2,945 [853, 5,982] |

Baselines on test: flag none 8,396.81; flag all 420,595.00.

## Paired comparisons on test

| Candidate | Baseline | Cost difference | Candidate cheaper in |
|---|---|---:|---:|
| logistic_regression/default_0_5 | logistic_regression/global_threshold | 774 [-854, 3,172] | 25% |
| logistic_regression/amount_aware_platt | logistic_regression/global_threshold | -882 [-1,122, -615] | 100% |
| logistic_regression/amount_aware_isotonic | logistic_regression/global_threshold | -929 [-1,112, -737] | 100% |
| gradient_boosting/default_0_5 | gradient_boosting/global_threshold | -14 [-35, 0] | 87% |
| gradient_boosting/amount_aware_platt | gradient_boosting/global_threshold | -860 [-2,910, 219] | 68% |
| gradient_boosting/amount_aware_isotonic | gradient_boosting/global_threshold | -906 [-2,944, 141] | 68% |
| gradient_boosting/global_threshold | logistic_regression/global_threshold | -71 [-1,376, 2,224] | 59% |
| gradient_boosting/default_0_5 | logistic_regression/default_0_5 | -858 [-1,705, -171] | 100% |
| gradient_boosting/amount_aware_platt | logistic_regression/amount_aware_platt | -49 [-435, 510] | 65% |
| gradient_boosting/amount_aware_isotonic | logistic_regression/amount_aware_isotonic | -48 [-370, 497] | 68% |

## Calibration (test)

| Model | Probabilities | Brier score | ECE |
|---|---|---:|---:|
| logistic_regression | raw | 5.61e-04 | 1.14e-03 |
| logistic_regression | platt | 5.58e-04 | 3.64e-04 |
| logistic_regression | isotonic | 5.71e-04 | 2.50e-04 |
| gradient_boosting | raw | 4.09e-04 | 1.70e-04 |
| gradient_boosting | platt | 4.03e-04 | 2.26e-04 |
| gradient_boosting | isotonic | 3.95e-04 | 1.86e-04 |

A single chronological holdout; intervals capture test sampling noise, not training variability.
Costs are hypothetical, not realised savings.

![Evaluation](evaluation.png)
