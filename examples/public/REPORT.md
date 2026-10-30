# Measured run: csv

Rows: 284,807. Selected using validation cost: **gradient_boosting**.

| Model | Test AP | Precision | Recall | Hypothetical test cost |
|---|---:|---:|---:|---:|
| logistic_regression | 0.7437 | 0.2444 | 0.8571 | 3922.20 |
| gradient_boosting | 0.7878 | 0.8636 | 0.7403 | 3851.44 |

Measured on a single chronological dataset holdout, not a production performance guarantee.
Costs are hypothetical, not realized savings.

![Evaluation](evaluation.png)
