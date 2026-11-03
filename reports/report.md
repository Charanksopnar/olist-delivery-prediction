Problem and business value: why late delivery matters (the review drop)
Data and why chosen
Cleaning decisions: delivered orders only, zero weight treated as missing, outliers kept
EDA insights: the 5 findings above, with 2 or 3 figures from reports/figures/
Method: time-based split, leakage rules, pipeline fitted on train only
Models and why: baseline first, then trees, and why Logistic won
Results: the table, the confusion matrix, and the 10% triage numbers
Drift test: hypothesis (purchase_month caused the Random Forest failure), result (rejected, because dropping it made models worse), which shows honest analysis
Drivers: the top coefficients or SHAP in plain words
Limitations and next steps: add seller history features, retrain monthly, test calibration