# Technical summary: FCA Financial Risk Analytics

**Independent educational project — November 2026**

## Executive summary
Four public FCA Uniform Call Report quarters (2025 Q4 through 2026 Q3) were parsed into 256 institution-quarter observations. Twenty observations from organizations with no net-loan portfolio were excluded from lending-ratio screening, leaving 236 observations across 59 institutions. A prior consecutive quarter was required for growth and average-assets features; 177 observations were scored.

A preliminary Isolation Forest with 250 trees, 5% contamination, and a fixed random seed flagged nine observations. The initial income-to-assets feature incorrectly used a year-to-date annualization convention for a quarterly income variable. Correcting this feature and rerunning the model resulted in nine flags, of which eight overlapped with the initial classifications. A quarter-relative median/MAD robust-z benchmark overlapped with four of the nine corrected model flags.

## Source and field definitions
The original archives can be downloaded at https://www.fca.gov/bank-oversight/call-report-data-for-download.

- `INST.UNINUM` and `INST.SHORTNAME` identify institutions.
- `RC.ASSETS` = total assets; `RC.NETLOANS` = net loans; `RC.NETWRTH` = net worth; `RC.NONACR` = nonaccrual loans.
- `RI.NETY` = quarterly net income, **not year-to-date income**.

## Ratios and methods
- Quarter-over-quarter net-loan growth: `100 * (current_loans / previous_loans - 1)` for consecutive quarters.
- Net worth/assets proxy: `100 * net_worth / total_assets`.
- Nonaccrual/net-loans proxy: `100 * nonaccrual_loans / net_loans`.
- Corrected annualized quarterly income/adjacent-quarter average-assets proxy: `400 * quarterly_income / mean(previous_assets, current_assets)`.

The original model's ratio incorrectly used `100 * quarterly_income * (4 / calendar_quarter_number) / ending_assets`. The corrected model uses the four features above, standardizes them, and runs Isolation Forest. The independent benchmark ranks observations based on their largest absolute quarter-relative robust z-score using median/MAD (IQR fallback).

## Model comparison
| Metric | Value |
|---|---:|
| Original anomalies | 9 |
| Corrected anomalies | 9 |
| Unchanged anomalies | 8 |
| Robust benchmark top-nine overlap | 4 |

The corrected model newly flagged FCB of Texas (2026 Q2) and no longer flagged Farm Credit East, ACA (2026 Q1). These are **statistical classification changes only**, not changes in real-world institution conditions.

## Limitations and recommendations
Unsupervised anomaly detection has no ground-truth labels here, so no precision, recall, default probability, or predictive accuracy is claimed. The 5% contamination parameter influences the number of flags. Four quarters permit quarter-over-quarter but not year-over-year loan comparisons. Ratios are simplified analytical proxies, not FCA supervisory standards. Differences in institution type, portfolio composition, mergers and reporting revisions can change comparability. Any flags should be reviewed against primary institution reports and broader financial context.

## Reproduce
Place the four downloaded ZIP files in `data/fca_raw/`, and run:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe real_fca_analysis.py
.\.venv\Scripts\python.exe validate_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## References
1. FCA, Call Report Data for Download: https://www.fca.gov/bank-oversight/call-report-data-for-download
2. FCA, Uniform Call Report Forms and Instructions: https://ww3.fca.gov/fcsinfo/crs/callreportfiles/forms/allitems.aspx
3. scikit-learn, IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

The full project bundle separately prepared for download includes the original public archives and graphics. The repository hosts text source, scripts, documentation, and tests. No confidential FCA information is included.
