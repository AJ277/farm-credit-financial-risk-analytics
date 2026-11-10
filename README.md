# Farm Credit System Financial Risk Analytics

**Python · SQL/SQLite · pandas · scikit-learn · financial ratio validation · unsupervised anomaly screening**

Independent portfolio project (November 2026). **Not affiliated with or endorsed by the Farm Credit Administration (FCA).**

## Executive summary

Analyzes four public FCA Uniform Call Report quarters (2025 Q4–2026 Q3). The pipeline imports **256 institution-quarter records**, excludes **20 zero-net-loan observations** from loan screening, and retains **236 observations across 59 lending institutions**. Isolation Forest scores **177 comparable observations** and flags **9**. Correcting a quarterly income-ratio calculation changes **two classifications**, with eight original flags retained. A median/MAD statistical benchmark overlaps with **four of the nine** corrected flags.

**Flags are statistical outliers—not evidence of financial distress, examination findings, or regulatory violations.** The model has no labeled outcomes to estimate predictive accuracy. Its 5% contamination setting influences the number of flags.

## Research question

Can public FCA financial reporting be used to screen for unusual institution-level financial patterns, and how sensitive are anomaly classifications to the underlying ratio definitions?

## Data

Source: [FCA Uniform Call Report Data Downloads](https://www.fca.gov/bank-oversight/call-report-data-for-download).

Quarters: December 2025; March, June and September 2026. The ZIPs contain reporting schedules and data layout dictionaries used for field mapping.

| Reported field | Use |
| --- | --- |
| `INST.UNINUM`, `INST.SHORTNAME` | Institution joins |
| `RC.ASSETS` | Total assets |
| `RC.NETLOANS` | Net loans / loan growth |
| `RC.NETWRTH` | Net worth-to-assets proxy |
| `RC.NONACR` | Nonaccrual-to-net-loans proxy |
| `RI.NETY` | **Quarterly** net income |

## Method

1. Extract quarterly archives, read embedded layouts, join schedules and check uniqueness.
2. Preserve 20 zero-net-loan records for audit; exclude them from loan-quality screening.
3. Calculate quarter-over-quarter loan growth and balance-sheet ratios.
4. Correct the income ratio to use quarterly income: `100 * 4 * quarterly_net_income / average(previous_period_assets, current_period_assets)`. This uses an adjacent-quarter average-assets approximation and **is not FCA's official ROA**.
5. Standardize four screening features; fit `IsolationForest(n_estimators=250, contamination=0.05, random_state=42)` to 177 observations.
6. Compare uncorrected and corrected rankings, plus a within-quarter median/MAD robust-z baseline (IQR fallback).

**Data-quality correction:** The legacy importer incorrectly treated Schedule RI quarterly net income as year-to-date. The original derived value `100 * NETY * (4 / quarter_number) / ending_assets` is **not valid for comparing quarters**. The authoritative model for this project is produced by `validate_models.py`.

## Results

| Metric | Result |
| --- | ---: |
| Imported institution-quarter records | 256 |
| Lending institution-quarter records | 236 |
| Lending institutions | 59 |
| Comparable observations scored | 177 |
| Original model flags | 9 |
| Corrected model flags | 9 |
| Flags retained | 8 |
| Corrected flags overlapping robust-z benchmark | 4 |

**Newly flagged:** FCB of Texas, 2026 Q2. **No longer flagged:** Farm Credit East, ACA, 2026 Q1. This is sensitivity to financial feature engineering, **not** a change in those institutions' condition.

## Reproduce locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe real_fca_analysis.py
.\.venv\Scripts\python.exe validate_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The source code, raw public data archives, figures, and full report are part of the prepared upload package. Generated outputs appear in `output/validation/`. The legacy `real_fca_analysis.py` creates the **uncorrected** baseline and should not be interpreted on its own. `analyze.py` is an independent synthetic demonstration.

## Project layout

- `real_fca_analysis.py`: public call-report importer / original audit baseline
- `validate_models.py`: corrected ratios, Isolation Forest rerun, robust-z benchmarking
- `analyze.py`: separate synthetic demo
- `tests/`: reproducibility and integrity tests
- `sql/`: SQL examples
- `data/fca_raw/`: original public FCA quarterly archives
- `figures/`: income-ratio and classification comparison charts
- `docs/`: complete technical report
- `output/`: generated local analysis, excluded from version control

## Limitations

Four consecutive quarters allow quarter-over-quarter comparisons but not year-over-year growth. Financial ratios are analytical proxies rather than supervisory measures. Institutions differ in size and business structure; reporting revisions or transfers may affect trends. The model fits and scores the same historical sample, and flags are not validated against future financial outcomes. Human review and source verification are required before drawing substantive conclusions.

## References

1. [FCA Call Report Data](https://www.fca.gov/bank-oversight/call-report-data-for-download)
2. [FCA Institution Data](https://reports.fca.gov/CRS/InstitutionDataIntro.aspx)
3. [FCA Uniform Call Report Forms and Instructions](https://ww3.fca.gov/fcsinfo/crs/callreportfiles/forms/allitems.aspx)
4. [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

Public FCA data are used for independent educational analysis. All analysis and interpretations are the project author's own.
