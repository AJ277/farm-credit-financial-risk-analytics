"""Reproducible FCA public call-report import and original model baseline.

The legacy income feature is intentionally retained to replicate the original
analysis. Run validate_models.py afterward for corrected, authoritative results.
"""
from pathlib import Path
import csv
import io
import re
import sqlite3
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "fca_raw"
OUT = ROOT / "output" / "real_fca"
ARCHIVES = ["2025December.zip", "2026March.zip", "2026June.zip", "2026September.zip"]
FIELDS = {
    "INST": ["UNINUM", "MONTH", "YEAR", "SHORTNAME", "STATE", "SYSTEM"],
    "RC": ["UNINUM", "MONTH", "YEAR", "ASSETS", "NETLOANS", "NETWRTH", "NONACR"],
    "RI": ["UNINUM", "MONTH", "YEAR", "NETY"],
}
FEATURES = ["loan_growth_qoq_pct", "equity_assets_pct",
            "nonaccrual_netloans_pct", "annualized_ytd_income_assets_pct"]


def load_schedule(archive, schedule):
    names = archive.namelist()
    data = [name for name in names if re.fullmatch(
        schedule + r"_Q\d{6}_G\d+\.TXT", Path(name).name, re.I)]
    layout = [name for name in names if Path(name).name.upper() == "D_" + schedule + ".TXT"]
    if len(data) != 1 or len(layout) != 1:
        raise ValueError(f"Ambiguous data/layout for {schedule}")
    columns = []
    for line in archive.read(layout[0]).decode("latin1").splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(?:Numeric|Alphanum\.)\s+0\b", line, re.I)
        if match:
            columns.append(match.group(1))
    if not set(FIELDS[schedule]).issubset(columns):
        raise ValueError(f"Missing {schedule} required fields")
    rows = list(csv.reader(io.StringIO(archive.read(data[0]).decode("latin1"))))
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"{schedule} record/layout width mismatch")
    frame = pd.DataFrame(rows, columns=columns)[FIELDS[schedule]].copy()
    if frame.UNINUM.duplicated().any():
        raise ValueError(f"Duplicate {schedule} institution IDs")
    for field in FIELDS[schedule]:
        if field not in ("SHORTNAME", "STATE"):
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame


def load_quarter(path):
    with zipfile.ZipFile(path) as archive:
        inst = load_schedule(archive, "INST")
        rc = load_schedule(archive, "RC")
        ri = load_schedule(archive, "RI")
    for table in (rc, ri):
        inst = inst.merge(table, on=["UNINUM", "MONTH", "YEAR"],
                          how="inner", validate="one_to_one")
    if len(inst) != len(rc) or len(inst) != len(ri):
        raise ValueError("Unmatched schedules: " + path.name)
    quarter = (inst.MONTH.astype(int) - 1) // 3 + 1
    inst["quarter"] = pd.PeriodIndex.from_fields(
        year=inst.YEAR.astype(int), quarter=quarter, freq="Q").astype(str)
    return inst


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ARCHIVES if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(f"Download source archives into {RAW}: {missing}")
    df = pd.concat([load_quarter(RAW / name) for name in ARCHIVES], ignore_index=True)
    if df.duplicated(["UNINUM", "quarter"]).any():
        raise ValueError("Duplicate institution-quarter keys")
    df = df.rename(columns={
        "UNINUM": "institution_id", "SHORTNAME": "institution_name",
        "ASSETS": "total_assets", "NETLOANS": "net_loans",
        "NETWRTH": "net_worth", "NONACR": "nonaccrual_loans",
        "NETY": "net_income_ytd",
    })
    numeric = ["total_assets", "net_loans", "net_worth",
               "nonaccrual_loans", "net_income_ytd"]
    if df[numeric].isna().any().any() or (df.total_assets <= 0).any():
        raise ValueError("Missing/invalid source fields")
    excluded = df[df.net_loans <= 0].copy()
    excluded.to_csv(OUT / "excluded_zero_loan_entities.csv", index=False)
    df = df[df.net_loans > 0].sort_values(["institution_id", "quarter"]).copy()
    prior = df.groupby("institution_id").net_loans.shift()
    prior_quarter = df.groupby("institution_id").quarter.shift()
    now = pd.PeriodIndex(df.quarter, freq="Q")
    prior_period = pd.PeriodIndex(prior_quarter.fillna(df.quarter), freq="Q")
    adjacent = prior_quarter.notna() & ((now.asi8 - prior_period.asi8) == 1)
    df["loan_growth_qoq_pct"] = np.where(adjacent, 100*(df.net_loans/prior-1), np.nan)
    df["equity_assets_pct"] = 100*df.net_worth/df.total_assets
    df["nonaccrual_netloans_pct"] = 100*df.nonaccrual_loans/df.net_loans
    # LEGACY / INCORRECT: this assumed quarterly RI.NETY was YTD.
    # Retained exclusively for pre-vs-post correction comparison.
    qnum = now.quarter
    df["annualized_ytd_income_assets_pct"] = 100*(df.net_income_ytd*4/qnum)/df.total_assets
    df.to_csv(OUT / "quarterly_metrics.csv", index=False)
    scored = df.dropna(subset=FEATURES).copy()
    x = StandardScaler().fit_transform(scored[FEATURES])
    model = IsolationForest(n_estimators=250, contamination=.05, random_state=42)
    model.fit(x)
    scored["anomaly_score"] = model.decision_function(x)
    scored["anomaly_flag"] = model.predict(x) == -1
    scored.to_csv(OUT / "anomaly_screening.csv", index=False)
    scored.sort_values("anomaly_score").head(10).to_csv(
        OUT / "lowest_anomaly_scores.csv", index=False)
    with sqlite3.connect(OUT / "fca_quarterly.sqlite") as db:
        df.to_sql("financial_metrics", db, if_exists="replace", index=False)
        scored.to_sql("anomaly_screening", db, if_exists="replace", index=False)
    print(f"Periods: {df.quarter.min()}–{df.quarter.max()} | "
          f"eligible lender rows: {len(df)} | excluded zero-loan rows: {len(excluded)} | "
          f"institutions: {df.institution_id.nunique()} | "
          f"scored: {len(scored)} | flagged: {int(scored.anomaly_flag.sum())}")
    return df, scored


if __name__ == "__main__":
    build()
