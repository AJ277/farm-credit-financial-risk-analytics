"""Validate FCA quarterly net-income definition and benchmark anomaly flags."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output" / "real_fca"
OUT = ROOT / "output" / "validation"
FEATURES = ["loan_growth_qoq_pct", "equity_assets_pct",
            "nonaccrual_netloans_pct", "annualized_quarterly_income_avg_assets_pct"]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE / "quarterly_metrics.csv")
    old = pd.read_csv(SOURCE / "anomaly_screening.csv")
    if df.duplicated(["institution_id", "quarter"]).any():
        raise ValueError("Duplicate institution-quarter keys")
    df = df.sort_values(["institution_id", "quarter"]).copy()
    previous_assets = df.groupby("institution_id").total_assets.shift()
    previous_quarter = df.groupby("institution_id").quarter.shift()
    now = pd.PeriodIndex(df.quarter, freq="Q")
    previous = pd.PeriodIndex(previous_quarter.fillna(df.quarter), freq="Q")
    adjacent = previous_quarter.notna() & ((now.asi8 - previous.asi8) == 1)
    # Legacy source label net_income_ytd is a MISNOMER: RI.NETY is quarterly.
    df["quarterly_net_income"] = df["net_income_ytd"]
    df["prior_quarter_assets"] = previous_assets
    df["average_adjacent_quarter_assets"] = np.where(
        adjacent, (df.total_assets + previous_assets)/2, np.nan)
    df["annualized_quarterly_income_ending_assets_pct"] = (
        400 * df.quarterly_net_income / df.total_assets)
    df["annualized_quarterly_income_avg_assets_pct"] = (
        400 * df.quarterly_net_income / df.average_adjacent_quarter_assets)
    df["prior_incorrect_income_assets_pct"] = df["annualized_ytd_income_assets_pct"]
    df.to_csv(OUT / "income_ratio_audit.csv", index=False)

    scored = df.dropna(subset=FEATURES).copy()
    if len(scored) != len(old):
        raise AssertionError(f"Original / corrected observation mismatch: {len(old)} vs {len(scored)}")
    x = StandardScaler().fit_transform(scored[FEATURES])
    model = IsolationForest(n_estimators=250, contamination=.05, random_state=42)
    model.fit(x)
    scored["corrected_score"] = model.decision_function(x)
    scored["corrected_flag"] = model.predict(x) == -1

    zcols = []
    for feature in FEATURES:
        name = "robust_z_" + feature
        zcols.append(name)
        median = scored.groupby("quarter")[feature].transform("median")
        mad = scored.groupby("quarter")[feature].transform(
            lambda v: np.median(np.abs(v - np.median(v))))
        iqr = scored.groupby("quarter")[feature].transform(
            lambda v: v.quantile(.75) - v.quantile(.25))
        scale = (1.4826*mad).where(mad > 1e-9, iqr/1.349)
        if scale.isna().any() or (scale <= 0).any():
            raise ValueError("Invalid robust dispersion for " + feature)
        scored[name] = (scored[feature]-median)/scale
    scored["robust_max_abs_z"] = scored[zcols].abs().max(axis=1)
    flags = old["anomaly_flag"].astype(str).str.lower().eq("true")
    n = int(flags.sum())
    scored["robust_rank"] = scored.robust_max_abs_z.rank(
        method="first", ascending=False)
    scored["robust_top_n"] = scored.robust_rank <= n
    original = old[["institution_id", "quarter", "anomaly_score", "anomaly_flag"]].rename(
        columns={"anomaly_score": "original_score", "anomaly_flag": "original_flag"})
    scored = scored.merge(original, on=["institution_id", "quarter"], validate="one_to_one")
    scored["original_flag"] = scored.original_flag.astype(str).str.lower().eq("true")
    scored["flags_changed"] = scored.original_flag != scored.corrected_flag
    scored.sort_values("corrected_score").to_csv(OUT/"model_comparison.csv",index=False)
    scored.loc[scored.original_flag | scored.corrected_flag | scored.robust_top_n].sort_values(
        "corrected_score").to_csv(OUT/"review_candidates.csv",index=False)

    oldset = set(map(tuple, scored.loc[scored.original_flag,
                                     ["institution_id","quarter"]].to_numpy()))
    newset = set(map(tuple, scored.loc[scored.corrected_flag,
                                     ["institution_id","quarter"]].to_numpy()))
    benchmark = set(map(tuple, scored.loc[scored.robust_top_n,
                                        ["institution_id","quarter"]].to_numpy()))
    changed = scored[scored.flags_changed]
    rows = ["| Institution | Quarter | Before | After |",
            "| --- | --- | --- | --- |"]
    for r in changed.itertuples():
        rows.append(f"| {r.institution_name} | {r.quarter} | "
                    f"{'Flagged' if r.original_flag else 'Not flagged'} | "
                    f"{'Flagged' if r.corrected_flag else 'Not flagged'} |")
    report = (
        "# Income-ratio correction and anomaly validation\n\n"
        "The original model incorrectly treated quarterly RI.NETY as year-to-date "
        "income. Corrected annualization uses four times quarterly net income "
        "divided by adjacent-quarter average total assets. This is a screening "
        "proxy, not FCA's official return-on-assets formula.\n\n"
        f"- Scored observations: {len(scored)}\n"
        f"- Original flags: {len(oldset)}; corrected flags: {len(newset)}\n"
        f"- Flags retained: {len(oldset & newset)}\n"
        f"- Corrected/robust-z top-{n} overlap: {len(newset & benchmark)}\n\n"
        "## Changed classifications\n\n" + "\n".join(rows) +
        "\n\nStatistical anomalies are not established credit or supervisory risks. "
        "The 5% contamination parameter influences flag count, and there are "
        "no known outcomes for predictive accuracy evaluation.\n"
    )
    (OUT/"VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Scored: {len(scored)} | old flags: {len(oldset)} | "
          f"corrected flags: {len(newset)} | shared: {len(oldset & newset)} | "
          f"robust-z agreement: {len(newset & benchmark)}")
    return scored


if __name__ == "__main__":
    run()
