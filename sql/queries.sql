-- Run after real_fca_analysis.py, using output/real_fca/fca_quarterly.sqlite
SELECT quarter, COUNT(*) AS lender_observations, ROUND(AVG(equity_assets_pct),2) AS avg_net_worth_assets_pct FROM financial_metrics GROUP BY quarter ORDER BY quarter;
SELECT institution_name, quarter, ROUND(anomaly_score,4) AS anomaly_score FROM anomaly_screening WHERE anomaly_flag=1 ORDER BY anomaly_score;
