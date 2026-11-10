"""Pipeline and validation checks against the four official quarterly archives."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_fca_analysis import RAW, ARCHIVES, load_quarter, build
from validate_models import run


class TestFarmCreditPipeline(unittest.TestCase):
    def test_quarterly_archives(self):
        for name in ARCHIVES:
            path = RAW / name
            self.assertTrue(path.exists(), f"Download required FCA archive: {name}")
            data = load_quarter(path)
            self.assertEqual(len(data), 64)
            self.assertFalse(data.UNINUM.duplicated().any())

    def test_screening_and_corrected_model(self):
        lending, scored = build()
        self.assertEqual(len(lending), 236)
        self.assertEqual(lending.institution_id.nunique(), 59)
        self.assertEqual(len(scored), 177)
        self.assertEqual(int(scored.anomaly_flag.sum()), 9)
        compared = run()
        self.assertEqual(len(compared), 177)
        self.assertEqual(int(compared.corrected_flag.sum()), 9)
        self.assertEqual(int((compared.corrected_flag & compared.original_flag).sum()), 8)
        self.assertEqual(int((compared.corrected_flag & compared.robust_top_n).sum()), 4)

    def test_income_ratio_math(self):
        result = run()
        row = result.iloc[0]
        target = 400 * row.quarterly_net_income / row.average_adjacent_quarter_assets
        self.assertAlmostEqual(row.annualized_quarterly_income_avg_assets_pct, target)


if __name__ == "__main__":
    unittest.main()
