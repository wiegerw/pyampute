"""Unittest for mcar_tests.py"""
import unittest
import pandas as pd
from pathlib import Path

from pyampute.exploration.mcar_statistical_tests import MCARTest


# load test data
csv_folder = Path(__file__).parent.parent / "data"
try:
    data_mar = pd.read_table(csv_folder / "missingdata.csv")
    data_mcar = pd.read_table(csv_folder / "missingdata_mcar.csv")
except:
    print(f"Failed to load CSV files from folder '{csv_folder}'.")

significance_level = 0.05


class TestMCARTest(unittest.TestCase):
    """Test for MCAR."""

    def test_little_mcar_test(self):
        self.assertFalse(MCARTest(method="little")(data_mcar) < significance_level)
        self.assertTrue(MCARTest(method="little")(data_mar) < significance_level)

    def test_mcar_t_tests(self):
        # Axis=None reduces in all dimensions
        # should fail to reject sometimes, since MCAR
        self.assertTrue(
            (MCARTest(method="ttest")(data_mcar) > significance_level).any(axis=None)
        )
        # reject all: missingness is not MCAR for any pair of vars
        self.assertFalse(
            (MCARTest(method="ttest")(data_mar) < significance_level).all(axis=None)
        )


if __name__ == "__main__":
    unittest.main()
