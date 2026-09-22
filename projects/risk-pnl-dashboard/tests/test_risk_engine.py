import unittest

import pandas as pd

from src.risk_engine import (
    Position,
    historical_var,
    parametric_var,
    portfolio_pnl,
    portfolio_returns,
    stress_test,
)


class RiskEngineTests(unittest.TestCase):

    def setUp(self):
        self.positions = [
            Position("A", 100, 100.0),
            Position("B", 50, 200.0),
        ]
        self.prices = {"A": 105.0, "B": 190.0}

    def test_pnl(self):
        result = portfolio_pnl(self.positions, self.prices)
        self.assertEqual(result["pnl"].sum(), 0.0)

    def test_portfolio_returns(self):
        returns = pd.DataFrame(
            {
                "A": [0.01, -0.02, 0.005],
                "B": [0.02, -0.01, 0.0],
            }
        )
        result = portfolio_returns(returns, self.positions, self.prices)

        self.assertEqual(len(result), 3)
        self.assertTrue(result.notna().all())

    def test_historical_var_is_positive_for_losses(self):
        returns = pd.Series([0.02, 0.01, -0.03, -0.04, 0.01])
        value = historical_var(returns, 0.95, 100000)

        self.assertGreater(value, 0)

    def test_parametric_var_is_non_negative(self):
        returns = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02])
        value = parametric_var(returns, 0.95, 100000)

        self.assertGreaterEqual(value, 0)

    def test_stress_test(self):
        result = stress_test(
            self.positions,
            self.prices,
            {"A": -0.10, "B": 0.05},
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc[0, "stressed_price"], 94.5)
        self.assertAlmostEqual(result.loc[1, "stressed_price"], 199.5)


if __name__ == "__main__":
    unittest.main()
