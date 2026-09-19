import unittest

from src.backtest import (
    moving_average_strategy,
    performance_metrics,
    run_backtest,
    sample_prices,
)


class BacktestTests(unittest.TestCase):

    def test_strategy_returns_binary_signals(self):
        prices = sample_prices(100)
        signals = moving_average_strategy(prices, 5, 20)

        self.assertTrue(set(signals.unique()).issubset({0, 1}))

    def test_no_lookahead(self):
        prices = sample_prices(100)
        signals = moving_average_strategy(prices, 5, 20)
        result = run_backtest(prices, signals)

        self.assertEqual(result["position"].iloc[0], 0)

    def test_metrics_are_returned(self):
        prices = sample_prices(100)
        signals = moving_average_strategy(prices, 5, 20)
        result = run_backtest(prices, signals)
        metrics = performance_metrics(result)

        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
