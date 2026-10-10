import pandas as pd
import pytest

from src.backtest.engine import (
    annualized_volatility, backtest_selected_portfolio, benchmark_returns,
    compare_gross_net_performance, compare_to_benchmark,
    equal_weight_portfolio_returns, cumulative_return, maximum_drawdown,
    sharpe_ratio, signal_persistence, strategy_returns,
)


def test_strategy_returns_lag_signal():
    data = pd.DataFrame({"ticker": ["AAA"] * 3, "close": [100.0, 110.0, 121.0], "signal": [0.0, 1.0, 1.0]})
    result = strategy_returns(data, "signal")
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.10)


def test_cumulative_return():
    assert cumulative_return(pd.Series([0.10, -0.10])) == pytest.approx(-0.01)


def test_maximum_drawdown():
    assert maximum_drawdown(pd.Series([0.10, -0.20, 0.05])) == pytest.approx(-0.20)


def test_zero_volatility_sharpe_is_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(returns) == 0.0
    assert annualized_volatility(returns) == 0.0


def test_signal_persistence():
    assert signal_persistence(pd.Series([1.0, 1.0, -1.0, -1.0])) == pytest.approx(2 / 3)


def test_equal_weight_portfolio_uses_lagged_selection():
    data = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2), "ticker": ["AAA", "AAA", "BBB", "BBB"], "close": [100.0, 110.0, 100.0, 90.0], "selected": [False, True, True, False]})
    result = equal_weight_portfolio_returns(data)
    assert result.loc[pd.Timestamp("2025-01-02")] == pytest.approx(0.0)


def test_backtest_selected_portfolio_returns_metrics():
    data = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"] * 2), "ticker": ["AAA"] * 3 + ["BBB"] * 3, "close": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0], "selected": [True, True, True, False, False, False]})
    metrics = backtest_selected_portfolio(data)
    assert set(metrics) == {"cumulative_return", "annualized_volatility", "sharpe_ratio", "maximum_drawdown"}
    assert metrics["cumulative_return"] == pytest.approx(0.21)


def test_benchmark_returns_are_equal_weighted():
    data = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2), "ticker": ["AAA", "AAA", "BBB", "BBB"], "close": [100.0, 110.0, 100.0, 90.0]})
    assert benchmark_returns(data).loc[pd.Timestamp("2025-01-02")] == pytest.approx(0.0)


def test_compare_to_benchmark_returns_excess_performance():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    result = compare_to_benchmark(pd.Series([0.10, 0.10], index=dates), pd.Series([0.05, 0.05], index=dates))
    assert result["portfolio_return"] == pytest.approx(0.21)
    assert result["benchmark_return"] == pytest.approx(0.1025)
    assert result["excess_return"] == pytest.approx(0.1175)


def test_gross_net_report_reflects_transaction_cost_drag():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    gross = pd.Series([0.01, 0.02, -0.005], index=dates)
    turnover = pd.Series([0.5, 1.0, 0.25], index=dates)
    result = compare_gross_net_performance(gross, turnover, cost_bps=10)
    assert result["net_cumulative_return"] < result["gross_cumulative_return"]
    assert result["transaction_cost_drag"] > 0
    assert result["net_sharpe_ratio"] != result["gross_sharpe_ratio"]


def test_gross_net_report_with_zero_costs_matches_gross_return():
    returns = pd.Series([0.01, -0.002, 0.005])
    turnover = pd.Series([0.5, 1.0, 0.25])
    result = compare_gross_net_performance(returns, turnover, cost_bps=0)
    assert result["net_cumulative_return"] == pytest.approx(result["gross_cumulative_return"])
    assert result["transaction_cost_drag"] == pytest.approx(0.0)
