import pandas as pd
import pytest

from src.backtest.engine import (
    annualized_volatility,
    backtest_selected_portfolio,
    equal_weight_portfolio_returns,
    cumulative_return,
    maximum_drawdown,
    sharpe_ratio,
    signal_persistence,
    strategy_returns,
)


def test_strategy_returns_lag_signal():
    data = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "close": [100.0, 110.0, 121.0],
            "signal": [0.0, 1.0, 1.0],
        }
    )

    result = strategy_returns(data, "signal")

    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.10)


def test_cumulative_return():
    returns = pd.Series([0.10, -0.10])

    assert cumulative_return(returns) == pytest.approx(-0.01)


def test_maximum_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])

    assert maximum_drawdown(returns) == pytest.approx(-0.20)


def test_zero_volatility_sharpe_is_zero():
    returns = pd.Series([0.0, 0.0, 0.0])

    assert sharpe_ratio(returns) == 0.0
    assert annualized_volatility(returns) == 0.0


def test_signal_persistence():
    signal = pd.Series([1.0, 1.0, -1.0, -1.0])

    assert signal_persistence(signal) == pytest.approx(2 / 3)


def test_equal_weight_portfolio_uses_lagged_selection():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "close": [100.0, 110.0, 100.0, 90.0],
            "selected": [False, True, True, False],
        }
    )

    result = equal_weight_portfolio_returns(data)

    assert result.loc[pd.Timestamp("2025-01-02")] == pytest.approx(0.0)


def test_backtest_selected_portfolio_returns_metrics():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03"] * 2
            ),
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "close": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0],
            "selected": [True, True, True, False, False, False],
        }
    )

    metrics = backtest_selected_portfolio(data)

    assert set(metrics) == {
        "cumulative_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    }
    assert metrics["cumulative_return"] == pytest.approx(0.21)
