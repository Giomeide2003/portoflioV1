import pandas as pd
import pytest

from src.backtest.engine import (
    annualized_volatility,
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
