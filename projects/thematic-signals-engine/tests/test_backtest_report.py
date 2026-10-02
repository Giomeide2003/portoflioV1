import pandas as pd
import pytest

from src.backtest.report import equity_curve, performance_summary


def test_equity_curve_is_compounded():
    returns = pd.Series([0.10, -0.10])

    result = equity_curve(returns)

    assert result.iloc[-1] == pytest.approx(0.99)


def test_performance_summary_contains_risk_metrics():
    portfolio = pd.Series([0.01, 0.02, -0.01])
    benchmark = pd.Series([0.005, 0.01, 0.0])

    result = performance_summary(portfolio, benchmark)

    assert "Cumulative Return" in result.index
    assert "Sharpe Ratio" in result.index
    assert "Maximum Drawdown" in result.index
    assert "Benchmark Return" in result.index
    assert "Excess Return" in result.index
