import pandas as pd
import pytest

from src.backtest.trading_costs import net_returns_after_costs, portfolio_turnover


def test_turnover_measures_weight_changes_and_initial_investment():
    weights = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]),
        "ticker": ["AAA", "BBB", "AAA", "CCC"],
        "weight": [0.5, 0.5, 0.0, 1.0],
    })
    result = portfolio_turnover(weights)
    assert result.iloc[0] == pytest.approx(0.5)
    assert result.iloc[1] == pytest.approx(1.0)


def test_net_returns_deduct_turnover_cost():
    index = pd.to_datetime(["2025-01-01", "2025-01-02"])
    gross = pd.Series([0.01, 0.02], index=index)
    turnover = pd.Series([0.5, 1.0], index=index)
    result = net_returns_after_costs(gross, turnover, cost_bps=10)
    assert result.iloc[0] == pytest.approx(0.0095)
    assert result.iloc[1] == pytest.approx(0.019)


def test_turnover_rejects_duplicate_date_ticker():
    weights = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"] * 2), "ticker": ["AAA", "AAA"], "weight": [0.5, 0.5]})
    with pytest.raises(ValueError):
        portfolio_turnover(weights)


def test_costs_reject_negative_inputs():
    with pytest.raises(ValueError):
        net_returns_after_costs(pd.Series([0.01]), pd.Series([0.5]), cost_bps=-1)
    with pytest.raises(ValueError):
        net_returns_after_costs(pd.Series([0.01]), pd.Series([-0.5]), cost_bps=5)
