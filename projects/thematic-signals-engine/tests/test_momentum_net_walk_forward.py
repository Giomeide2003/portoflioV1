import pandas as pd
import pytest

from src.backtest.momentum_net_walk_forward import run_momentum_walk_forward_net


def synthetic_prices(n_dates=50):
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="D")
    rows = []
    for ticker, drift in [("AAA", 0.012), ("BBB", 0.004), ("CCC", -0.002)]:
        for i, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": 100 * (1 + drift) ** i})
    return pd.DataFrame(rows)


def test_net_walk_forward_reports_costs_and_oos_returns():
    result = run_momentum_walk_forward_net(
        synthetic_prices(), train_size=20, test_size=5,
        lookbacks=(2, 3, 5), n_positions=1, cost_bps=10,
    )
    assert {"gross_return", "turnover", "net_return", "lookback"} <= set(result.columns)
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert (result["turnover"] >= 0).all()
    assert (result["net_return"] <= result["gross_return"] + 1e-12).all()


def test_net_walk_forward_rejects_duplicate_observations():
    data = synthetic_prices()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate date/ticker"):
        run_momentum_walk_forward_net(data, train_size=20, test_size=5, lookbacks=(2,))


def test_net_walk_forward_rejects_negative_costs():
    with pytest.raises(ValueError, match="cost_bps"):
        run_momentum_walk_forward_net(
            synthetic_prices(), train_size=20, test_size=5,
            lookbacks=(2,), cost_bps=-1,
        )
