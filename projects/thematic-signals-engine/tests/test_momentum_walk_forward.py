import pandas as pd
import pytest

from src.backtest.momentum_walk_forward import (
    evaluate_momentum_fold,
    run_momentum_walk_forward,
)


def synthetic_prices(n_dates=40):
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="D")
    rows = []
    for ticker, drift in [("AAA", 0.012), ("BBB", 0.004), ("CCC", -0.002)]:
        for i, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "close": 100 * (1 + drift) ** i})
    return pd.DataFrame(rows)


def test_momentum_fold_returns_only_test_dates():
    data = synthetic_prices()
    train = data[data["date"] < data["date"].sort_values().unique()[25]]
    test = data[data["date"] >= data["date"].sort_values().unique()[25]]
    result = evaluate_momentum_fold(train, test, lookbacks=(2, 3, 5), n_positions=1)
    assert result.index.isin(pd.Index(test["date"].unique())).all()
    assert result.index.is_unique


def test_walk_forward_momentum_runs_multiple_test_windows():
    data = synthetic_prices(40)
    result = run_momentum_walk_forward(
        data, train_size=20, test_size=5, lookbacks=(2, 3, 5), n_positions=1
    )
    assert len(result) > 0
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique


def test_momentum_fold_requires_train_before_test():
    data = synthetic_prices()
    with pytest.raises(ValueError, match="training dates must precede"):
        evaluate_momentum_fold(data, data, lookbacks=(2,), n_positions=1)
