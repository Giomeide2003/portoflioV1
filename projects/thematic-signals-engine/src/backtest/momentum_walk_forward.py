"""Walk-forward evaluation for a simple cross-sectional momentum strategy."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.backtest.engine import equal_weight_portfolio_returns, sharpe_ratio
from src.backtest.walk_forward import run_walk_forward
from src.signals.thematic_signals import momentum_signal, select_long_positions


def evaluate_momentum_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    lookbacks: Sequence[int] = (5, 10, 20),
    n_positions: int = 5,
    date_column: str = "date",
    price_column: str = "close",
) -> pd.Series:
    """Select a momentum lookback on train, then return test-period portfolio returns.

    Momentum signals are computed using prices available up to each date.
    The portfolio engine applies a one-period lag to selections before returns.
    The lookback is chosen by training-period Sharpe ratio; test returns are
    never used to choose the parameter.
    """
    if not lookbacks or any(window <= 0 for window in lookbacks):
        raise ValueError("lookbacks must contain positive integers")
    if n_positions <= 0:
        raise ValueError("n_positions must be positive")
    for column in (date_column, "ticker", price_column):
        if column not in train.columns or column not in test.columns:
            raise ValueError(f"both train and test must contain column: {column}")

    train = train.copy()
    test = test.copy()
    train[date_column] = pd.to_datetime(train[date_column])
    test[date_column] = pd.to_datetime(test[date_column])
    if train[date_column].max() >= test[date_column].min():
        raise ValueError("training dates must precede test dates")
    train = train.sort_values([date_column, "ticker"]).reset_index(drop=True)
    test = test.sort_values([date_column, "ticker"]).reset_index(drop=True)

    best_window = None
    best_score = float("-inf")
    for window in lookbacks:
        candidate = momentum_signal(train, lookback=int(window))
        candidate = select_long_positions(candidate, "momentum", n_positions)
        returns = equal_weight_portfolio_returns(candidate, price_column=price_column)
        score = sharpe_ratio(returns)
        if score > best_score:
            best_score = score
            best_window = int(window)

    # Include training prices so the first test signals have enough lookback history.
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.sort_values([date_column, "ticker"]).reset_index(drop=True)
    combined = momentum_signal(combined, lookback=best_window)
    combined = select_long_positions(combined, "momentum", n_positions)
    all_returns = equal_weight_portfolio_returns(
        combined,
        selection_column="selected",
        price_column=price_column,
    )
    test_dates = pd.Index(test[date_column].unique())
    result = all_returns.loc[all_returns.index.isin(test_dates)].sort_index()
    result.name = f"momentum_{best_window}d_return"
    return result


def run_momentum_walk_forward(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    lookbacks: Sequence[int] = (5, 10, 20),
    n_positions: int = 5,
    date_column: str = "date",
    price_column: str = "close",
) -> pd.Series:
    """Tune the momentum lookback on each training window and collect test returns."""
    required = {date_column, "ticker", price_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    ordered = data.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    ordered = ordered.sort_values([date_column, "ticker"]).reset_index(drop=True)

    def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return evaluate_momentum_fold(
            train,
            test,
            lookbacks=lookbacks,
            n_positions=n_positions,
            date_column=date_column,
            price_column=price_column,
        )

    return run_walk_forward(
        ordered,
        evaluate,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        date_column=date_column,
    )
