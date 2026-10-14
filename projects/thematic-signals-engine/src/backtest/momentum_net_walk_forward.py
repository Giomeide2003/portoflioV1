"""Walk-forward momentum evaluation with holdings-based turnover costs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.backtest.engine import equal_weight_portfolio_returns, sharpe_ratio
from src.backtest.trading_costs import net_returns_after_costs, portfolio_turnover
from src.backtest.walk_forward import apply_date_split, walk_forward_splits
from src.signals.thematic_signals import momentum_signal, select_long_positions


def run_momentum_walk_forward_net(
    data: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    lookbacks: Sequence[int] = (5, 10, 20),
    n_positions: int = 5,
    cost_bps: float = 5.0,
    date_column: str = "date",
    price_column: str = "close",
) -> pd.DataFrame:
    """Return OOS gross/net returns and turnover from lagged equal-weight holdings.

    The lookback is selected by training Sharpe only. Costs are charged to the
    return period in which the rebalanced holdings are held. Turnover is half
    the sum of absolute changes in portfolio weights, including entries/exits.
    This simple model excludes spread, market impact, taxes and broker fees.
    """
    if not lookbacks or any(int(window) <= 0 for window in lookbacks):
        raise ValueError("lookbacks must contain positive integers")
    if n_positions <= 0:
        raise ValueError("n_positions must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    required = {date_column, "ticker", price_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    ordered = data.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    if ordered.duplicated([date_column, "ticker"]).any():
        raise ValueError("duplicate date/ticker observations are not allowed")
    if ordered[price_column].isna().any() or (ordered[price_column] <= 0).any():
        raise ValueError("prices must be non-missing and positive")
    ordered = ordered.sort_values([date_column, "ticker"]).reset_index(drop=True)
    splits = walk_forward_splits(
        ordered[date_column], train_size, test_size, step_size
    )
    if not splits:
        raise ValueError("not enough unique dates for one complete train/test split")

    reports = []
    for train_dates, test_dates in splits:
        train, test = apply_date_split(
            ordered, train_dates, test_dates, date_column
        )
        best_window = None
        best_score = float("-inf")
        for window in lookbacks:
            candidate = momentum_signal(
                train.rename(columns={date_column: "date", price_column: "close"}),
                lookback=int(window),
            )
            candidate = select_long_positions(candidate, "momentum", n_positions)
            score = sharpe_ratio(
                equal_weight_portfolio_returns(candidate, price_column="close")
            )
            if score > best_score:
                best_score = score
                best_window = int(window)

        combined = pd.concat([train, test], ignore_index=True)
        combined = combined.rename(
            columns={date_column: "date", price_column: "close"}
        ).sort_values(["date", "ticker"]).reset_index(drop=True)
        selected = select_long_positions(
            momentum_signal(combined, lookback=best_window),
            "momentum",
            n_positions,
        )
        selected["asset_return"] = selected.groupby("ticker")["close"].pct_change()
        selected["held"] = (
            selected.groupby("ticker")["selected"].shift(1).fillna(False).astype(float)
        )
        held_count = selected.groupby("date")["held"].transform("sum")
        selected["weight"] = selected["held"].div(held_count.where(held_count > 0))
        selected["weight"] = selected["weight"].fillna(0.0)

        daily = selected.groupby("date").apply(
            lambda group: float((group["weight"] * group["asset_return"].fillna(0.0)).sum()),
            include_groups=False,
        ).rename("gross_return")
        weights = selected[["date", "ticker", "weight"]]
        turnover = portfolio_turnover(weights)
        test_index = pd.Index(pd.to_datetime(test["date"].unique())).sort_values()
        gross = daily.reindex(test_index).fillna(0.0)
        fold_turnover = turnover.reindex(test_index).fillna(0.0)
        net = net_returns_after_costs(gross, fold_turnover, cost_bps)
        report = pd.concat([gross, fold_turnover.rename("turnover"), net], axis=1)
        report["lookback"] = best_window
        reports.append(report)

    result = pd.concat(reports).sort_index()
    if result.index.has_duplicates:
        raise ValueError("test windows overlap; use step_size >= test_size")
    result.index.name = date_column
    result["net_return"] = result["net_return"].astype(float)
    return result
