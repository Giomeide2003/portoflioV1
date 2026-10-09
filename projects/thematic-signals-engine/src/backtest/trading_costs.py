"""Turnover and transaction-cost utilities for portfolio backtests."""

import pandas as pd


def portfolio_turnover(weights: pd.DataFrame, weight_column: str = "weight") -> pd.Series:
    """Compute one-way turnover as half the absolute weight changes per date.

    Input requires one row per date/ticker. Missing tickers are treated as zero
    weight; the first date is measured against an initially flat portfolio.
    """
    required = {"date", "ticker", weight_column}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if weights.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker weights are not allowed")
    if weights[weight_column].isna().any():
        raise ValueError("portfolio weights cannot be missing")

    pivot = weights.pivot(index="date", columns="ticker", values=weight_column)
    pivot = pivot.sort_index().fillna(0.0)
    changes = pivot.diff()
    if not changes.empty:
        changes.iloc[0] = pivot.iloc[0]
    return changes.abs().sum(axis=1).mul(0.5).rename("turnover")


def net_returns_after_costs(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float = 5.0) -> pd.Series:
    """Subtract transaction costs from gross returns using one-way turnover.

    Example: 10 bps with 50% turnover deducts 0.0005 from that period.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    aligned_turnover = turnover.reindex(gross_returns.index).fillna(0.0)
    if (aligned_turnover < 0).any():
        raise ValueError("turnover cannot be negative")
    return (gross_returns - aligned_turnover * (cost_bps / 10_000.0)).rename("net_return")
