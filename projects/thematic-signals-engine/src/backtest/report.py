"""Research report helpers for backtest results."""

from pathlib import Path

import pandas as pd

from src.backtest.engine import (
    annualized_volatility,
    cumulative_return,
    maximum_drawdown,
    sharpe_ratio,
)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Build a normalized equity curve from periodic returns."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).cumprod()


def performance_summary(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Create a compact performance summary for research reporting."""
    rows = {
        "Cumulative Return": cumulative_return(portfolio_returns),
        "Annualized Volatility": annualized_volatility(portfolio_returns),
        "Sharpe Ratio": sharpe_ratio(portfolio_returns),
        "Maximum Drawdown": maximum_drawdown(portfolio_returns),
    }

    if benchmark_returns is not None:
        rows["Benchmark Return"] = cumulative_return(benchmark_returns)
        rows["Excess Return"] = (
            rows["Cumulative Return"] - rows["Benchmark Return"]
        )

    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


def save_performance_summary(
    summary: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a performance summary as CSV."""
    summary.to_csv(path, float_format="%.6f")
