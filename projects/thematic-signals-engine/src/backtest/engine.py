"""Vectorized backtest metrics for research signals."""

import math

import pandas as pd


def strategy_returns(
    data: pd.DataFrame,
    signal_column: str,
    price_column: str = "close",
) -> pd.Series:
    """Compute signal returns using a one-period signal lag."""
    if signal_column not in data.columns:
        raise ValueError(f"column not found: {signal_column}")
    if price_column not in data.columns:
        raise ValueError(f"column not found: {price_column}")

    returns = data.groupby("ticker")[price_column].pct_change()
    lagged_signal = data.groupby("ticker")[signal_column].shift(1)

    return lagged_signal * returns


def cumulative_return(returns: pd.Series) -> float:
    """Return cumulative compounded performance."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Return annualized volatility."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(clean.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Return the annualized Sharpe ratio using zero risk-free rate."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    volatility = clean.std(ddof=1)
    if volatility == 0:
        return 0.0

    return float(clean.mean() / volatility * math.sqrt(periods_per_year))


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the maximum peak-to-trough drawdown."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def signal_persistence(signal: pd.Series) -> float:
    """Measure the fraction of observations retaining the same signal sign."""
    clean = signal.dropna()
    if len(clean) < 2:
        return 0.0

    signs = clean.gt(0).astype(int)
    return float(signs.eq(signs.shift(1)).iloc[1:].mean())



def equal_weight_portfolio_returns(
    data: pd.DataFrame,
    selection_column: str = "selected",
    price_column: str = "close",
) -> pd.Series:
    """Compute equal-weight portfolio returns from selected securities.

    Selection is shifted by one period within each ticker so that today's
    return cannot use today's selection decision.
    """
    if selection_column not in data.columns:
        raise ValueError(f"column not found: {selection_column}")
    if price_column not in data.columns:
        raise ValueError(f"column not found: {price_column}")

    result = data.copy()
    result["asset_return"] = result.groupby("ticker")[price_column].pct_change()
    result["position"] = (
        result.groupby("ticker")[selection_column].shift(1).fillna(False).astype(float)
    )

    daily = result[result["position"] > 0].groupby("date")["asset_return"].mean()
    return daily.sort_index()


def backtest_selected_portfolio(
    data: pd.DataFrame,
    selection_column: str = "selected",
    price_column: str = "close",
) -> dict[str, float]:
    """Run an equal-weight selected portfolio and return risk metrics."""
    portfolio_returns = equal_weight_portfolio_returns(
        data,
        selection_column=selection_column,
        price_column=price_column,
    )

    return {
        "cumulative_return": cumulative_return(portfolio_returns),
        "annualized_volatility": annualized_volatility(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "maximum_drawdown": maximum_drawdown(portfolio_returns),
    }
