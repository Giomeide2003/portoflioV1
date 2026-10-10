"""Vectorized backtest metrics for research signals."""

import math

import pandas as pd

from src.backtest.trading_costs import net_returns_after_costs


def strategy_returns(data: pd.DataFrame, signal_column: str, price_column: str = "close") -> pd.Series:
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
    return float((1.0 + clean).prod() - 1.0) if not clean.empty else 0.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return annualized volatility."""
    clean = returns.dropna()
    return float(clean.std(ddof=1) * math.sqrt(periods_per_year)) if len(clean) > 1 else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return annualized Sharpe ratio assuming a zero risk-free rate."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    volatility = clean.std(ddof=1)
    return float(clean.mean() / volatility * math.sqrt(periods_per_year)) if volatility else 0.0


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the maximum peak-to-trough drawdown."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def signal_persistence(signal: pd.Series) -> float:
    """Measure the fraction of observations retaining the same signal sign."""
    clean = signal.dropna()
    if len(clean) < 2:
        return 0.0
    signs = clean.gt(0).astype(int)
    return float(signs.eq(signs.shift(1)).iloc[1:].mean())


def equal_weight_portfolio_returns(data: pd.DataFrame, selection_column: str = "selected", price_column: str = "close") -> pd.Series:
    """Compute equal-weight portfolio returns with a one-period selection lag."""
    for column in (selection_column, price_column, "ticker", "date"):
        if column not in data.columns:
            raise ValueError(f"column not found: {column}")
    result = data.copy()
    result["asset_return"] = result.groupby("ticker")[price_column].pct_change()
    result["position"] = result.groupby("ticker")[selection_column].shift(1).fillna(False).astype(float)
    daily = result[result["position"] > 0].groupby("date")["asset_return"].mean()
    return daily.sort_index()


def backtest_selected_portfolio(data: pd.DataFrame, selection_column: str = "selected", price_column: str = "close") -> dict[str, float]:
    """Run an equal-weight selected portfolio and return gross risk metrics."""
    returns = equal_weight_portfolio_returns(data, selection_column, price_column)
    return {"cumulative_return": cumulative_return(returns), "annualized_volatility": annualized_volatility(returns), "sharpe_ratio": sharpe_ratio(returns), "maximum_drawdown": maximum_drawdown(returns)}


def benchmark_returns(data: pd.DataFrame, price_column: str = "close") -> pd.Series:
    """Compute equal-weight benchmark returns across the available universe."""
    if price_column not in data.columns:
        raise ValueError(f"column not found: {price_column}")
    result = data.copy()
    result["asset_return"] = result.groupby("ticker")[price_column].pct_change()
    return result.groupby("date")["asset_return"].mean().sort_index()


def compare_to_benchmark(portfolio_returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Compare compounded portfolio and equal-weight universe benchmark returns."""
    aligned = pd.concat([portfolio_returns.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {"portfolio_return": 0.0, "benchmark_return": 0.0, "excess_return": 0.0}
    portfolio_return = cumulative_return(aligned["portfolio"])
    benchmark_return = cumulative_return(aligned["benchmark"])
    return {"portfolio_return": portfolio_return, "benchmark_return": benchmark_return, "excess_return": portfolio_return - benchmark_return}


def compare_gross_net_performance(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float = 5.0) -> dict[str, float]:
    """Report gross/net cumulative return and the performance drag from costs."""
    net_returns = net_returns_after_costs(gross_returns, turnover, cost_bps)
    gross_return = cumulative_return(gross_returns)
    net_return = cumulative_return(net_returns)
    return {
        "gross_cumulative_return": gross_return,
        "net_cumulative_return": net_return,
        "transaction_cost_drag": gross_return - net_return,
        "gross_sharpe_ratio": sharpe_ratio(gross_returns),
        "net_sharpe_ratio": sharpe_ratio(net_returns),
        "gross_maximum_drawdown": maximum_drawdown(gross_returns),
        "net_maximum_drawdown": maximum_drawdown(net_returns),
    }
