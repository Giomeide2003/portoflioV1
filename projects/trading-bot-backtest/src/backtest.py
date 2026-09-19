from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average_strategy(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.Series:
    """Generate long/flat signals from two moving averages."""
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()

    return (short_ma > long_ma).astype(int)


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """Run a long/flat backtest with proportional transaction costs."""
    data = pd.DataFrame({"price": prices, "signal": signals}).dropna()
    data["asset_return"] = data["price"].pct_change().fillna(0.0)

    # Trade on the next observation to avoid look-ahead bias.
    data["position"] = data["signal"].shift(1).fillna(0)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
    data["strategy_return"] = (
        data["position"] * data["asset_return"]
        - data["turnover"] * transaction_cost
    )
    data["equity"] = (1.0 + data["strategy_return"]).cumprod()

    return data


def performance_metrics(
    backtest: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate basic return and risk metrics."""
    returns = backtest["strategy_return"]
    equity = backtest["equity"]

    cumulative_return = equity.iloc[-1] - 1.0
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)

    if volatility > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    trades = int(backtest["turnover"].sum())
    winning_periods = int((returns > 0).sum())
    active_periods = int((backtest["position"] != 0).sum())
    win_rate = winning_periods / active_periods if active_periods else 0.0

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "trades": trades,
        "win_rate": float(win_rate),
    }


def sample_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Create deterministic sample prices for a reproducible example."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0003, 0.012, n)
    prices = 100 * np.cumprod(1 + daily_returns)
    index = pd.bdate_range("2024-01-02", periods=n)

    return pd.Series(prices, index=index, name="price")


if __name__ == "__main__":
    prices = sample_prices()
    signals = moving_average_strategy(prices)
    result = run_backtest(prices, signals)
    metrics = performance_metrics(result)

    print("Backtest sur données synthétiques")
    print(f"Rendement cumulé : {metrics['cumulative_return']:.2%}")
    print(f"Volatilité annualisée : {metrics['annualized_volatility']:.2%}")
    print(f"Ratio de Sharpe : {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum drawdown : {metrics['max_drawdown']:.2%}")
    print(f"Nombre de transactions : {metrics['trades']}")
    print(f"Taux de réussite : {metrics['win_rate']:.2%}")
