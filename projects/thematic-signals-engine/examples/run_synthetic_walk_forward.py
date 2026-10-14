"""Run a reproducible walk-forward momentum experiment on synthetic prices.

From the project directory:
    python -m examples.run_synthetic_walk_forward

This is a software/research smoke test, not evidence of a tradable edge.
"""

import numpy as np
import pandas as pd

from src.backtest.engine import (
    annualized_volatility,
    benchmark_returns,
    compare_to_benchmark,
    cumulative_return,
    maximum_drawdown,
    sharpe_ratio,
)
from src.backtest.momentum_net_walk_forward import run_momentum_walk_forward_net


def make_synthetic_prices(
    n_dates: int = 504,
    n_tickers: int = 8,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate deterministic daily prices with a shared factor and ticker drift."""
    if n_dates < 2 or n_tickers < 2:
        raise ValueError("n_dates and n_tickers must both be at least 2")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    common_returns = rng.normal(0.0002, 0.008, size=n_dates)
    rows = []

    for ticker_idx in range(n_tickers):
        idiosyncratic = rng.normal(0.0001 * (ticker_idx + 1), 0.012, size=n_dates)
        returns = 0.35 * common_returns + idiosyncratic
        prices = 100.0 * np.cumprod(1.0 + returns)
        rows.extend(
            {"date": date, "ticker": f"SYN{ticker_idx + 1:02d}", "close": float(price)}
            for date, price in zip(dates, prices)
        )

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    prices = make_synthetic_prices()
    report = run_momentum_walk_forward_net(
        prices, train_size=252, test_size=63, step_size=63,
        lookbacks=(5, 10, 20), n_positions=3, cost_bps=5.0,
    )
    benchmark = benchmark_returns(prices).reindex(report.index).dropna()
    aligned = report.reindex(benchmark.index).dropna()
    benchmark = benchmark.reindex(aligned.index)
    gross_comparison = compare_to_benchmark(aligned["gross_return"], benchmark)
    net_comparison = compare_to_benchmark(aligned["net_return"], benchmark)

    print("Synthetic walk-forward momentum research")
    print(f"Out-of-sample periods: {len(aligned)}")
    print(f"Average one-way turnover: {aligned['turnover'].mean():.2%}")
    print(f"Gross cumulative return: {cumulative_return(aligned['gross_return']):.2%}")
    print(f"Net cumulative return (5 bps): {cumulative_return(aligned['net_return']):.2%}")
    print(f"Net annualized volatility: {annualized_volatility(aligned['net_return']):.2%}")
    print(f"Net Sharpe ratio (zero risk-free rate): {sharpe_ratio(aligned['net_return']):.3f}")
    print(f"Net maximum drawdown: {maximum_drawdown(aligned['net_return']):.2%}")
    print(f"Equal-weight universe benchmark: {gross_comparison['benchmark_return']:.2%}")
    print(f"Gross excess cumulative return: {gross_comparison['excess_return']:.2%}")
    print(f"Net excess cumulative return: {net_comparison['excess_return']:.2%}")
    print("Synthetic prices; costs are simplified assumptions, not market-calibrated.")


if __name__ == "__main__":
    main()
