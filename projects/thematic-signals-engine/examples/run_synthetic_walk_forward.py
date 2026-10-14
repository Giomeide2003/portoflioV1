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
from src.backtest.momentum_walk_forward import run_momentum_walk_forward


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
    oos_returns = run_momentum_walk_forward(
        prices,
        train_size=252,
        test_size=63,
        step_size=63,
        lookbacks=(5, 10, 20),
        n_positions=3,
    ).dropna()

    benchmark = benchmark_returns(prices).reindex(oos_returns.index).dropna()
    aligned = oos_returns.reindex(benchmark.index).dropna()
    benchmark = benchmark.reindex(aligned.index)
    comparison = compare_to_benchmark(aligned, benchmark)

    print("Synthetic walk-forward momentum research")
    print(f"Out-of-sample periods: {len(aligned)}")
    print(f"Gross cumulative return: {cumulative_return(aligned):.2%}")
    print(f"Annualized volatility: {annualized_volatility(aligned):.2%}")
    print(f"Sharpe ratio (zero risk-free rate): {sharpe_ratio(aligned):.3f}")
    print(f"Maximum drawdown: {maximum_drawdown(aligned):.2%}")
    print(f"Equal-weight universe benchmark: {comparison['benchmark_return']:.2%}")
    print(f"Excess cumulative return: {comparison['excess_return']:.2%}")
    print("Costs excluded; all input prices are synthetic.")


if __name__ == "__main__":
    main()
