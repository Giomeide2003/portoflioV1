from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Position:
    asset: str
    quantity: float
    entry_price: float


def portfolio_pnl(
    positions: list[Position],
    current_prices: dict[str, float],
) -> pd.DataFrame:
    """Calculate mark-to-market P&L for each portfolio position."""
    rows = []

    for position in positions:
        if position.asset not in current_prices:
            raise KeyError(f"Missing current price for {position.asset}")

        current_price = current_prices[position.asset]
        pnl = position.quantity * (current_price - position.entry_price)

        rows.append(
            {
                "asset": position.asset,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(rows)


def portfolio_returns(
    asset_returns: pd.DataFrame,
    positions: list[Position],
    prices: dict[str, float] | None = None,
) -> pd.Series:
    """Build daily portfolio returns from asset returns and position weights."""
    missing = [position.asset for position in positions if position.asset not in asset_returns]
    if missing:
        raise KeyError(f"Missing return series for {missing}")

    if prices is None:
        prices = {position.asset: position.entry_price for position in positions}

    notionals = {
        position.asset: position.quantity * prices[position.asset]
        for position in positions
    }
    total_notional = sum(abs(value) for value in notionals.values())

    if total_notional == 0:
        raise ValueError("Portfolio notional must be non-zero")

    weights = {
        asset: notional / total_notional
        for asset, notional in notionals.items()
    }

    weighted = pd.Series(0.0, index=asset_returns.index)
    for asset, weight in weights.items():
        weighted = weighted + weight * asset_returns[asset]

    return weighted


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Historical one-period VaR reported as a positive loss amount."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    quantile = returns.quantile(1.0 - confidence)
    return float(max(0.0, -quantile * portfolio_value))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Normal parametric one-period VaR using the sample standard deviation."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    z_scores = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.99: 2.3263,
    }

    if confidence not in z_scores:
        raise ValueError("Supported confidence levels: 0.90, 0.95, 0.99")

    mean = returns.mean()
    volatility = returns.std(ddof=1)
    loss_threshold = -(mean - z_scores[confidence] * volatility)

    return float(max(0.0, loss_threshold * portfolio_value))


def stress_test(
    positions: list[Position],
    current_prices: dict[str, float],
    shocks: dict[str, float],
) -> pd.DataFrame:
    """Apply percentage price shocks and calculate stressed P&L."""
    rows = []

    for position in positions:
        if position.asset not in current_prices:
            raise KeyError(f"Missing current price for {position.asset}")

        shock = shocks.get(position.asset, 0.0)
        current_price = current_prices[position.asset]
        stressed_price = current_price * (1.0 + shock)
        pnl = position.quantity * (stressed_price - current_price)

        rows.append(
            {
                "asset": position.asset,
                "shock": shock,
                "current_price": current_price,
                "stressed_price": stressed_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(rows)


def sample_market_data() -> tuple[list[Position], pd.DataFrame, dict[str, float]]:
    """Return deterministic synthetic positions and market returns."""
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "INDEX": rng.normal(0.0002, 0.012, 500),
            "TECH": rng.normal(0.0003, 0.018, 500),
            "BOND": rng.normal(0.0001, 0.006, 500),
        }
    )

    positions = [
        Position("INDEX", 100, 100.0),
        Position("TECH", 80, 150.0),
        Position("BOND", 200, 95.0),
    ]
    prices = {"INDEX": 103.0, "TECH": 158.0, "BOND": 96.0}

    return positions, returns, prices


if __name__ == "__main__":
    positions, returns, prices = sample_market_data()

    pnl = portfolio_pnl(positions, prices)
    portfolio_value = sum(
        position.quantity * prices[position.asset]
        for position in positions
    )
    portfolio_returns_series = portfolio_returns(returns, positions, prices)

    print("Analyse P&L et Risque")
    print(f"P&L total : {pnl['pnl'].sum():.2f}")
    print(f"Valeur du portefeuille : {portfolio_value:.2f}")
    print(
        f"VaR historique 95 % : "
        f"{historical_var(portfolio_returns_series, 0.95, portfolio_value):.2f}"
    )
    print(
        f"VaR paramétrique 95 % : "
        f"{parametric_var(portfolio_returns_series, 0.95, portfolio_value):.2f}"
    )

    stress = stress_test(
        positions,
        prices,
        {"INDEX": -0.10, "TECH": -0.15, "BOND": 0.02},
    )
    print(f"Stress P&L : {stress['pnl'].sum():.2f}")
