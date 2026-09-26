"""Implied volatility solver for European vanilla options."""

from src.black_scholes import black_scholes_price
from src.greeks import vega


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    option_type: str,
    initial_volatility: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Recover implied volatility using the Newton-Raphson method."""
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if initial_volatility <= 0:
        raise ValueError("initial_volatility must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    volatility = initial_volatility

    for _ in range(max_iterations):
        price = black_scholes_price(
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            volatility,
            option_type,
        )
        price_error = price - market_price

        if abs(price_error) < tolerance:
            return volatility

        volatility_vega = vega(
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            volatility,
        )

        if volatility_vega <= 1e-12:
            raise ValueError("vega is too small for a stable Newton-Raphson step")

        volatility -= price_error / volatility_vega

        if volatility <= 0:
            raise ValueError("Newton-Raphson produced a non-positive volatility")

    raise ValueError("implied volatility did not converge")
