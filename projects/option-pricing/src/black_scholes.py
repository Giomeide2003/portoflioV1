"""Black-Scholes pricing for European vanilla options."""

import math


def _normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal law."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """Return the Black-Scholes price of a European call or put."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discount_factor = math.exp(-risk_free_rate * time_to_maturity)

    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * discount_factor * _normal_cdf(d2)

    return strike * discount_factor * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
