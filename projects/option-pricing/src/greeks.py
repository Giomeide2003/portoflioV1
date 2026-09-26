"""Black-Scholes Greeks for European vanilla options."""

import math

from src.black_scholes import _normal_cdf


def _normal_pdf(x: float) -> float:
    """Probability density function of the standard normal law."""
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _d1(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float, volatility: float) -> float:
    """Return the Black-Scholes d1 term."""
    return (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * math.sqrt(time_to_maturity))


def delta(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float, volatility: float, option_type: str) -> float:
    """Return the Black-Scholes delta of a European call or put."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    d1 = _d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
    if option_type == "call":
        return _normal_cdf(d1)
    return _normal_cdf(d1) - 1.0


def gamma(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float, volatility: float) -> float:
    """Return the Black-Scholes gamma."""
    d1 = _d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
    return _normal_pdf(d1) / (spot * volatility * math.sqrt(time_to_maturity))


def vega(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float, volatility: float) -> float:
    """Return vega for a one-unit change in volatility."""
    d1 = _d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
    return spot * _normal_pdf(d1) * math.sqrt(time_to_maturity)


def theta(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float, volatility: float, option_type: str) -> float:
    """Return the Black-Scholes theta per year."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    d1 = _d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    first_term = -(spot * _normal_pdf(d1) * volatility) / (2.0 * math.sqrt(time_to_maturity))

    if option_type == "call":
        return first_term - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_maturity) * _normal_cdf(d2)

    return first_term + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_maturity) * _normal_cdf(-d2)
