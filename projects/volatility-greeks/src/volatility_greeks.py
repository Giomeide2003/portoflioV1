from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option_type: str = "call"


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> tuple[float, float]:
    if min(spot, strike, time_to_expiry, volatility) <= 0:
        raise ValueError("spot, strike, time_to_expiry and volatility must be positive")

    sqrt_t = sqrt(time_to_expiry)
    d1 = (
        log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def option_price(option: Option) -> float:
    """Black-Scholes price for a European call or put."""
    d1, d2 = _d1_d2(
        option.spot,
        option.strike,
        option.time_to_expiry,
        option.risk_free_rate,
        option.volatility,
    )
    discount = exp(-option.risk_free_rate * option.time_to_expiry)

    if option.option_type == "call":
        return option.spot * normal_cdf(d1) - option.strike * discount * normal_cdf(d2)
    if option.option_type == "put":
        return option.strike * discount * normal_cdf(-d2) - option.spot * normal_cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")


def greeks(option: Option) -> dict[str, float]:
    """Return Delta, Gamma, Vega and Theta."""
    d1, d2 = _d1_d2(
        option.spot,
        option.strike,
        option.time_to_expiry,
        option.risk_free_rate,
        option.volatility,
    )
    discount = exp(-option.risk_free_rate * option.time_to_expiry)
    gamma = normal_pdf(d1) / (option.spot * option.volatility * sqrt(option.time_to_expiry))
    vega = option.spot * normal_pdf(d1) * sqrt(option.time_to_expiry)

    if option.option_type == "call":
        delta = normal_cdf(d1)
        theta = (
            -option.spot * normal_pdf(d1) * option.volatility / (2 * sqrt(option.time_to_expiry))
            - option.risk_free_rate * option.strike * discount * normal_cdf(d2)
        )
    elif option.option_type == "put":
        delta = normal_cdf(d1) - 1.0
        theta = (
            -option.spot * normal_pdf(d1) * option.volatility / (2 * sqrt(option.time_to_expiry))
            + option.risk_free_rate * option.strike * discount * normal_cdf(-d2)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    initial_volatility: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Estimate Black-Scholes implied volatility with Newton-Raphson."""
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    volatility = max(initial_volatility, 1e-6)

    for _ in range(max_iterations):
        option = Option(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            option_type,
        )
        model_price = option_price(option)
        vega = greeks(option)["vega"]

        error = model_price - market_price
        if abs(error) < tolerance:
            return float(volatility)

        if vega < 1e-10:
            break

        volatility -= error / vega
        volatility = float(np.clip(volatility, 1e-6, 5.0))

    # Robust fallback: bisection over a broad volatility interval.
    low, high = 1e-6, 5.0
    low_price = option_price(
        Option(spot, strike, time_to_expiry, risk_free_rate, low, option_type)
    )
    high_price = option_price(
        Option(spot, strike, time_to_expiry, risk_free_rate, high, option_type)
    )

    if not low_price <= market_price <= high_price:
        raise ValueError("market_price is outside the supported volatility range")

    for _ in range(max_iterations * 2):
        mid = 0.5 * (low + high)
        mid_price = option_price(
            Option(spot, strike, time_to_expiry, risk_free_rate, mid, option_type)
        )

        if abs(mid_price - market_price) < tolerance:
            return float(mid)

        if mid_price < market_price:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def synthetic_option() -> Option:
    """Return a reproducible synthetic option for demonstration."""
    return Option(
        spot=100.0,
        strike=105.0,
        time_to_expiry=45 / 365,
        risk_free_rate=0.03,
        volatility=0.22,
        option_type="call",
    )


if __name__ == "__main__":
    option = synthetic_option()
    market_price = option_price(option)
    iv = implied_volatility(
        market_price,
        option.spot,
        option.strike,
        option.time_to_expiry,
        option.risk_free_rate,
        option.option_type,
    )

    print("Analyse Volatilité Implicite et Greeks")
    print(f"Prix Black-Scholes : {market_price:.4f}")
    print(f"Volatilité implicite : {iv:.2%}")

    for name, value in greeks(option).items():
        print(f"{name.capitalize()} : {value:.6f}")
