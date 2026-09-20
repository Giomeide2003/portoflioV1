from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class Option:
    strike: float
    time_to_expiry: float
    volatility: float
    open_interest: int
    option_type: str
    contract_size: int = 100


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Return Black-Scholes gamma for a European call or put."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("time_to_expiry and volatility must be positive")

    d1 = (
        log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))

    return exp(-risk_free_rate * time_to_expiry) * normal_pdf(d1) / (
        spot * volatility * sqrt(time_to_expiry)
    )


def gamma_exposure(
    spot: float,
    option: Option,
    dealer_sign: float = 1.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Estimate gamma exposure for one option position."""
    if dealer_sign not in (-1.0, 1.0):
        raise ValueError("dealer_sign must be 1 or -1")

    gamma = black_scholes_gamma(
        spot,
        option.strike,
        option.time_to_expiry,
        option.volatility,
        risk_free_rate,
    )

    return (
        dealer_sign
        * gamma
        * option.open_interest
        * option.contract_size
        * spot**2
    )


def aggregate_gamma_exposure(
    spot: float,
    options: list[Option],
    dealer_signs: list[float] | None = None,
) -> float:
    """Aggregate gamma exposure across an option chain."""
    if dealer_signs is None:
        dealer_signs = [1.0] * len(options)

    if len(dealer_signs) != len(options):
        raise ValueError("dealer_signs must match options length")

    return sum(
        gamma_exposure(spot, option, sign)
        for option, sign in zip(options, dealer_signs)
    )


def gamma_profile(
    spots: np.ndarray,
    options: list[Option],
    dealer_signs: list[float] | None = None,
) -> np.ndarray:
    """Compute aggregate gamma exposure over a range of spot levels."""
    return np.array(
        [
            aggregate_gamma_exposure(spot, options, dealer_signs)
            for spot in spots
        ]
    )


def find_gamma_flip(
    spots: np.ndarray,
    exposure: np.ndarray,
) -> float | None:
    """Linearly interpolate the first spot where gamma exposure changes sign."""
    for left in range(len(spots) - 1):
        right = left + 1

        if exposure[left] == 0:
            return float(spots[left])

        if exposure[left] * exposure[right] < 0:
            weight = -exposure[left] / (exposure[right] - exposure[left])
            return float(spots[left] + weight * (spots[right] - spots[left]))

    return None


def sample_option_chain() -> tuple[list[Option], list[float]]:
    """Return a deterministic synthetic option chain for demonstration."""
    options = [
        Option(90, 30 / 365, 0.24, 1200, "call"),
        Option(95, 30 / 365, 0.22, 1800, "call"),
        Option(100, 30 / 365, 0.20, 2500, "call"),
        Option(105, 30 / 365, 0.21, 2100, "put"),
        Option(110, 30 / 365, 0.23, 1500, "put"),
    ]

    # Synthetic dealer-position assumptions used only for the example.
    dealer_signs = [1.0, 1.0, -1.0, -1.0, -1.0]
    return options, dealer_signs


if __name__ == "__main__":
    options, dealer_signs = sample_option_chain()
    spots = np.linspace(90.0, 110.0, 401)
    exposure = gamma_profile(spots, options, dealer_signs)
    flip = find_gamma_flip(spots, exposure)

    print("Analyse Gamma Exposure sur données synthétiques")
    print(f"Gamma Flip estimé : {flip:.2f}" if flip else "Gamma Flip non détecté")
    print(f"Exposition gamma à 100 : "
          f"{aggregate_gamma_exposure(100.0, options, dealer_signs):.2f}")
