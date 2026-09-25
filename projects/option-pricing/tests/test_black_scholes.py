import math

import pytest

from src.black_scholes import black_scholes_price


def test_call_price_matches_known_value():
    price = black_scholes_price(
        spot=100,
        strike=100,
        time_to_maturity=1,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )

    assert price == pytest.approx(10.4506, abs=1e-4)


def test_put_price_matches_known_value():
    price = black_scholes_price(
        spot=100,
        strike=100,
        time_to_maturity=1,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="put",
    )

    assert price == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    spot = 100
    strike = 105
    maturity = 0.75
    rate = 0.04
    volatility = 0.25

    call = black_scholes_price(spot, strike, maturity, rate, volatility, "call")
    put = black_scholes_price(spot, strike, maturity, rate, volatility, "put")

    parity_difference = call - put
    theoretical_difference = spot - strike * math.exp(-rate * maturity)

    assert parity_difference == pytest.approx(theoretical_difference, abs=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0},
        {"strike": 0},
        {"time_to_maturity": 0},
        {"volatility": 0},
    ],
)
def test_invalid_inputs_raise_value_error(kwargs):
    parameters = {
        "spot": 100,
        "strike": 100,
        "time_to_maturity": 1,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "option_type": "call",
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError):
        black_scholes_price(**parameters)


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 1, 0.05, 0.20, "digital")
