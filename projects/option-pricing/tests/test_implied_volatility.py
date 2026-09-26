import pytest

from src.black_scholes import black_scholes_price
from src.implied_volatility import implied_volatility


PARAMETERS = {
    "spot": 100,
    "strike": 100,
    "time_to_maturity": 1,
    "risk_free_rate": 0.05,
}


def test_recovers_call_implied_volatility():
    market_price = black_scholes_price(**PARAMETERS, volatility=0.20, option_type="call")

    result = implied_volatility(
        market_price=market_price,
        **PARAMETERS,
        option_type="call",
    )

    assert result == pytest.approx(0.20, abs=1e-8)


def test_recovers_put_implied_volatility():
    market_price = black_scholes_price(**PARAMETERS, volatility=0.30, option_type="put")

    result = implied_volatility(
        market_price=market_price,
        **PARAMETERS,
        option_type="put",
    )

    assert result == pytest.approx(0.30, abs=1e-8)


def test_invalid_market_price_raises_value_error():
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=0,
            **PARAMETERS,
            option_type="call",
        )


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=10,
            **PARAMETERS,
            option_type="digital",
        )
