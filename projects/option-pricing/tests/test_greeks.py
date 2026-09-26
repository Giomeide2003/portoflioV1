import pytest

from src.greeks import delta, gamma, theta, vega


PARAMETERS = {
    "spot": 100,
    "strike": 100,
    "time_to_maturity": 1,
    "risk_free_rate": 0.05,
    "volatility": 0.20,
}


def test_call_delta():
    assert delta(**PARAMETERS, option_type="call") == pytest.approx(0.6368, abs=1e-4)


def test_put_delta():
    assert delta(**PARAMETERS, option_type="put") == pytest.approx(-0.3632, abs=1e-4)


def test_call_and_put_gamma_are_equal():
    call_gamma = gamma(**PARAMETERS)
    put_gamma = gamma(**PARAMETERS)
    assert call_gamma == pytest.approx(put_gamma, abs=1e-12)
    assert call_gamma == pytest.approx(0.018762, abs=1e-5)


def test_vega():
    assert vega(**PARAMETERS) == pytest.approx(37.5240, abs=1e-4)


def test_call_theta():
    assert theta(**PARAMETERS, option_type="call") == pytest.approx(-6.4140, abs=1e-4)


def test_put_theta():
    assert theta(**PARAMETERS, option_type="put") == pytest.approx(-1.6579, abs=1e-4)


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        delta(**PARAMETERS, option_type="digital")

    with pytest.raises(ValueError):
        theta(**PARAMETERS, option_type="digital")
