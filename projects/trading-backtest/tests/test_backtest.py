import pytest

from src.backtest import (
    backtest_long_only,
    compute_returns,
    generate_long_only_signal,
    moving_average,
)


def test_compute_returns():
    prices = [100, 110, 99]

    assert compute_returns(prices) == pytest.approx([0.10, -0.10])


def test_moving_average():
    prices = [1, 2, 3, 4]

    assert moving_average(prices, 2) == [None, 1.5, 2.5, 3.5]


def test_signal_is_flat_until_long_average_exists():
    prices = [1, 2, 3, 4, 5]

    signals = generate_long_only_signal(prices, short_window=2, long_window=3)

    assert signals[:2] == [0, 0]
    assert signals[2:] == [1, 1, 1]


def test_backtest_uses_previous_period_signal():
    prices = [1, 2, 3, 4]

    returns = backtest_long_only(prices, short_window=2, long_window=3)

    assert returns == pytest.approx([0.0, 0.0, 1 / 3])


def test_invalid_prices_raise_value_error():
    with pytest.raises(ValueError):
        compute_returns([100, 0, 110])


def test_invalid_window_configuration_raises_value_error():
    with pytest.raises(ValueError):
        generate_long_only_signal([1, 2, 3, 4], short_window=3, long_window=2)
