import pandas as pd
import pytest

from src.signals.thematic_signals import (
    combine_factor_signals,
    multi_horizon_momentum,
    rank_cross_sectional_signal,
    select_long_positions,
    cross_sectional_zscore,
    momentum_signal,
    relative_valuation_signal,
)


def sample_prices():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "close": [100.0, 110.0, 100.0, 105.0],
        }
    )


def test_momentum_is_computed_by_ticker():
    result = momentum_signal(sample_prices(), lookback=1)

    assert result.loc[result["ticker"] == "AAA", "momentum"].iloc[1] == pytest.approx(0.10)
    assert result.loc[result["ticker"] == "BBB", "momentum"].iloc[1] == pytest.approx(0.05)


def test_cross_sectional_zscore_has_zero_mean():
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "factor": [1.0, 2.0, 3.0],
        }
    )

    zscore = cross_sectional_zscore(data, "factor")

    assert zscore.mean() == pytest.approx(0.0)
    assert zscore.iloc[0] < zscore.iloc[1] < zscore.iloc[2]


def test_relative_valuation_signal_adds_zscore():
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 2,
            "ticker": ["AAA", "BBB"],
            "pe_ratio": [10.0, 20.0],
        }
    )

    result = relative_valuation_signal(data, "pe_ratio")

    assert "valuation_zscore" in result.columns


def test_composite_signal_requires_matching_weights():
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 2,
            "ticker": ["AAA", "BBB"],
            "momentum": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError):
        combine_factor_signals(data, ["momentum"], weights=[0.5, 0.5])


def test_multi_horizon_momentum_uses_each_window():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    data = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "close": [100.0, 110.0, 121.0, 133.1, 100.0, 105.0, 110.25, 115.7625],
        }
    )

    result = multi_horizon_momentum(
        data,
        short_window=1,
        medium_window=2,
        long_window=3,
        weights=[1.0, 2.0, 1.0],
    )

    assert result.loc[result["ticker"] == "AAA", "momentum_1d"].iloc[-1] == pytest.approx(0.10)
    assert result.loc[result["ticker"] == "AAA", "momentum_2d"].iloc[-1] == pytest.approx(0.21)
    assert result.loc[result["ticker"] == "AAA", "momentum_3d"].iloc[-1] == pytest.approx(0.331)
    assert result.loc[result["ticker"] == "AAA", "multi_horizon_momentum"].iloc[-1] == pytest.approx(
        (0.10 + 2 * 0.21 + 0.331) / 4
    )


def test_multi_horizon_momentum_rejects_invalid_windows():
    data = sample_prices()

    with pytest.raises(ValueError):
        multi_horizon_momentum(data, short_window=0)

    with pytest.raises(ValueError):
        multi_horizon_momentum(data, short_window=5, medium_window=5, long_window=10)

    with pytest.raises(ValueError):
        multi_horizon_momentum(data, short_window=1, medium_window=2, long_window=3, weights=[1.0, -1.0, 0.0])



def test_cross_sectional_rank_is_computed_by_date():
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "signal": [0.1, 0.3, 0.2],
        }
    )

    result = rank_cross_sectional_signal(data, "signal")

    assert result["signal_rank"].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3])


def test_select_long_positions_keeps_top_n():
    data = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "signal": [0.1, 0.4, 0.2, 0.3],
        }
    )

    result = select_long_positions(data, "signal", n_positions=2)

    selected = result.loc[result["selected"], "ticker"].tolist()
    assert selected == ["BBB", "DDD"]


def test_select_long_positions_rejects_invalid_number():
    with pytest.raises(ValueError):
        select_long_positions(sample_prices(), "close", n_positions=0)
