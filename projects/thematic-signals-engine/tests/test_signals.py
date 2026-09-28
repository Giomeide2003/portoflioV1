import pandas as pd
import pytest

from src.signals.thematic_signals import (
    combine_factor_signals,
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
