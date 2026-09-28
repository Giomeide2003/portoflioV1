import pandas as pd
import pytest

from src.data.market_data import prepare_market_data, validate_market_data


def test_prepare_market_data_normalizes_and_sorts():
    data = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-01"],
            "ticker": ["aapl", "AAPL"],
            "close": [102.0, 100.0],
        }
    )

    result = prepare_market_data(data)

    assert result["ticker"].tolist() == ["AAPL", "AAPL"]
    assert result["close"].tolist() == [100.0, 102.0]


def test_missing_column_raises_value_error():
    data = pd.DataFrame({"date": ["2025-01-01"], "ticker": ["AAPL"]})

    with pytest.raises(ValueError):
        validate_market_data(data)


def test_non_positive_price_raises_value_error():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "ticker": ["AAPL"],
            "close": [0.0],
        }
    )

    with pytest.raises(ValueError):
        validate_market_data(data)
