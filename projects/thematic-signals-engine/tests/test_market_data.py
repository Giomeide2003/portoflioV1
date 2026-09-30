import pandas as pd
import pytest

from src.data.market_data import load_market_data, prepare_market_data, validate_market_data


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


def test_duplicate_date_ticker_raises_value_error():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01"],
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError):
        prepare_market_data(data)


def test_load_market_data_reads_csv(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-01"],
            "ticker": ["MSFT", "MSFT"],
            "close": [410.0, 400.0],
        }
    ).to_csv(path, index=False)

    result = load_market_data(path)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-02",
    ]
    assert result["ticker"].tolist() == ["MSFT", "MSFT"]
