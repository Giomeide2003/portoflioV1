"""Utilities for validating and normalizing market data."""

import pandas as pd


REQUIRED_COLUMNS = {"date", "ticker", "close"}


def validate_market_data(data: pd.DataFrame) -> None:
    """Validate the minimum schema required by the research pipeline."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("market data cannot be empty")

    if data["date"].isna().any():
        raise ValueError("date cannot contain missing values")

    if data["ticker"].isna().any():
        raise ValueError("ticker cannot contain missing values")

    if (data["close"] <= 0).any():
        raise ValueError("close prices must be positive")


def prepare_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return market data sorted and normalized for downstream research."""
    validate_market_data(data)

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    return result
