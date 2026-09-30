"""Utilities for loading, validating and normalizing market data."""

from pathlib import Path

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

    if not pd.api.types.is_numeric_dtype(data["close"]):
        raise ValueError("close prices must be numeric")


def prepare_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return market data sorted and normalized for downstream research."""
    validate_market_data(data)

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()

    if result["ticker"].eq("").any():
        raise ValueError("ticker cannot be empty")

    if result.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker observations are not allowed")

    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_market_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV market-data file and return normalized observations."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"market data file not found: {file_path}")

    data = pd.read_csv(file_path)
    return prepare_market_data(data)
