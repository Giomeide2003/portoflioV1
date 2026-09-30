"""Thematic and multi-factor signal calculations."""

import pandas as pd


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute trailing price momentum by ticker."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    result = prices.copy()
    result["momentum"] = (
        result.groupby("ticker")["close"].pct_change(periods=lookback)
    )
    return result


def multi_horizon_momentum(
    prices: pd.DataFrame,
    short_window: int = 20,
    medium_window: int = 60,
    long_window: int = 120,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Build a weighted momentum signal across multiple lookback horizons."""
    windows = [short_window, medium_window, long_window]
    if any(window <= 0 for window in windows):
        raise ValueError("lookback windows must be positive")
    if len(set(windows)) != len(windows):
        raise ValueError("lookback windows must be distinct")

    if weights is None:
        weights = [1.0, 1.0, 1.0]
    if len(weights) != 3:
        raise ValueError("three weights are required")

    result = prices.copy()
    for window in windows:
        result[f"momentum_{window}d"] = (
            result.groupby("ticker")["close"].pct_change(periods=window)
        )

    available = [result[f"momentum_{window}d"] for window in windows]
    result["multi_horizon_momentum"] = sum(
        weight * signal
        for weight, signal in zip(weights, available)
    ) / sum(weights)

    return result


def cross_sectional_zscore(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Standardize a factor cross-sectionally by date."""
    if column not in data.columns:
        raise ValueError(f"column not found: {column}")

    def zscore(group: pd.Series) -> pd.Series:
        mean = group.mean()
        std = group.std(ddof=0)
        if std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - mean) / std

    return data.groupby("date")[column].transform(zscore)


def relative_valuation_signal(
    data: pd.DataFrame,
    valuation_column: str,
) -> pd.DataFrame:
    """Create a cross-sectional valuation z-score."""
    if valuation_column not in data.columns:
        raise ValueError(f"column not found: {valuation_column}")

    result = data.copy()
    result["valuation_zscore"] = cross_sectional_zscore(
        result,
        valuation_column,
    )
    return result


def combine_factor_signals(
    data: pd.DataFrame,
    factors: list[str],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Combine standardized factors into one composite signal."""
    if not factors:
        raise ValueError("at least one factor is required")

    missing = set(factors).difference(data.columns)
    if missing:
        raise ValueError(f"missing factors: {sorted(missing)}")

    if weights is None:
        weights = [1.0] * len(factors)

    if len(weights) != len(factors):
        raise ValueError("weights and factors must have the same length")

    result = data.copy()
    standardized = [
        cross_sectional_zscore(result, factor)
        for factor in factors
    ]

    result["composite_signal"] = sum(
        weight * factor
        for weight, factor in zip(weights, standardized)
    )
    return result
