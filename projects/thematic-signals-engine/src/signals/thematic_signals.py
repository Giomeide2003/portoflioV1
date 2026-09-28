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
