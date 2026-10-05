"""Sector exposure and concentration analysis for selected portfolios."""

import pandas as pd


def sector_exposure(
    data: pd.DataFrame,
    sector_column: str = "sector",
    selection_column: str = "selected",
) -> pd.DataFrame:
    """Compute equal-weight sector exposure among selected positions by date."""
    for column in (sector_column, selection_column):
        if column not in data.columns:
            raise ValueError(f"column not found: {column}")

    selected = data.loc[data[selection_column].fillna(False).astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["date", sector_column, "weight"])

    counts = (
        selected.groupby(["date", sector_column])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("date")["count"].transform("sum")
    counts["weight"] = counts["count"] / totals

    return counts[["date", sector_column, "weight"]].sort_values(
        ["date", sector_column]
    ).reset_index(drop=True)


def sector_concentration(
    data: pd.DataFrame,
    sector_column: str = "sector",
    selection_column: str = "selected",
) -> pd.DataFrame:
    """Measure the largest sector weight and HHI for each date."""
    exposure = sector_exposure(data, sector_column, selection_column)

    if exposure.empty:
        return pd.DataFrame(columns=["date", "largest_sector_weight", "sector_hhi"])

    grouped = exposure.groupby("date")["weight"]
    result = grouped.agg(
        largest_sector_weight="max",
        sector_hhi=lambda weights: float((weights**2).sum()),
    ).reset_index()

    return result
