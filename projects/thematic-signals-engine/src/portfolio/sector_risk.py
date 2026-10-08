"""Portfolio sector-limit diagnostics."""

import pandas as pd

from src.portfolio.sector_exposure import sector_exposure


def sector_limit_breaches(
    data: pd.DataFrame,
    max_sector_weight: float = 0.35,
    sector_column: str = "sector",
    selection_column: str = "selected",
) -> pd.DataFrame:
    """Flag dates where a sector's equal-weight exposure exceeds a limit.

    The threshold must be in the inclusive interval (0, 1]. The function
    reports breaches; it does not rebalance or alter portfolio selections.
    """
    if not 0 < max_sector_weight <= 1:
        raise ValueError("max_sector_weight must be in (0, 1]")

    exposure = sector_exposure(
        data,
        sector_column=sector_column,
        selection_column=selection_column,
    )
    if exposure.empty:
        exposure["limit"] = pd.Series(dtype=float)
        exposure["breach"] = pd.Series(dtype=bool)
        return exposure

    exposure["limit"] = float(max_sector_weight)
    exposure["breach"] = exposure["weight"] > max_sector_weight
    return exposure
