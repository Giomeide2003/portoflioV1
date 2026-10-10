"""Chronological walk-forward splits for out-of-sample evaluation."""

from __future__ import annotations

import pandas as pd


def walk_forward_splits(
    dates: list | pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    """Return expanding-window train/test date splits without shuffling.

    Sizes are measured in unique ordered dates, not rows. The training window
    expands by default; `step_size` controls how far the next test window moves.
    Incomplete trailing test windows are omitted.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    unique_dates = pd.Index(pd.to_datetime(pd.Index(dates).unique())).sort_values()
    if len(unique_dates) < train_size + test_size:
        return []

    splits = []
    test_start = train_size
    while test_start + test_size <= len(unique_dates):
        train_dates = unique_dates[:test_start]
        test_dates = unique_dates[test_start : test_start + test_size]
        splits.append((train_dates, test_dates))
        test_start += step
    return splits


def apply_date_split(
    data: pd.DataFrame,
    train_dates: pd.Index,
    test_dates: pd.Index,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations by date, preserving chronology and avoiding overlap."""
    if date_column not in data.columns:
        raise ValueError(f"column not found: {date_column}")
    dates = pd.to_datetime(data[date_column])
    train_set = set(pd.to_datetime(train_dates))
    test_set = set(pd.to_datetime(test_dates))
    if train_set.intersection(test_set):
        raise ValueError("train and test dates must not overlap")
    train = data.loc[dates.isin(train_set)].copy()
    test = data.loc[dates.isin(test_set)].copy()
    return train.sort_values([date_column]).reset_index(drop=True), test.sort_values([date_column]).reset_index(drop=True)
