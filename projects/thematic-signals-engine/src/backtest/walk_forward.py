"""Chronological walk-forward splits and out-of-sample evaluation."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def walk_forward_splits(
    dates: list | pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    """Return expanding-window train/test date splits without shuffling.

    Sizes are measured in unique ordered dates, not rows. The training window
    expands by default. Incomplete trailing test windows are omitted.
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
        splits.append((unique_dates[:test_start], unique_dates[test_start:test_start + test_size]))
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
    return (
        train.sort_values([date_column]).reset_index(drop=True),
        test.sort_values([date_column]).reset_index(drop=True),
    )


def run_walk_forward(
    data: pd.DataFrame,
    evaluate_fold: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    date_column: str = "date",
) -> pd.Series:
    """Run a fold evaluator and concatenate out-of-sample portfolio returns.

    evaluate_fold(train, test) must calibrate using train only and return a
    Series of portfolio returns indexed by dates from the test window.
    """
    if date_column not in data.columns:
        raise ValueError(f"column not found: {date_column}")
    splits = walk_forward_splits(data[date_column], train_size, test_size, step_size)
    if not splits:
        raise ValueError("not enough unique dates for one complete train/test split")

    out_of_sample = []
    for train_dates, test_dates in splits:
        train, test = apply_date_split(data, train_dates, test_dates, date_column)
        fold_returns = evaluate_fold(train, test)
        if not isinstance(fold_returns, pd.Series):
            raise TypeError("evaluate_fold must return a pandas Series indexed by date")
        if fold_returns.index.has_duplicates:
            raise ValueError("fold returns contain duplicate dates")
        returned_dates = set(pd.to_datetime(fold_returns.index))
        allowed_dates = set(pd.to_datetime(test_dates))
        if not returned_dates.issubset(allowed_dates):
            raise ValueError("evaluate_fold returned dates outside the test window")
        fold_returns = fold_returns.copy()
        fold_returns.index = pd.to_datetime(fold_returns.index)
        out_of_sample.append(fold_returns.sort_index())

    result = pd.concat(out_of_sample).sort_index()
    if result.index.has_duplicates:
        raise ValueError("test windows overlap; use step_size >= test_size")
    result.name = "out_of_sample_return"
    return result
