import pandas as pd
import pytest

from src.backtest.walk_forward import apply_date_split, walk_forward_splits


def test_walk_forward_splits_are_chronological_and_non_overlapping():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    splits = walk_forward_splits(dates, train_size=4, test_size=2)
    assert len(splits) == 3
    for train, test in splits:
        assert train.max() < test.min()
        assert len(train.intersection(test)) == 0
    assert len(splits[0][0]) == 4
    assert len(splits[1][0]) == 6


def test_walk_forward_omits_incomplete_test_window():
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    assert len(walk_forward_splits(dates, train_size=4, test_size=2)) == 1


def test_walk_forward_rejects_invalid_window_sizes():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    with pytest.raises(ValueError):
        walk_forward_splits(dates, train_size=0, test_size=2)
    with pytest.raises(ValueError):
        walk_forward_splits(dates, train_size=4, test_size=2, step_size=0)


def test_apply_date_split_keeps_rows_in_correct_partition():
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    data = pd.DataFrame({"date": dates, "ticker": ["AAA"] * 6, "close": range(100, 106)})
    train_dates, test_dates = dates[:4], dates[4:]
    train, test = apply_date_split(data, train_dates, test_dates)
    assert len(train) == 4
    assert len(test) == 2
    assert train["date"].max() < test["date"].min()
