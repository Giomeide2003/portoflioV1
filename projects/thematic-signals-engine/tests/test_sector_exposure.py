import pandas as pd
import pytest

from src.portfolio.sector_exposure import sector_concentration, sector_exposure


def sample_portfolio():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Tech", "Tech", "Energy", "Health"],
            "selected": [True, True, True, False],
        }
    )


def test_sector_exposure_is_equal_weighted():
    result = sector_exposure(sample_portfolio())

    tech = result.loc[result["sector"] == "Tech", "weight"].iloc[0]
    energy = result.loc[result["sector"] == "Energy", "weight"].iloc[0]

    assert tech == pytest.approx(2 / 3)
    assert energy == pytest.approx(1 / 3)
    assert result["weight"].sum() == pytest.approx(1.0)


def test_sector_exposure_handles_no_selected_positions():
    data = sample_portfolio()
    data["selected"] = False

    result = sector_exposure(data)

    assert result.empty
    assert list(result.columns) == ["date", "sector", "weight"]


def test_sector_concentration_returns_largest_weight_and_hhi():
    result = sector_concentration(sample_portfolio())

    assert result.loc[0, "largest_sector_weight"] == pytest.approx(2 / 3)
    assert result.loc[0, "sector_hhi"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)


def test_sector_exposure_requires_sector_column():
    data = sample_portfolio().drop(columns="sector")

    with pytest.raises(ValueError):
        sector_exposure(data)
