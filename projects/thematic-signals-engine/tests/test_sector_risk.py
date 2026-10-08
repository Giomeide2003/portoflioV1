import pandas as pd
import pytest

from src.portfolio.sector_risk import sector_limit_breaches


def portfolio():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Technology", "Technology", "Technology", "Energy"],
            "selected": [True, True, True, True],
        }
    )


def test_flags_sector_exposure_above_limit():
    result = sector_limit_breaches(portfolio(), max_sector_weight=0.60)
    tech = result.loc[result["sector"] == "Technology"].iloc[0]
    energy = result.loc[result["sector"] == "Energy"].iloc[0]

    assert tech["weight"] == pytest.approx(0.75)
    assert bool(tech["breach"])
    assert not bool(energy["breach"])


def test_accepts_exact_limit_without_flagging():
    result = sector_limit_breaches(portfolio(), max_sector_weight=0.75)
    tech = result.loc[result["sector"] == "Technology"].iloc[0]

    assert not bool(tech["breach"])


@pytest.mark.parametrize("limit", [0, -0.1, 1.1])
def test_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        sector_limit_breaches(portfolio(), max_sector_weight=limit)
