import pandas as pd
import pytest

from bessopt.data import generate_synthetic_data, validate_timeseries


def test_synthetic_data_is_reproducible():
    first = generate_synthetic_data(days=2, seed=7)
    second = generate_synthetic_data(days=2, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_change_profiles_but_not_tariffs():
    first = generate_synthetic_data(days=2, seed=7)
    second = generate_synthetic_data(days=2, seed=8)
    assert not first["load_kw"].equals(second["load_kw"])
    assert first["buy_price_usd_per_kwh"].equals(second["buy_price_usd_per_kwh"])


def test_sell_price_above_buy_price_is_rejected():
    data = generate_synthetic_data(days=1)
    data.loc[0, "sell_price_usd_per_kwh"] = 1.0
    with pytest.raises(ValueError, match="Sell price"):
        validate_timeseries(data)


def test_irregular_time_steps_are_rejected():
    data = generate_synthetic_data(days=1)
    data.loc[3, "timestamp"] += pd.Timedelta(minutes=10)
    with pytest.raises(ValueError, match="uniformly spaced"):
        validate_timeseries(data)
