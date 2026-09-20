import numpy as np

from bessopt.baselines import no_battery_dispatch
from bessopt.config import BatteryConfig, TariffConfig
from bessopt.control import simulate_rolling_horizon
from bessopt.data import generate_synthetic_data
from bessopt.metrics import compute_metrics, validate_dispatch


def test_rolling_horizon_is_feasible_and_terminally_fair():
    data = generate_synthetic_data(days=2, seed=17)
    battery = BatteryConfig()
    tariff = TariffConfig()
    result = simulate_rolling_horizon(
        data,
        battery,
        tariff,
        horizon_steps=12,
        forecast_seed=99,
    )

    assert result.optimizations == len(data)
    assert result.load_forecast_rmse_kw > 0
    assert result.solar_forecast_rmse_kw > 0
    assert validate_dispatch(result.dispatch, battery)["passed"]
    assert np.isclose(result.dispatch["battery_soc_percent"].iloc[-1], 50.0)


def test_perfect_forecast_mpc_beats_no_storage_on_tou_case():
    data = generate_synthetic_data(days=2, seed=21)
    battery = BatteryConfig()
    tariff = TariffConfig()
    baseline = no_battery_dispatch(data, tariff)
    mpc = simulate_rolling_horizon(
        data,
        battery,
        tariff,
        horizon_steps=24,
        forecast_error_scale=0.0,
    )
    metrics = compute_metrics(mpc.dispatch, baseline, battery)

    assert metrics["absolute_savings_usd"] > 0
