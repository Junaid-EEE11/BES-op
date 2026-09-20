import numpy as np
import pandas as pd

from bessopt.baselines import no_battery_dispatch
from bessopt.config import BatteryConfig, TariffConfig
from bessopt.data import generate_synthetic_data
from bessopt.metrics import compute_metrics, validate_dispatch
from bessopt.optimization import optimize_dispatch


def test_dispatch_obeys_physics_and_terminal_soc():
    data = generate_synthetic_data(days=3, seed=11)
    battery = BatteryConfig()
    result = optimize_dispatch(data, battery, TariffConfig())
    checks = validate_dispatch(result.dispatch, battery)

    assert checks["passed"]
    assert result.max_equality_residual < 1e-7
    assert np.isclose(result.dispatch["battery_soc_percent"].iloc[-1], 50.0)
    assert np.isclose(result.objective_usd, result.dispatch["total_cost_usd"].sum())


def test_flat_tariff_with_losses_and_no_demand_charge_does_not_cycle():
    data = generate_synthetic_data(days=1, seed=3)
    data["buy_price_usd_per_kwh"] = 0.15
    data["sell_price_usd_per_kwh"] = 0.0
    battery = BatteryConfig(degradation_cost_usd_per_kwh_throughput=0.02)
    tariff = TariffConfig(demand_charge_usd_per_kw=0.0)
    dispatch = optimize_dispatch(data, battery, tariff).dispatch

    assert dispatch["battery_charge_kw"].max() < 1e-7
    assert dispatch["battery_discharge_kw"].max() < 1e-7


def test_optimized_cost_is_no_worse_than_no_battery_operation():
    data = generate_synthetic_data(days=7, seed=5)
    battery = BatteryConfig()
    tariff = TariffConfig()
    dispatch = optimize_dispatch(data, battery, tariff).dispatch
    baseline = no_battery_dispatch(data, tariff)
    metrics = compute_metrics(dispatch, baseline, battery)

    assert metrics["optimized_cost_usd"] <= metrics["baseline_cost_usd"] + 1e-6
    assert metrics["peak_import_after_kw"] <= metrics["peak_import_before_kw"] + 1e-6


def test_pv_surplus_is_curtailed_when_export_is_disabled():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
            "load_kw": [0.0, 0.0],
            "solar_kw": [10.0, 10.0],
            "buy_price_usd_per_kwh": [0.1, 0.1],
            "sell_price_usd_per_kwh": [0.04, 0.04],
        }
    )
    battery = BatteryConfig(
        capacity_kwh=10,
        max_charge_kw=5,
        max_discharge_kw=5,
        soc_min=0.5,
        soc_max=0.9,
        initial_soc=0.9,
        terminal_soc=0.9,
    )
    dispatch = optimize_dispatch(
        data,
        battery,
        TariffConfig(demand_charge_usd_per_kw=0, max_grid_export_kw=0),
    ).dispatch

    assert np.isclose(dispatch["solar_curtailment_kw"].sum(), 20.0)
    assert dispatch["grid_export_kw"].max() < 1e-8
