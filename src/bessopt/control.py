"""Receding-horizon dispatch under controlled forecast error."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import BatteryConfig, TariffConfig
from .data import validate_timeseries
from .optimization import optimize_dispatch


@dataclass(frozen=True)
class MPCResult:
    """Realized MPC trajectory and forecast diagnostics."""

    dispatch: pd.DataFrame
    solve_time_seconds: float
    optimizations: int
    load_forecast_rmse_kw: float
    solar_forecast_rmse_kw: float


def _perturbed_forecast(
    actual: pd.DataFrame,
    rng: np.random.Generator,
    error_scale: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a controlled forecast whose variance grows with lead time.

    This is an oracle-perturbation experiment, not a trained forecasting model. It
    isolates controller sensitivity to forecast error while keeping the experiment
    reproducible and the error process visible.
    """

    forecast = actual.copy()
    lead = np.arange(len(actual), dtype=float)
    load_sigma = error_scale * (0.015 + 0.005 * np.sqrt(lead))
    solar_sigma = error_scale * (0.040 + 0.020 * np.sqrt(lead))
    load_error = actual["load_kw"].to_numpy() * rng.normal(0.0, load_sigma)
    solar_error = actual["solar_kw"].to_numpy() * rng.normal(0.0, solar_sigma)
    load_error[0] = 0.0
    solar_error[0] = 0.0
    forecast["load_kw"] = np.maximum(0.0, actual["load_kw"].to_numpy() + load_error)
    forecast["solar_kw"] = np.maximum(0.0, actual["solar_kw"].to_numpy() + solar_error)
    return forecast, load_error, solar_error


def simulate_rolling_horizon(
    data: pd.DataFrame,
    battery: BatteryConfig | None = None,
    tariff: TariffConfig | None = None,
    dt_hours: float = 1.0,
    horizon_steps: int = 24,
    forecast_seed: int = 2026,
    forecast_error_scale: float = 1.0,
) -> MPCResult:
    """Simulate hourly model-predictive control on realized load and PV.

    At each interval the controller builds a new forecast, solves over the next
    ``horizon_steps``, applies only the first battery action, observes the realized
    site balance, and repeats. Energy prices are treated as day-ahead known.
    """

    battery = battery or BatteryConfig()
    tariff = tariff or TariffConfig()
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    if forecast_error_scale < 0:
        raise ValueError("forecast_error_scale cannot be negative.")
    if battery.terminal_soc is None:
        raise ValueError("MPC evaluation requires terminal_soc for a fair comparison.")

    actual = validate_timeseries(data, dt_hours)
    rng = np.random.default_rng(forecast_seed)
    current_energy = battery.initial_energy_kwh
    realized_peak = 0.0
    total_solve_time = 0.0
    rows: list[dict[str, object]] = []
    load_errors: list[np.ndarray] = []
    solar_errors: list[np.ndarray] = []

    for step in range(len(actual)):
        stop = min(len(actual), step + horizon_steps)
        actual_window = actual.iloc[step:stop].reset_index(drop=True)
        forecast, load_error, solar_error = _perturbed_forecast(
            actual_window, rng, forecast_error_scale
        )
        if len(load_error) > 1:
            load_errors.append(load_error[1:])
            solar_errors.append(solar_error[1:])

        current_soc = float(
            np.clip(
                current_energy / battery.capacity_kwh,
                battery.soc_min,
                battery.soc_max,
            )
        )
        current_energy = current_soc * battery.capacity_kwh
        controller_battery = replace(battery, initial_soc=current_soc)
        plan = optimize_dispatch(
            forecast,
            controller_battery,
            tariff,
            dt_hours,
            prior_peak_kw=realized_peak,
        )
        total_solve_time += plan.solve_time_seconds
        charge_kw = float(plan.dispatch.loc[0, "battery_charge_kw"])
        discharge_kw = float(plan.dispatch.loc[0, "battery_discharge_kw"])
        next_energy = (
            current_energy
            + battery.charge_efficiency * charge_kw * dt_hours
            - discharge_kw * dt_hours / battery.discharge_efficiency
        )

        observation = actual.iloc[step]
        net_grid_kw = (
            float(observation["load_kw"])
            - float(observation["solar_kw"])
            + charge_kw
            - discharge_kw
        )
        grid_import_kw = max(net_grid_kw, 0.0)
        if tariff.max_grid_import_kw is not None and grid_import_kw > tariff.max_grid_import_kw + 1e-7:
            raise RuntimeError("Realized MPC import violates max_grid_import_kw.")
        surplus_kw = max(-net_grid_kw, 0.0)
        grid_export_kw = min(surplus_kw, tariff.max_grid_export_kw)
        curtailment_kw = surplus_kw - grid_export_kw
        realized_peak = max(realized_peak, grid_import_kw)

        row = observation.to_dict()
        row.update(
            {
                "battery_charge_kw": charge_kw,
                "battery_discharge_kw": discharge_kw,
                "battery_energy_start_kwh": current_energy,
                "battery_energy_end_kwh": next_energy,
                "battery_soc_percent": 100.0 * next_energy / battery.capacity_kwh,
                "grid_import_kw": grid_import_kw,
                "grid_export_kw": grid_export_kw,
                "solar_curtailment_kw": curtailment_kw,
                "energy_import_cost_usd": grid_import_kw
                * float(observation["buy_price_usd_per_kwh"])
                * dt_hours,
                "export_revenue_usd": grid_export_kw
                * float(observation["sell_price_usd_per_kwh"])
                * dt_hours,
                "degradation_cost_usd": 0.5
                * battery.degradation_cost_usd_per_kwh_throughput
                * (charge_kw + discharge_kw)
                * dt_hours,
                "demand_charge_cost_usd": 0.0,
            }
        )
        rows.append(row)
        current_energy = next_energy

    dispatch = pd.DataFrame(rows)
    dispatch.loc[dispatch.index[0], "demand_charge_cost_usd"] = (
        tariff.demand_charge_usd_per_kw * realized_peak
    )
    dispatch["total_cost_usd"] = (
        dispatch["energy_import_cost_usd"]
        - dispatch["export_revenue_usd"]
        + dispatch["degradation_cost_usd"]
        + dispatch["demand_charge_cost_usd"]
    )
    all_load_errors = np.concatenate(load_errors) if load_errors else np.zeros(1)
    all_solar_errors = np.concatenate(solar_errors) if solar_errors else np.zeros(1)
    return MPCResult(
        dispatch=dispatch,
        solve_time_seconds=total_solve_time,
        optimizations=len(actual),
        load_forecast_rmse_kw=float(np.sqrt(np.mean(all_load_errors**2))),
        solar_forecast_rmse_kw=float(np.sqrt(np.mean(all_solar_errors**2))),
    )
