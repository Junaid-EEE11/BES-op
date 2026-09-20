"""Performance metrics and physical consistency checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import BatteryConfig


def compute_metrics(
    optimized: pd.DataFrame,
    baseline: pd.DataFrame,
    battery: BatteryConfig,
    dt_hours: float = 1.0,
) -> dict[str, float]:
    """Return technical, economic, and renewable-integration metrics."""

    optimized_cost = float(optimized["total_cost_usd"].sum())
    baseline_cost = float(baseline["total_cost_usd"].sum())
    savings = baseline_cost - optimized_cost
    solar_energy = float(optimized["solar_kw"].sum() * dt_hours)
    load_energy = float(optimized["load_kw"].sum() * dt_hours)
    throughput = float(
        0.5
        * (optimized["battery_charge_kw"] + optimized["battery_discharge_kw"]).sum()
        * dt_hours
    )
    equivalent_cycles = throughput / battery.capacity_kwh
    peak_before = float(baseline["grid_import_kw"].max())
    peak_after = float(optimized["grid_import_kw"].max())
    return {
        "baseline_cost_usd": baseline_cost,
        "optimized_cost_usd": optimized_cost,
        "absolute_savings_usd": savings,
        "relative_savings_percent": 100.0 * savings / baseline_cost if baseline_cost else 0.0,
        "energy_import_cost_usd": float(optimized["energy_import_cost_usd"].sum()),
        "demand_charge_cost_usd": float(optimized["demand_charge_cost_usd"].sum()),
        "export_revenue_usd": float(optimized["export_revenue_usd"].sum()),
        "degradation_cost_usd": float(optimized["degradation_cost_usd"].sum()),
        "peak_import_before_kw": peak_before,
        "peak_import_after_kw": peak_after,
        "peak_reduction_kw": peak_before - peak_after,
        "peak_reduction_percent": 100.0 * (peak_before - peak_after) / peak_before if peak_before else 0.0,
        "grid_import_before_kwh": float(baseline["grid_import_kw"].sum() * dt_hours),
        "grid_import_after_kwh": float(optimized["grid_import_kw"].sum() * dt_hours),
        "solar_energy_kwh": solar_energy,
        "load_energy_kwh": load_energy,
        "solar_curtailment_kwh": float(optimized["solar_curtailment_kw"].sum() * dt_hours),
        "solar_utilization_percent": (
            100.0 * (solar_energy - optimized["solar_curtailment_kw"].sum() * dt_hours) / solar_energy
            if solar_energy
            else 0.0
        ),
        "battery_throughput_kwh": throughput,
        "equivalent_full_cycles": equivalent_cycles,
        "terminal_soc_percent": float(optimized["battery_soc_percent"].iloc[-1]),
    }


def validate_dispatch(
    dispatch: pd.DataFrame,
    battery: BatteryConfig,
    dt_hours: float = 1.0,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check the principal physical invariants of a solved schedule."""

    energy_balance = (
        dispatch["battery_energy_end_kwh"]
        - dispatch["battery_energy_start_kwh"]
        - battery.charge_efficiency * dispatch["battery_charge_kw"] * dt_hours
        + dispatch["battery_discharge_kw"] * dt_hours / battery.discharge_efficiency
    )
    power_balance = (
        dispatch["grid_import_kw"]
        - dispatch["grid_export_kw"]
        + dispatch["battery_discharge_kw"]
        - dispatch["battery_charge_kw"]
        + dispatch["solar_kw"]
        - dispatch["solar_curtailment_kw"]
        - dispatch["load_kw"]
    )
    simultaneous = np.minimum(
        dispatch["battery_charge_kw"].to_numpy(),
        dispatch["battery_discharge_kw"].to_numpy(),
    )
    simultaneous_grid = np.minimum(
        dispatch["grid_import_kw"].to_numpy(),
        dispatch["grid_export_kw"].to_numpy(),
    )
    continuity = (
        dispatch["battery_energy_start_kwh"].iloc[1:].to_numpy()
        - dispatch["battery_energy_end_kwh"].iloc[:-1].to_numpy()
    )
    checks = {
        "max_energy_balance_residual": float(np.abs(energy_balance).max()),
        "max_power_balance_residual": float(np.abs(power_balance).max()),
        "max_interstep_energy_discontinuity_kwh": float(
            np.max(np.abs(continuity)) if len(continuity) else 0.0
        ),
        "initial_energy_deviation_kwh": float(
            abs(dispatch["battery_energy_start_kwh"].iloc[0] - battery.initial_energy_kwh)
        ),
        "terminal_energy_deviation_kwh": float(
            abs(dispatch["battery_energy_end_kwh"].iloc[-1] - battery.terminal_energy_kwh)
            if battery.terminal_energy_kwh is not None
            else 0.0
        ),
        "max_simultaneous_charge_discharge_kw": float(simultaneous.max()),
        "max_simultaneous_grid_exchange_kw": float(simultaneous_grid.max()),
        "soc_below_min_kwh": float(
            max(0.0, battery.min_energy_kwh - dispatch["battery_energy_end_kwh"].min())
        ),
        "soc_above_max_kwh": float(
            max(0.0, dispatch["battery_energy_end_kwh"].max() - battery.max_energy_kwh)
        ),
        "charge_limit_violation_kw": float(
            max(0.0, dispatch["battery_charge_kw"].max() - battery.max_charge_kw)
        ),
        "discharge_limit_violation_kw": float(
            max(0.0, dispatch["battery_discharge_kw"].max() - battery.max_discharge_kw)
        ),
        "curtailment_bound_violation_kw": float(
            max(
                0.0,
                -dispatch["solar_curtailment_kw"].min(),
                (dispatch["solar_curtailment_kw"] - dispatch["solar_kw"]).max(),
            )
        ),
    }
    checks["passed"] = all(float(value) <= tolerance for value in checks.values())
    return checks
