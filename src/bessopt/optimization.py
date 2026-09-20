"""Sparse linear program for behind-the-meter battery dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .config import BatteryConfig, TariffConfig
from .data import validate_timeseries


@dataclass(frozen=True)
class DispatchResult:
    """Solver output plus a tidy, auditable dispatch table."""

    dispatch: pd.DataFrame
    objective_usd: float
    solver_status: str
    solve_time_seconds: float
    max_equality_residual: float


@dataclass(frozen=True)
class _Slices:
    charge: slice
    discharge: slice
    energy: slice
    grid_import: slice
    grid_export: slice
    curtailment: slice
    peak: int
    size: int


def _variable_slices(n: int) -> _Slices:
    charge = slice(0, n)
    discharge = slice(n, 2 * n)
    energy = slice(2 * n, 3 * n + 1)
    grid_import = slice(3 * n + 1, 4 * n + 1)
    grid_export = slice(4 * n + 1, 5 * n + 1)
    curtailment = slice(5 * n + 1, 6 * n + 1)
    peak = 6 * n + 1
    return _Slices(charge, discharge, energy, grid_import, grid_export, curtailment, peak, peak + 1)


def optimize_dispatch(
    data: pd.DataFrame,
    battery: BatteryConfig | None = None,
    tariff: TariffConfig | None = None,
    dt_hours: float = 1.0,
    prior_peak_kw: float = 0.0,
) -> DispatchResult:
    """Find the minimum-cost battery schedule with perfect foresight.

    The LP co-optimizes energy imports, export revenue, a billing-period demand
    charge, PV curtailment, and a linear battery-throughput degradation proxy.
    A terminal SOC constraint makes different strategies economically comparable.
    """

    battery = battery or BatteryConfig()
    tariff = tariff or TariffConfig()
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive.")
    if prior_peak_kw < 0:
        raise ValueError("prior_peak_kw cannot be negative.")
    if tariff.max_grid_import_kw is not None and prior_peak_kw > tariff.max_grid_import_kw:
        raise ValueError("prior_peak_kw cannot exceed max_grid_import_kw.")
    frame = validate_timeseries(data, dt_hours)
    n = len(frame)
    idx = _variable_slices(n)

    load = frame["load_kw"].to_numpy(dtype=float)
    solar = frame["solar_kw"].to_numpy(dtype=float)
    buy = frame["buy_price_usd_per_kwh"].to_numpy(dtype=float)
    sell = frame["sell_price_usd_per_kwh"].to_numpy(dtype=float)

    objective = np.zeros(idx.size)
    throughput_coefficient = 0.5 * battery.degradation_cost_usd_per_kwh_throughput * dt_hours
    objective[idx.charge] = throughput_coefficient
    objective[idx.discharge] = throughput_coefficient
    objective[idx.grid_import] = buy * dt_hours
    objective[idx.grid_export] = -sell * dt_hours
    objective[idx.peak] = tariff.demand_charge_usd_per_kw

    import_upper = tariff.max_grid_import_kw
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0.0, battery.max_charge_kw)] * n)
    bounds.extend([(0.0, battery.max_discharge_kw)] * n)
    bounds.extend([(battery.min_energy_kwh, battery.max_energy_kwh)] * (n + 1))
    bounds.extend([(0.0, import_upper)] * n)
    bounds.extend([(0.0, tariff.max_grid_export_kw)] * n)
    bounds.extend([(0.0, float(value)) for value in solar])
    bounds.append((prior_peak_kw, import_upper))

    # Equalities: initial/terminal energy, intertemporal dynamics, and nodal balance.
    equality_rows = 1 + n + n + int(battery.terminal_soc is not None)
    a_eq = lil_matrix((equality_rows, idx.size), dtype=float)
    b_eq = np.zeros(equality_rows)
    row = 0
    a_eq[row, idx.energy.start] = 1.0
    b_eq[row] = battery.initial_energy_kwh
    row += 1

    for step in range(n):
        a_eq[row, idx.energy.start + step] = -1.0
        a_eq[row, idx.energy.start + step + 1] = 1.0
        a_eq[row, idx.charge.start + step] = -battery.charge_efficiency * dt_hours
        a_eq[row, idx.discharge.start + step] = dt_hours / battery.discharge_efficiency
        row += 1

    for step in range(n):
        # import - export + discharge - charge - curtailment = load - solar
        a_eq[row, idx.charge.start + step] = -1.0
        a_eq[row, idx.discharge.start + step] = 1.0
        a_eq[row, idx.grid_import.start + step] = 1.0
        a_eq[row, idx.grid_export.start + step] = -1.0
        a_eq[row, idx.curtailment.start + step] = -1.0
        b_eq[row] = load[step] - solar[step]
        row += 1

    if battery.terminal_soc is not None:
        a_eq[row, idx.energy.stop - 1] = 1.0
        b_eq[row] = battery.terminal_energy_kwh

    # Every import interval must remain below the optimized billing peak.
    a_ub = lil_matrix((n, idx.size), dtype=float)
    for step in range(n):
        a_ub[step, idx.grid_import.start + step] = 1.0
        a_ub[step, idx.peak] = -1.0

    started = perf_counter()
    solution = linprog(
        objective,
        A_ub=a_ub.tocsr(),
        b_ub=np.zeros(n),
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8},
    )
    elapsed = perf_counter() - started
    if not solution.success:
        raise RuntimeError(f"Dispatch optimization failed ({solution.status}): {solution.message}")

    x = solution.x
    result = frame.copy()
    result["battery_charge_kw"] = x[idx.charge]
    result["battery_discharge_kw"] = x[idx.discharge]
    result["battery_energy_start_kwh"] = x[idx.energy][:-1]
    result["battery_energy_end_kwh"] = x[idx.energy][1:]
    result["battery_soc_percent"] = 100.0 * result["battery_energy_end_kwh"] / battery.capacity_kwh
    result["grid_import_kw"] = x[idx.grid_import]
    result["grid_export_kw"] = x[idx.grid_export]
    result["solar_curtailment_kw"] = x[idx.curtailment]
    result["energy_import_cost_usd"] = result["grid_import_kw"] * result["buy_price_usd_per_kwh"] * dt_hours
    result["export_revenue_usd"] = result["grid_export_kw"] * result["sell_price_usd_per_kwh"] * dt_hours
    result["degradation_cost_usd"] = (
        0.5
        * battery.degradation_cost_usd_per_kwh_throughput
        * (result["battery_charge_kw"] + result["battery_discharge_kw"])
        * dt_hours
    )
    result["demand_charge_cost_usd"] = 0.0
    result.loc[result.index[0], "demand_charge_cost_usd"] = tariff.demand_charge_usd_per_kw * x[idx.peak]
    result["total_cost_usd"] = (
        result["energy_import_cost_usd"]
        - result["export_revenue_usd"]
        + result["degradation_cost_usd"]
        + result["demand_charge_cost_usd"]
    )

    residual = a_eq.tocsr() @ x - b_eq
    max_residual = float(np.max(np.abs(residual)))
    return DispatchResult(
        dispatch=result,
        objective_usd=float(solution.fun),
        solver_status=str(solution.message),
        solve_time_seconds=elapsed,
        max_equality_residual=max_residual,
    )
