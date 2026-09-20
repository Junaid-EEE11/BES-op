"""Transparent comparison policies for optimized dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TariffConfig
from .data import validate_timeseries


def no_battery_dispatch(
    data: pd.DataFrame,
    tariff: TariffConfig | None = None,
    dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Evaluate the site without storage under the same tariff."""

    tariff = tariff or TariffConfig()
    frame = validate_timeseries(data, dt_hours)
    net_load = frame["load_kw"].to_numpy() - frame["solar_kw"].to_numpy()
    grid_import = np.maximum(net_load, 0.0)
    if tariff.max_grid_import_kw is not None and np.any(grid_import > tariff.max_grid_import_kw + 1e-9):
        raise ValueError("The no-battery case violates max_grid_import_kw.")
    available_export = np.maximum(-net_load, 0.0)
    grid_export = np.minimum(available_export, tariff.max_grid_export_kw)
    curtailment = available_export - grid_export

    result = frame.copy()
    result["grid_import_kw"] = grid_import
    result["grid_export_kw"] = grid_export
    result["solar_curtailment_kw"] = curtailment
    result["energy_import_cost_usd"] = grid_import * result["buy_price_usd_per_kwh"] * dt_hours
    result["export_revenue_usd"] = grid_export * result["sell_price_usd_per_kwh"] * dt_hours
    result["degradation_cost_usd"] = 0.0
    result["demand_charge_cost_usd"] = 0.0
    result.loc[result.index[0], "demand_charge_cost_usd"] = (
        tariff.demand_charge_usd_per_kw * float(np.max(grid_import))
    )
    result["total_cost_usd"] = (
        result["energy_import_cost_usd"]
        - result["export_revenue_usd"]
        + result["demand_charge_cost_usd"]
    )
    return result
