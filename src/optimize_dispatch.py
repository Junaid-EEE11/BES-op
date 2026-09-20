"""Compatibility wrapper around the research-grade dispatch model."""

from pathlib import Path
import sys

import pandas as pd

try:
    from bessopt.baselines import no_battery_dispatch
    from bessopt.config import BatteryConfig, TariffConfig
    from bessopt.optimization import optimize_dispatch
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from bessopt.baselines import no_battery_dispatch
    from bessopt.config import BatteryConfig, TariffConfig
    from bessopt.optimization import optimize_dispatch


def optimize_bess_dispatch(
    df: pd.DataFrame,
    params: BatteryConfig,
    dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Return the legacy wide table while using the corrected LP."""

    tariff = TariffConfig()
    solution = optimize_dispatch(df, params, tariff, dt_hours).dispatch
    baseline = no_battery_dispatch(df, tariff, dt_hours)
    solution["grid_import_without_battery_kw"] = baseline["grid_import_kw"]
    solution["grid_import_with_battery_kw"] = solution["grid_import_kw"]
    solution["cost_without_battery_usd"] = baseline["total_cost_usd"]
    solution["cost_with_battery_usd"] = solution["total_cost_usd"]
    return solution


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data' / 'synthetic_bess_data.csv'
    if not data_path.exists():
        from generate_data import generate_synthetic_data
        generate_synthetic_data().to_csv(data_path, index=False)
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    res = optimize_bess_dispatch(df, BatteryConfig())
    out_path = root / "data" / "optimized_bess_dispatch.csv"
    res.to_csv(out_path, index=False)
    print(f'Saved optimized dispatch to {out_path}')
    print(f"Baseline cost: ${res['cost_without_battery_usd'].sum():,.2f}")
    print(f"Optimized cost: ${res['cost_with_battery_usd'].sum():,.2f}")
