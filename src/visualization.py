"""Visualization utilities for BESS optimization."""
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_bess_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_datetime(df['timestamp'])

    plt.figure(figsize=(12, 5))
    plt.plot(x, df['load_kw'], label='Load')
    plt.plot(x, df['solar_kw'], label='Solar PV')
    plt.plot(x, df['grid_import_with_battery_kw'], label='Grid Import with BESS')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title('Load, Solar PV, and Optimized Grid Import')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'load_solar_grid_import.png', dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(x, df['battery_soc_percent'], label='Battery SOC')
    plt.xlabel('Time')
    plt.ylabel('SOC (%)')
    plt.title('Battery State of Charge')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'battery_soc.png', dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(x, df['battery_charge_kw'], label='Charge')
    plt.plot(x, df['battery_discharge_kw'], label='Discharge')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title('Battery Charge and Discharge Schedule')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'battery_dispatch.png', dpi=160)
    plt.close()


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'data' / 'optimized_bess_dispatch.csv')
    plot_bess_results(df, root / 'figures')
    print('Figures saved.')
