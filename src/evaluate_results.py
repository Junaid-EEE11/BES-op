"""Legacy summary helper retained for existing notebooks."""
from pathlib import Path
import pandas as pd


def summarize_results(df: pd.DataFrame) -> dict:
    baseline = df["cost_without_battery_usd"].sum()
    optimized = df["cost_with_battery_usd"].sum()
    savings = baseline - optimized
    savings_pct = 100 * savings / baseline if baseline else 0
    peak_before = df["grid_import_without_battery_kw"].max()
    peak_after = df["grid_import_with_battery_kw"].max()
    peak_reduction = peak_before - peak_after
    return {
        'baseline_cost_usd': baseline,
        'optimized_cost_usd': optimized,
        'savings_usd': savings,
        'savings_percent': savings_pct,
        'peak_before_kw': peak_before,
        'peak_after_kw': peak_after,
        'peak_reduction_kw': peak_reduction,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'data' / 'optimized_bess_dispatch.csv')
    metrics = summarize_results(df)
    for key, value in metrics.items():
        print(f'{key}: {value:.3f}')
