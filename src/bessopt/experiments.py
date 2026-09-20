"""Reproducible case study, sensitivity analysis, and Monte Carlo evaluation."""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .baselines import no_battery_dispatch
from .config import BatteryConfig, ExperimentConfig, TariffConfig
from .control import simulate_rolling_horizon
from .data import generate_synthetic_data
from .metrics import compute_metrics, validate_dispatch
from .optimization import optimize_dispatch
from .plots import plot_dispatch, plot_experiments


def _single_case(
    data: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
    dt_hours: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object], float]:
    baseline = no_battery_dispatch(data, tariff, dt_hours)
    solution = optimize_dispatch(data, battery, tariff, dt_hours)
    metrics = compute_metrics(solution.dispatch, baseline, battery, dt_hours)
    checks = validate_dispatch(solution.dispatch, battery, dt_hours)
    if not checks["passed"]:
        raise RuntimeError(f"Dispatch invariant check failed: {checks}")
    return solution.dispatch, metrics, checks, solution.solve_time_seconds


def run_research_study(
    root: Path,
    experiment: ExperimentConfig | None = None,
    battery: BatteryConfig | None = None,
    tariff: TariffConfig | None = None,
) -> dict[str, object]:
    """Run the complete deterministic study and persist machine-readable artifacts."""

    experiment = experiment or ExperimentConfig()
    battery = battery or BatteryConfig()
    tariff = tariff or TariffConfig()
    data_dir = root / "data"
    figure_dir = root / "figures"
    result_dir = root / "results"
    report_dir = root / "reports"
    for directory in (data_dir, figure_dir, result_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_data(
        days=experiment.days,
        seed=experiment.seed,
        dt_hours=experiment.dt_hours,
    )
    dispatch, metrics, checks, solve_time = _single_case(
        data, battery, tariff, experiment.dt_hours
    )
    data.to_csv(data_dir / "synthetic_bess_data.csv", index=False)
    dispatch.to_csv(result_dir / "optimized_dispatch.csv", index=False)

    perfect_mpc = simulate_rolling_horizon(
        data,
        battery,
        tariff,
        experiment.dt_hours,
        horizon_steps=experiment.mpc_horizon_steps,
        forecast_seed=experiment.forecast_seed,
        forecast_error_scale=0.0,
    )
    noisy_mpc = simulate_rolling_horizon(
        data,
        battery,
        tariff,
        experiment.dt_hours,
        horizon_steps=experiment.mpc_horizon_steps,
        forecast_seed=experiment.forecast_seed,
        forecast_error_scale=1.0,
    )
    baseline = no_battery_dispatch(data, tariff, experiment.dt_hours)
    perfect_mpc_metrics = compute_metrics(
        perfect_mpc.dispatch, baseline, battery, experiment.dt_hours
    )
    noisy_mpc_metrics = compute_metrics(
        noisy_mpc.dispatch, baseline, battery, experiment.dt_hours
    )
    for label, trajectory in (
        ("perfect-forecast MPC", perfect_mpc.dispatch),
        ("noisy-forecast MPC", noisy_mpc.dispatch),
    ):
        mpc_checks = validate_dispatch(trajectory, battery, experiment.dt_hours)
        if not mpc_checks["passed"]:
            raise RuntimeError(f"{label} invariant check failed: {mpc_checks}")
    noisy_mpc.dispatch.to_csv(result_dir / "mpc_dispatch.csv", index=False)

    controller_records = [
        {
            "controller": "No storage",
            "total_cost_usd": metrics["baseline_cost_usd"],
            "savings_percent": 0.0,
            "perfect_foresight_gap_percent": 100.0
            * (metrics["baseline_cost_usd"] - metrics["optimized_cost_usd"])
            / metrics["optimized_cost_usd"],
        },
        {
            "controller": "Perfect foresight",
            "total_cost_usd": metrics["optimized_cost_usd"],
            "savings_percent": metrics["relative_savings_percent"],
            "perfect_foresight_gap_percent": 0.0,
        },
        {
            "controller": "24 h perfect MPC",
            "total_cost_usd": perfect_mpc_metrics["optimized_cost_usd"],
            "savings_percent": perfect_mpc_metrics["relative_savings_percent"],
            "perfect_foresight_gap_percent": 100.0
            * (perfect_mpc_metrics["optimized_cost_usd"] - metrics["optimized_cost_usd"])
            / metrics["optimized_cost_usd"],
        },
        {
            "controller": "24 h noisy MPC",
            "total_cost_usd": noisy_mpc_metrics["optimized_cost_usd"],
            "savings_percent": noisy_mpc_metrics["relative_savings_percent"],
            "perfect_foresight_gap_percent": 100.0
            * (noisy_mpc_metrics["optimized_cost_usd"] - metrics["optimized_cost_usd"])
            / metrics["optimized_cost_usd"],
        },
    ]
    controller_comparison = pd.DataFrame(controller_records)
    controller_comparison.to_csv(result_dir / "controller_comparison.csv", index=False)

    capacity_records: list[dict[str, float]] = []
    for capacity in experiment.capacity_sweep_kwh:
        candidate = replace(
            battery,
            capacity_kwh=capacity,
            max_charge_kw=capacity / 3.0,
            max_discharge_kw=capacity / 3.0,
        )
        _, candidate_metrics, _, candidate_time = _single_case(
            data, candidate, tariff, experiment.dt_hours
        )
        capacity_records.append(
            {
                "capacity_kwh": capacity,
                "power_kw": capacity / 3.0,
                "solve_time_seconds": candidate_time,
                **candidate_metrics,
            }
        )
    capacity_results = pd.DataFrame(capacity_records)
    capacity_results.to_csv(result_dir / "capacity_sensitivity.csv", index=False)

    degradation_records: list[dict[str, float]] = []
    for degradation_cost in experiment.degradation_sweep_usd_per_kwh:
        candidate = replace(
            battery,
            degradation_cost_usd_per_kwh_throughput=degradation_cost,
        )
        _, candidate_metrics, _, candidate_time = _single_case(
            data, candidate, tariff, experiment.dt_hours
        )
        degradation_records.append(
            {
                "degradation_cost_assumption_usd_per_kwh": degradation_cost,
                "solve_time_seconds": candidate_time,
                **candidate_metrics,
            }
        )
    degradation_results = pd.DataFrame(degradation_records)
    degradation_results.to_csv(result_dir / "degradation_sensitivity.csv", index=False)

    monte_carlo_records: list[dict[str, float]] = []
    for scenario in range(experiment.monte_carlo_runs):
        scenario_seed = 10_000 + scenario
        scenario_data = generate_synthetic_data(
            days=experiment.days,
            seed=scenario_seed,
            dt_hours=experiment.dt_hours,
        )
        _, scenario_metrics, _, scenario_time = _single_case(
            scenario_data, battery, tariff, experiment.dt_hours
        )
        monte_carlo_records.append(
            {
                "scenario": float(scenario),
                "seed": float(scenario_seed),
                "solve_time_seconds": scenario_time,
                **scenario_metrics,
            }
        )
    monte_carlo_results = pd.DataFrame(monte_carlo_records)
    monte_carlo_results.to_csv(result_dir / "monte_carlo_results.csv", index=False)

    plot_dispatch(dispatch, figure_dir)
    plot_experiments(
        capacity_results,
        monte_carlo_results,
        controller_comparison,
        figure_dir,
    )

    mc_savings = monte_carlo_results["relative_savings_percent"]
    summary: dict[str, object] = {
        "case_study": metrics,
        "validation": checks,
        "uncertainty": {
            "runs": experiment.monte_carlo_runs,
            "mean_savings_percent": float(mc_savings.mean()),
            "standard_deviation_percent": float(mc_savings.std(ddof=1)),
            "p05_savings_percent": float(np.quantile(mc_savings, 0.05)),
            "p95_savings_percent": float(np.quantile(mc_savings, 0.95)),
            "positive_savings_runs": int((mc_savings > 0).sum()),
        },
        "forecast_control": {
            "horizon_steps": experiment.mpc_horizon_steps,
            "perfect_forecast_savings_percent": perfect_mpc_metrics[
                "relative_savings_percent"
            ],
            "noisy_forecast_savings_percent": noisy_mpc_metrics[
                "relative_savings_percent"
            ],
            "noisy_forecast_cost_usd": noisy_mpc_metrics["optimized_cost_usd"],
            "noisy_forecast_gap_to_perfect_foresight_percent": controller_records[3][
                "perfect_foresight_gap_percent"
            ],
            "load_forecast_rmse_kw": noisy_mpc.load_forecast_rmse_kw,
            "solar_forecast_rmse_kw": noisy_mpc.solar_forecast_rmse_kw,
            "optimizations_per_controller": noisy_mpc.optimizations,
            "noisy_mpc_solver_time_seconds": noisy_mpc.solve_time_seconds,
        },
        "solver": {
            "name": "SciPy HiGHS linear programming",
            "case_study_solve_time_seconds": solve_time,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
        },
        "configuration": {
            "experiment": experiment.to_dict(),
            "battery": battery.to_dict(),
            "tariff": tariff.to_dict(),
        },
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    _write_research_report(report_dir / "research_report.md", summary)
    return summary


def _write_research_report(path: Path, summary: dict[str, object]) -> None:
    case = summary["case_study"]
    uncertainty = summary["uncertainty"]
    forecast = summary["forecast_control"]
    validation = summary["validation"]
    config = summary["configuration"]
    battery = config["battery"]
    tariff = config["tariff"]
    report = f"""# Reproducible research report

## Research question

How much operational value can a behind-the-meter battery create through energy
arbitrage and peak shaving after accounting for conversion losses, cycling cost,
and a fair terminal state-of-charge condition?

## Experimental design

The primary experiment uses a deterministic, seeded {config['experiment']['days']}-day
synthetic commercial load/PV trace. The perfect-foresight linear program is compared
with a no-storage baseline under identical energy and demand tariffs. Battery capacity
and degradation cost are varied one factor at a time. A {uncertainty['runs']}-run Monte
Carlo study changes load and weather realizations while holding the controller and
technical assumptions fixed.

Key assumptions: {battery['capacity_kwh']:.0f} kWh / {battery['max_discharge_kw']:.0f} kW
battery, {battery['charge_efficiency'] * battery['discharge_efficiency'] * 100:.1f}%
round-trip efficiency, ${battery['degradation_cost_usd_per_kwh_throughput']:.3f}/kWh
throughput proxy, and ${tariff['demand_charge_usd_per_kw']:.2f}/kW demand charge.

## Primary result

| Metric | No storage | Optimized BESS | Change |
|---|---:|---:|---:|
| Total bill | ${case['baseline_cost_usd']:,.2f} | ${case['optimized_cost_usd']:,.2f} | ${case['absolute_savings_usd']:,.2f} ({case['relative_savings_percent']:.2f}%) |
| Peak import | {case['peak_import_before_kw']:.2f} kW | {case['peak_import_after_kw']:.2f} kW | {case['peak_reduction_kw']:.2f} kW ({case['peak_reduction_percent']:.2f}%) |
| Grid energy | {case['grid_import_before_kwh']:,.1f} kWh | {case['grid_import_after_kwh']:,.1f} kWh | {case['grid_import_after_kwh'] - case['grid_import_before_kwh']:,.1f} kWh |

The optimized schedule completes {case['equivalent_full_cycles']:.2f} equivalent full
cycles and returns to {case['terminal_soc_percent']:.1f}% SOC. Increased grid energy can
coexist with lower cost because storage loses energy while shifting consumption to less
expensive intervals; this is an economic optimization, not an energy-reduction claim.

## Robustness across synthetic realizations

Mean savings are {uncertainty['mean_savings_percent']:.2f}% (sample SD
{uncertainty['standard_deviation_percent']:.2f} percentage points). The empirical
5th–95th percentile interval is {uncertainty['p05_savings_percent']:.2f}% to
{uncertainty['p95_savings_percent']:.2f}%, with positive savings in
{uncertainty['positive_savings_runs']} of {uncertainty['runs']} scenarios.

## Forecast-driven control

A {forecast['horizon_steps']}-hour receding-horizon controller with perfect short-term
forecasts saves {forecast['perfect_forecast_savings_percent']:.2f}%. Under the stated
lead-time-dependent forecast perturbations (load RMSE {forecast['load_forecast_rmse_kw']:.2f}
kW; solar RMSE {forecast['solar_forecast_rmse_kw']:.2f} kW), realized savings are
{forecast['noisy_forecast_savings_percent']:.2f}%. Its realized cost is
{forecast['noisy_forecast_gap_to_perfect_foresight_percent']:.2f}% above the full-horizon
perfect-foresight lower bound. This separates horizon effects and forecast error from
the storage value claimed by the clairvoyant benchmark.

## Verification

Maximum power-balance residual: {validation['max_power_balance_residual']:.2e} kW.
Maximum energy-state residual: {validation['max_energy_balance_residual']:.2e} kWh.
The automated physical-invariant suite passed: **{validation['passed']}**.

## Interpretation and limitations

These results establish computational correctness and controlled synthetic evidence;
they do not establish field performance or investment bankability. The optimizer has
perfect foresight, the degradation model is linear, network constraints are omitted,
and Monte Carlo traces come from one stated data-generating process. Stronger follow-up
work should use measured feeder data, rolling forecasts, electrochemical aging models,
and distribution-network constraints.

## Reproduction

From the repository root, install the package and run `bess-study`. Raw dispatch,
sensitivity tables, Monte Carlo samples, figures, this report, and environment metadata
are regenerated deterministically.
"""
    path.write_text(report, encoding="utf-8")
