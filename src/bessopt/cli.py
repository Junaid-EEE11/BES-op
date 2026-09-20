"""Command-line entry point for the complete reproducible workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .experiments import run_research_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bess-study",
        description="Run the reproducible BESS optimization research study.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository/output root.")
    parser.add_argument("--days", type=int, default=30, help="Days per synthetic scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Primary case-study random seed.")
    parser.add_argument("--monte-carlo-runs", type=int, default=20, help="Uncertainty scenarios.")
    parser.add_argument("--mpc-horizon", type=int, default=24, help="MPC look-ahead intervals.")
    parser.add_argument("--forecast-seed", type=int, default=2026, help="Forecast-error seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = ExperimentConfig(
        days=args.days,
        seed=args.seed,
        monte_carlo_runs=args.monte_carlo_runs,
        mpc_horizon_steps=args.mpc_horizon,
        forecast_seed=args.forecast_seed,
    )
    summary = run_research_study(args.root.resolve(), experiment=experiment)
    print(json.dumps(summary["case_study"], indent=2))
    print(f"\nArtifacts written beneath: {args.root.resolve()}")


if __name__ == "__main__":
    main()
