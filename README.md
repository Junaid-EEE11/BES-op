# BES-op

**BES-op** is an auditable research benchmark for optimizing behind-the-meter battery energy storage systems (BESS). It evaluates battery dispatch for energy arbitrage and peak shaving while accounting for conversion losses, battery throughput degradation, demand charges, solar curtailment, and terminal state-of-charge requirements.

The project provides a reproducible synthetic case study, rolling-horizon model predictive control (MPC), sensitivity analyses, Monte Carlo uncertainty analysis, validation checks, plots, and machine-readable research artifacts.

## Features

- Perfect-foresight battery dispatch formulated as a sparse linear program.
- SciPy HiGHS solver integration.
- No-storage baseline under the same tariff assumptions.
- 24-hour rolling-horizon MPC with perfect or perturbed forecasts.
- Capacity and degradation-cost sensitivity analyses.
- Seeded Monte Carlo experiments across synthetic load and solar scenarios.
- Physical invariant checks for power balance, energy balance, SOC limits, and terminal SOC.
- CSV, JSON, Markdown, and figure outputs suitable for audit and further analysis.
- Command-line entry point: `bess-study`.

## Requirements

- Python 3.11 or newer
- NumPy
- pandas
- SciPy
- Matplotlib

Development and notebook dependencies are defined as optional extras in `pyproject.toml`.

## Installation

Clone the repository and install it from the repository root:

```bash
git clone https://github.com/Junaid-EEE11/BES-op.git
cd BES-op
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To install development tools and notebook support:

```bash
python -m pip install -e ".[dev,notebook]"
```

Alternatively, install the runtime dependencies directly with:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run the default reproducible study from the repository root:

```bash
bess-study
```

The command uses a 30-day synthetic scenario, seed `42`, 20 Monte Carlo runs, and a 24-step MPC horizon. Generated artifacts are written beneath the selected root directory.

Customize the experiment with:

```bash
bess-study \\
  --root ./study-output \\
  --days 30 \\
  --seed 42 \\
  --monte-carlo-runs 50 \\
  --mpc-horizon 24 \\
  --forecast-seed 2026
```

The command prints the primary case-study metrics as JSON and reports the artifact directory.

## Outputs

A study creates the following directories:

- `data/` — generated synthetic load, solar, and tariff data.
- `results/` — optimized dispatch, MPC dispatch, controller comparisons, sensitivity tables, Monte Carlo results, and `summary.json`.
- `figures/` — dispatch and experiment plots.
- `reports/` — generated `research_report.md` describing the design, results, verification, and limitations.

Important result files include:

- `results/optimized_dispatch.csv`
- `results/mpc_dispatch.csv`
- `results/controller_comparison.csv`
- `results/capacity_sensitivity.csv`
- `results/degradation_sensitivity.csv`
- `results/monte_carlo_results.csv`
- `results/summary.json`

## Model formulation

The dispatch optimizer minimizes the total operating cost over the study horizon. Its objective combines:

- Grid energy import cost.
- Export revenue where export is permitted.
- A demand charge based on the optimized billing-period peak.
- A linear battery-throughput degradation proxy.
- Solar curtailment through the power-balance constraints.

The model enforces:

- Battery charge and discharge power limits.
- Charge and discharge efficiencies.
- Minimum and maximum stored energy.
- Initial and optional terminal SOC conditions.
- Grid import and export limits.
- Nodal power balance.
- Import power below the optimized peak variable.

The default battery configuration is 120 kWh with 40 kW charge and discharge limits, 95% charge and discharge efficiency, 10–90% SOC limits, 50% initial and terminal SOC, and a throughput degradation proxy of `$0.02/kWh`.

## Python API

Core functionality is available from the `bessopt` package. For example:

```python
from pathlib import Path

from bessopt.config import ExperimentConfig
from bessopt.experiments import run_research_study

summary = run_research_study(
    Path("study-output"),
    experiment=ExperimentConfig(days=7, monte_carlo_runs=5),
)

print(summary["case_study"])
```

The lower-level components include:

- `bessopt.data` — synthetic time-series generation and validation.
- `bessopt.optimization` — perfect-foresight linear-program dispatch.
- `bessopt.baselines` — no-storage comparison policy.
- `bessopt.control` — rolling-horizon MPC simulation.
- `bessopt.metrics` — cost, savings, dispatch, and invariant metrics.
- `bessopt.experiments` — end-to-end experiment orchestration.
- `bessopt.plots` — figure generation.

## Testing and linting

Run the test suite with:

```bash
pytest
```

Run the configured linter with:

```bash
ruff check .
```

## Reproducibility

Experiments use explicit random seeds and persist their configuration, solver information, package versions, validation results, and generated data. Re-running the same command with the same configuration reproduces the study design and its machine-readable artifacts, subject to differences between software environments and solver versions.

## Limitations

This repository establishes computational correctness and controlled synthetic evidence; it is not a field-performance or investment-bankability study. The benchmark uses perfect foresight for its lower-bound optimizer, a linear degradation proxy, synthetic data from one stated data-generating process, and no distribution-network constraints. Follow-up work should evaluate measured feeder data, forecast uncertainty calibrated to operations, electrochemical aging models, and network constraints.

## Citation

If you use this software or its experimental design, please cite the repository using `CITATION.cff`:

> Hossain, Junaid. *Battery Energy Storage Optimization: An Auditable Research Benchmark*. Version 1.0.0, 2026.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
