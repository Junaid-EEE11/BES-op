"""Research-grade battery energy storage optimization toolkit."""

from .config import BatteryConfig, ExperimentConfig, TariffConfig
from .control import MPCResult, simulate_rolling_horizon
from .data import generate_synthetic_data
from .metrics import compute_metrics
from .optimization import DispatchResult, optimize_dispatch

__all__ = [
    "BatteryConfig",
    "DispatchResult",
    "ExperimentConfig",
    "MPCResult",
    "TariffConfig",
    "compute_metrics",
    "generate_synthetic_data",
    "optimize_dispatch",
    "simulate_rolling_horizon",
]

__version__ = "1.0.0"
