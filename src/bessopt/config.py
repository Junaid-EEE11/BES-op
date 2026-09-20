"""Validated configuration objects used by the BESS experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and economic battery parameters.

    The degradation coefficient is a throughput proxy. A complete equivalent
    cycle is one full-capacity charge plus one full-capacity discharge, so the
    model charges half the coefficient to each kWh entering or leaving the
    battery.
    """

    capacity_kwh: float = 120.0
    max_charge_kw: float = 40.0
    max_discharge_kw: float = 40.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50
    terminal_soc: float | None = 0.50
    degradation_cost_usd_per_kwh_throughput: float = 0.02

    def __post_init__(self) -> None:
        positive = {
            "capacity_kwh": self.capacity_kwh,
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; received {value}.")
        for name in ("charge_efficiency", "discharge_efficiency"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]; received {value}.")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("SOC limits must satisfy 0 <= soc_min < soc_max <= 1.")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must lie within the SOC limits.")
        if self.terminal_soc is not None and not self.soc_min <= self.terminal_soc <= self.soc_max:
            raise ValueError("terminal_soc must be None or lie within the SOC limits.")
        if self.degradation_cost_usd_per_kwh_throughput < 0:
            raise ValueError("degradation cost cannot be negative.")

    @property
    def min_energy_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min

    @property
    def max_energy_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max

    @property
    def initial_energy_kwh(self) -> float:
        return self.capacity_kwh * self.initial_soc

    @property
    def terminal_energy_kwh(self) -> float | None:
        if self.terminal_soc is None:
            return None
        return self.capacity_kwh * self.terminal_soc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TariffConfig:
    """Grid constraints and non-energy tariff components."""

    demand_charge_usd_per_kw: float = 12.0
    max_grid_import_kw: float | None = None
    max_grid_export_kw: float = 0.0

    def __post_init__(self) -> None:
        if self.demand_charge_usd_per_kw < 0:
            raise ValueError("demand charge cannot be negative.")
        if self.max_grid_import_kw is not None and self.max_grid_import_kw <= 0:
            raise ValueError("max_grid_import_kw must be positive or None.")
        if self.max_grid_export_kw < 0:
            raise ValueError("max_grid_export_kw cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible default study design."""

    days: int = 30
    seed: int = 42
    dt_hours: float = 1.0
    monte_carlo_runs: int = 20
    mpc_horizon_steps: int = 24
    forecast_seed: int = 2026
    capacity_sweep_kwh: tuple[float, ...] = (40.0, 80.0, 120.0, 160.0, 200.0)
    degradation_sweep_usd_per_kwh: tuple[float, ...] = (0.0, 0.01, 0.02, 0.04, 0.08)

    def __post_init__(self) -> None:
        if (
            self.days <= 0
            or self.monte_carlo_runs <= 0
            or self.dt_hours <= 0
            or self.mpc_horizon_steps <= 0
        ):
            raise ValueError(
                "days, monte_carlo_runs, dt_hours, and mpc_horizon_steps must be positive."
            )
        if any(value <= 0 for value in self.capacity_sweep_kwh):
            raise ValueError("capacity sweep values must be positive.")
        if any(value < 0 for value in self.degradation_sweep_usd_per_kwh):
            raise ValueError("degradation sweep values cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["capacity_sweep_kwh"] = list(self.capacity_sweep_kwh)
        result["degradation_sweep_usd_per_kwh"] = list(
            self.degradation_sweep_usd_per_kwh
        )
        return result
