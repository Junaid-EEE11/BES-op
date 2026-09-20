"""Publication-style figures for the case study and experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd


COLORS = {"load": "#34495e", "solar": "#f39c12", "grid": "#2471a3", "battery": "#148f77", "accent": "#922b21"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def plot_dispatch(dispatch: pd.DataFrame, output_dir: Path, hours: int = 168) -> None:
    """Plot the first week so operational decisions remain legible."""

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = dispatch.iloc[: min(hours, len(dispatch))].copy()
    time = pd.to_datetime(sample["timestamp"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(time, sample["load_kw"], label="Load", color=COLORS["load"], linewidth=1.3)
    axes[0].plot(time, sample["solar_kw"], label="PV generation", color=COLORS["solar"], linewidth=1.3)
    axes[0].plot(time, sample["grid_import_kw"], label="Grid import", color=COLORS["grid"], linewidth=1.3)
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(ncol=3, frameon=False)

    axes[1].fill_between(time, 0, sample["battery_charge_kw"], label="Charge", color=COLORS["battery"], alpha=0.75)
    axes[1].fill_between(time, 0, -sample["battery_discharge_kw"], label="Discharge", color=COLORS["accent"], alpha=0.75)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Battery power (kW)")
    axes[1].legend(ncol=2, frameon=False)

    axes[2].plot(time, sample["battery_soc_percent"], color=COLORS["battery"], linewidth=1.5)
    axes[2].set_ylabel("State of charge (%)")
    axes[2].set_xlabel("Time")
    fig.suptitle("Optimal BESS operation — first seven days", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "dispatch_week.png", bbox_inches="tight")
    plt.close(fig)


def plot_experiments(
    capacity: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    controller_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(capacity["capacity_kwh"], capacity["relative_savings_percent"], marker="o", color=COLORS["grid"])
    ax1.set_xticks(capacity["capacity_kwh"])
    ax1.set_xlabel("Battery nameplate capacity (kWh)")
    ax1.set_ylabel("Bill savings (%)")
    ax1.set_title("Storage value versus battery capacity")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(monte_carlo["relative_savings_percent"], bins="auto", color=COLORS["battery"], edgecolor="white")
    ax.axvline(monte_carlo["relative_savings_percent"].mean(), color=COLORS["accent"], linestyle="--", label="Mean")
    ax.set_xlabel("Bill savings (%)")
    ax.set_ylabel("Simulation count")
    ax.set_title("Savings across synthetic operating scenarios")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "monte_carlo_savings.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(
        controller_comparison["controller"],
        controller_comparison["total_cost_usd"],
        color=["#7f8c8d", COLORS["grid"], COLORS["battery"], COLORS["accent"]],
    )
    ax.bar_label(bars, fmt="$%.0f", padding=3)
    ax.set_ylabel("Realized 30-day cost (USD)")
    ax.set_title("Value of information and rolling-horizon control")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(output_dir / "controller_comparison.png", bbox_inches="tight")
    plt.close(fig)
