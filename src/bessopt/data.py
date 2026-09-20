"""Deterministic synthetic data with realistic temporal structure."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    days: int = 30,
    seed: int = 42,
    start: str = "2026-01-01",
    dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Generate load, PV, and tariff observations for a commercial site.

    The generator combines diurnal and weekly structure, day-level weather,
    autocorrelated deviations, and a deterministic time-of-use tariff. It is a
    transparent benchmark rather than a claim of field-data validation.
    """

    if days <= 0 or dt_hours <= 0:
        raise ValueError("days and dt_hours must be positive.")
    steps_float = days * 24 / dt_hours
    if not np.isclose(steps_float, round(steps_float)):
        raise ValueError("days * 24 must be divisible by dt_hours.")

    rng = np.random.default_rng(seed)
    steps = int(round(steps_float))
    frequency = pd.to_timedelta(dt_hours, unit="h")
    timestamp = pd.date_range(start, periods=steps, freq=frequency)
    hour = timestamp.hour.to_numpy() + timestamp.minute.to_numpy() / 60
    day_index = ((timestamp - timestamp[0]).total_seconds() / 86400).astype(int)
    day_of_week = timestamp.dayofweek.to_numpy()

    # Commercial load: occupancy, morning ramp, evening peak, and correlated noise.
    base = 48.0 + 5.0 * np.cos(2 * np.pi * (hour - 15) / 24)
    morning = 15.0 * np.exp(-0.5 * ((hour - 9.0) / 2.1) ** 2)
    evening = 23.0 * np.exp(-0.5 * ((hour - 19.0) / 2.6) ** 2)
    weekend = np.where(day_of_week >= 5, 0.84, 1.0)
    innovations = rng.normal(0, 1.9, steps)
    correlated_noise = np.empty(steps)
    correlated_noise[0] = innovations[0]
    for index in range(1, steps):
        correlated_noise[index] = 0.72 * correlated_noise[index - 1] + innovations[index]
    load_kw = np.maximum(20.0, (base + morning + evening) * weekend + correlated_noise)

    # PV: smooth clear-sky envelope modulated by a daily cloud state and fast noise.
    unique_days = int(day_index.max()) + 1
    daily_cloud = np.clip(rng.beta(8, 2, unique_days) + rng.normal(0, 0.05, unique_days), 0.35, 1.05)
    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) ** 1.55
    solar_kw = 58.0 * daylight * daily_cloud[day_index]
    solar_kw *= np.clip(1.0 + rng.normal(0, 0.035, steps), 0.75, 1.10)
    solar_kw = np.maximum(0.0, solar_kw)

    # Buy tariff plus a deliberately lower export credit to prevent grid arbitrage.
    buy_price = np.select(
        [
            (hour >= 0) & (hour < 6),
            (hour >= 6) & (hour < 16),
            (hour >= 16) & (hour < 22),
            (hour >= 22),
        ],
        [0.08, 0.14, 0.31, 0.11],
        default=0.14,
    )
    sell_price = np.full(steps, 0.045)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "load_kw": np.round(load_kw, 4),
            "solar_kw": np.round(solar_kw, 4),
            "buy_price_usd_per_kwh": np.round(buy_price, 4),
            "sell_price_usd_per_kwh": np.round(sell_price, 4),
        }
    )


def validate_timeseries(data: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    """Validate and normalize an optimization input table."""

    frame = data.copy()
    if "buy_price_usd_per_kwh" not in frame and "price_usd_per_kwh" in frame:
        frame["buy_price_usd_per_kwh"] = frame["price_usd_per_kwh"]
    if "sell_price_usd_per_kwh" not in frame:
        frame["sell_price_usd_per_kwh"] = 0.0

    required = {
        "timestamp",
        "load_kw",
        "solar_kw",
        "buy_price_usd_per_kwh",
        "sell_price_usd_per_kwh",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input data must contain at least one row.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("timestamp must be strictly increasing and unique.")
    if len(frame) > 1:
        observed = frame["timestamp"].diff().dropna().dt.total_seconds().to_numpy() / 3600
        if not np.allclose(observed, dt_hours, rtol=0, atol=1e-8):
            raise ValueError(f"Timestamps must be uniformly spaced at {dt_hours} hour intervals.")

    numeric = [
        "load_kw",
        "solar_kw",
        "buy_price_usd_per_kwh",
        "sell_price_usd_per_kwh",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Input contains NaN or infinite values.")
    if (frame[["load_kw", "solar_kw"]] < 0).any().any():
        raise ValueError("Load and solar generation cannot be negative.")
    if (frame[["buy_price_usd_per_kwh", "sell_price_usd_per_kwh"]] < 0).any().any():
        raise ValueError("This benchmark requires non-negative energy prices.")
    if (frame["sell_price_usd_per_kwh"] > frame["buy_price_usd_per_kwh"] + 1e-12).any():
        raise ValueError("Sell price cannot exceed buy price in the linear no-arbitrage model.")
    return frame.reset_index(drop=True)
