"""Backward-compatible battery configuration import.

New code should import :class:`bessopt.config.BatteryConfig` directly.
"""

from bessopt.config import BatteryConfig

BatteryParams = BatteryConfig

__all__ = ["BatteryConfig", "BatteryParams"]
