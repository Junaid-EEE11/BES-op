import pytest

from bessopt.config import BatteryConfig, ExperimentConfig, TariffConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_kwh": 0},
        {"charge_efficiency": 1.01},
        {"soc_min": 0.8, "soc_max": 0.2},
        {"initial_soc": 0.95},
        {"terminal_soc": -0.1},
        {"degradation_cost_usd_per_kwh_throughput": -0.01},
    ],
)
def test_invalid_battery_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BatteryConfig(**kwargs)


def test_invalid_tariff_parameters_are_rejected():
    with pytest.raises(ValueError):
        TariffConfig(demand_charge_usd_per_kw=-1)
    with pytest.raises(ValueError):
        TariffConfig(max_grid_export_kw=-1)


def test_invalid_experiment_parameters_are_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(days=0)
