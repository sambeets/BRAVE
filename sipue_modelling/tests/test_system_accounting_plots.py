from dataclasses import replace

import pytest

from brave.config import Config
from brave.plotting import export_plots


pytestmark = pytest.mark.unit


def _plot_row(i: int):
    return {
        "Scenario": f"Scenario-{i}",
        "PUE_no_services": 1.20,
        "PUE_carbon_minmax": 1.21,
        "siPUE_no_services": 1.20,
        "siPUE_carbon_minmax": 1.14,
        "SiPUECreditLossMWh_carbon_minmax": 10.0 + i,
        "CarbonMarketCostEUR_carbon_minmax": 50_000.0 + i,
        "CarbonDamageEUR_carbon_minmax": 75_000.0 + i,
        "CoolingMWh_carbon_minmax": 100.0 + i,
        "RecoveredHeatITMWhth_carbon_minmax": 80.0 + i,
        "RecoveredHeatCoolingMWhth_carbon_minmax": 20.0 + i,
    }


def test_export_plots_generates_descriptive_non_correlation_set(tmp_path):
    cfg = replace(
        Config(enforce_locked_calibration=False),
        export_plots=True,
        plot_output_dir=str(tmp_path),
        plot_format="png",
    )
    export_plots([_plot_row(i) for i in range(4)], cfg)
    assert {path.name for path in tmp_path.iterdir()} == {
        "pue_sipue_system_direction.png",
        "delay_credit_loss.png",
        "carbon_ledgers.png",
        "cooling_heat_sources.png",
    }
