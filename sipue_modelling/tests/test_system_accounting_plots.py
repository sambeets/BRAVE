from dataclasses import replace

import pytest

from brave.config import Config
from brave.plotting import (
    export_creative_figure_package,
    export_sensitivity_plots,
)


pytestmark = pytest.mark.unit


def _creative_row(archetype: str, key: str, scenario_id: int):
    row = {
        "archetype": archetype,
        "archetype_key": key,
        "scenario": f"{archetype}-{scenario_id}",
        "scenario_id": scenario_id,
        "construction_delay_months": 6.0,
        "carbon_cap_tco2": 1000.0 + scenario_id,
        "pue_no_services": 1.20,
        "sipue_no_services": 1.20,
        "steady_sipue_no_services": 1.20,
        "cue_no_services": 40.0,
        "e_it_mwh_no_services": 1000.0,
        "e_fac_mwh_no_services": 1200.0,
        "heat_mwhth_no_services": 0.0,
        "system_benefit_per_mwh_it_no_services": 0.0,
        "emissions_tco2_no_services": 40000.0,
        "carbon_damage_eur_no_services": 500000.0,
    }
    for case, pue, sipue, heat in (
        ("baseline", 1.25, 1.18, 70.0),
        ("robust", 1.35, 1.17, 220.0),
    ):
        row.update({
            f"pue_{case}": pue,
            f"sipue_{case}": sipue,
            f"steady_sipue_{case}": sipue - 0.01,
            f"cue_{case}": 39.0,
            f"e_it_mwh_{case}": 1000.0,
            f"e_fac_mwh_{case}": pue * 1000.0,
            f"heat_mwhth_{case}": heat,
            f"flex_load_mwh_{case}": 20.0,
            f"land_utilization_pct_{case}": 72.0,
            f"K_HX_per_MW_IT_{case}": 0.20 if case == "baseline" else 0.32,
            f"K_HP_per_MW_IT_{case}": 0.08 if case == "baseline" else 0.20,
            f"K_FLEX_per_MW_IT_{case}": 0.05 if case == "baseline" else 0.11,
            f"K_BAT_E_per_MW_IT_{case}": 0.60 if case == "baseline" else 0.90,
            f"K_BAT_P_per_MW_IT_{case}": 0.15 if case == "baseline" else 0.20,
            f"power_emissions_tco2_{case}": 39000.0,
            f"dc_power_emissions_tco2_{case}": 5000.0,
            f"avoided_heat_emissions_tco2_{case}": 120.0,
            f"carbon_market_cost_eur_{case}": 300000.0,
            f"carbon_damage_eur_{case}": 480000.0,
            f"system_benefit_per_mwh_it_{case}": 4.0 if case == "baseline" else 11.0,
            f"emissions_tco2_{case}": 39000.0,
            f"sipue_credit_loss_pct_{case}": 3.0,
            f"sipue_credit_loss_mwh_{case}": 40.0,
        })
    return row


def test_creative_figure_package_generates_mixed_audience_set(tmp_path):
    cfg = replace(
        Config(enforce_locked_calibration=False),
        export_plots=True,
        plot_output_dir=str(tmp_path),
        canonical_output_dir=str(tmp_path),
        validation_output_dir=str(tmp_path / "validation"),
        plot_format="png",
    )
    rows = [
        _creative_row("AI/HPC", "ai_hpc", 1),
        _creative_row("Hyperscale cloud", "hyperscale", 2),
    ]
    decomposition = [
        {
            "design": "robust",
            "ledger": "market",
            "component": component,
            "value_eur": value,
        }
        for component, value in (
            ("heating revenue", 1000000.0),
            ("flexibility revenue", 500000.0),
            ("resource electricity", -300000.0),
            ("cooling electricity", -200000.0),
            ("CO2 market", -100000.0),
            ("capex", -900000.0),
            ("land", -50000.0),
            ("construction delay", -75000.0),
        )
    ]
    export_creative_figure_package(rows, [], decomposition, cfg)
    expected = {
        "final_01_optimization_formulation_map.png",
        "final_03_representative_day_weights.png",
        "final_04_representative_day_load_trace.png",
        "final_05_load_and_renewables.png",
        "final_06_kpi_pue.png",
        "final_07_kpi_sipue.png",
        "final_08_kpi_system_benefit_per_mwh_it.png",
        "final_09_pue_sipue_tradeoff_compass.png",
        "final_10_design_portfolio_parallel.png",
        "final_11_system_value_sankey.png",
        "final_13_carbon_ledger_power_emissions.png",
        "final_14_carbon_ledger_avoided_heat_emissions.png",
        "final_15_delay_credit_bridge.png",
        "final_17_train_test_regret_distribution.png",
        "final_18_out_of_sample_regret_distribution.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_sensitivity_plots_skip_when_sensitivity_output_missing(tmp_path):
    cfg = replace(
        Config(enforce_locked_calibration=False),
        export_plots=True,
        plot_output_dir=str(tmp_path),
        canonical_output_dir=str(tmp_path),
        plot_format="png",
    )
    export_sensitivity_plots(cfg)
    assert not any(path.name.startswith("final_19_") or path.name.startswith("final_20_") for path in tmp_path.iterdir())
