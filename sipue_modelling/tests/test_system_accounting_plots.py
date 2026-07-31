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
    # Figures that used to be multi-panel subplot grids are now exported as
    # separate single-axis files (one per panel), so each subfigure name is
    # asserted individually rather than one combined filename per figure.
    expected = {
        "fig_01_sipue_metric_story.png",
        "fig_01b_why_sipue_matters.png",
        "fig_02_model_pipeline_subway.png",
        "fig_02b_full_study_roadmap.png",
        "fig_03_pue_sipue_tradeoff_compass.png",
        "fig_04_kpi_pue.png",
        "fig_04_kpi_sipue.png",
        "fig_04_kpi_cue.png",
        "fig_04_kpi_modeled_erf.png",
        "fig_04_kpi_system_benefit_per_mwh_it.png",
        "fig_04_kpi_emissions_tco2.png",
        "fig_04_kpi_carbon_damage_eur.png",
        "fig_05_optimization_formulation_map.png",
        "fig_06_input_scenario_fingerprint.png",
        "fig_07a_representative_day_weights.png",
        "fig_07b_representative_day_load_trace.png",
        "fig_08a_network_topology.png",
        "fig_08b_load_and_renewables.png",
        "fig_08c_carbon_cap_by_archetype.png",
        "fig_09_design_portfolio_parallel.png",
        "fig_10_system_value_sankey.png",
        "fig_11_carbon_ledger_power_emissions.png",
        "fig_11_carbon_ledger_avoided_heat_emissions.png",
        "fig_11_carbon_ledger_co2_cost_change.png",
        "fig_11_carbon_ledger_carbon_damage_change.png",
        "fig_12_delay_credit_bridge.png",
        "fig_13_property_check_matrix.png",
        "fig_14a_train_test_feasibility.png",
        "fig_14b_train_test_regret.png",
        "fig_15a_out_of_sample_feasibility.png",
        "fig_15b_out_of_sample_regret.png",
        "fig_16_provenance_map.png",
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
    assert not any(path.name.startswith("fig_17_") or path.name.startswith("fig_18_") for path in tmp_path.iterdir())
