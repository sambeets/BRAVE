import json
import os
import random
import pytest
from main import (
    Config, Scenario, make_scenarios, apply_global_heat_assumptions,
    make_hourly_profiles, apply_services_mode,
    solve_ops_given_design, baseline_design_for_scenario,
    _ensure_network_defaults,
)

pytestmark = [pytest.mark.integration]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture
def ops_result_24h(cfg_test):
    """Solve operations for scenario 1 with baseline design at T=24."""
    _ensure_network_defaults(cfg_test)
    all_sc = make_scenarios(cfg_test)
    apply_global_heat_assumptions(all_sc, cfg_test)
    sc1 = all_sc[0]

    random.seed(cfg_test.seed)
    prof = make_hourly_profiles(sc1, cfg_test.T, cfg_test)
    prof = apply_services_mode(sc1, prof, cfg_test)

    k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(sc1, prof, cfg_test)
    ops = solve_ops_given_design(sc1, prof, cfg_test, k_hx, k_flex, k_bat_e, k_bat_p, eff, K_HP=k_hp)
    return ops


class TestSolveOpsGivenDesign:
    def test_status_optimal(self, ops_result_24h):
        assert ops_result_24h["status"] == "Optimal"

    def test_pue_reasonable(self, ops_result_24h):
        pue = ops_result_24h["PUE"]
        assert 1.0 <= pue <= 3.0

    def test_sipue_reasonable(self, ops_result_24h):
        sipue = ops_result_24h["siPUE"]
        assert 0.0 <= sipue <= 3.0

    def test_energy_consistency(self, ops_result_24h):
        assert ops_result_24h["E_FAC"] >= ops_result_24h["E_IT"] - 1e-6

    def test_total_system_carbon_reconciles(self, ops_result_24h, cfg_test):
        assert ops_result_24h["emissions_tco2"] == pytest.approx(
            ops_result_24h["power_emissions_tco2"]
            + ops_result_24h["baseline_heat_emissions_tco2"]
            - ops_result_24h["avoided_heat_emissions_tco2"]
        )
        assert ops_result_24h["co2_cost_eur"] == pytest.approx(
            ops_result_24h["power_co2_cost_eur"]
            + ops_result_24h["baseline_heat_co2_cost_eur"]
            - ops_result_24h["avoided_heat_co2_cost_eur"]
        )
        assert ops_result_24h["independent_carbon_damage_eur"] == pytest.approx(
            cfg_test.independent_social_carbon_cost_eur_per_tco2
            * ops_result_24h["emissions_tco2"]
        )

    def test_market_and_independent_ledgers_reconcile_separately(self, ops_result_24h):
        market = (
            ops_result_24h["el_cost_eur"]
            + ops_result_24h["baseline_heat_resource_cost_eur"]
            + ops_result_24h["trans_cost_eur"]
            + ops_result_24h["co2_cost_eur"]
            + ops_result_24h["demand_charge_eur"]
            + ops_result_24h["sla_penalty_eur"]
            + ops_result_24h["capex_eur"]
            + ops_result_24h["eff_capex_eur"]
            + ops_result_24h["land_cost_eur"]
            - ops_result_24h["heat_revenue_eur"]
            - ops_result_24h["flex_revenue_eur"]
            - ops_result_24h["eff_benefit_eur"]
        )
        independent = (
            ops_result_24h["el_cost_eur"]
            + ops_result_24h["baseline_heat_resource_cost_eur"]
            + ops_result_24h["trans_cost_eur"]
            + ops_result_24h["independent_carbon_damage_eur"]
            + ops_result_24h["independent_peak_capacity_cost_eur"]
            - ops_result_24h["independent_avoided_heat_cost_eur"]
            - ops_result_24h["independent_avoided_flex_cost_eur"]
        )
        assert ops_result_24h["market_planning_cost_eur"] == pytest.approx(market)
        assert ops_result_24h["independent_system_cost_eur"] == pytest.approx(independent)

    def test_output_keys_present(self, ops_result_24h):
        required = {
            "status", "E_IT", "E_FAC", "E_GRID",
            "planning_objective_cost_eur", "market_planning_cost_npv_eur",
            "market_planning_value_npv_eur", "annual_market_operating_cost_eur",
            "PUE", "siPUE", "system_credit_mwh",
            "land_m2", "land_cost_eur", "heat_MWhth",
            "independent_system_cost_eur", "independent_avoided_heat_cost_eur",
            "independent_avoided_flex_cost_eur",
            "CUE_tCO2_per_MWh_IT", "emissions_tco2",
            "K_HP", "hp_electric_mw_peak", "hp_electric_mwh",
            "flex_load_mwh", "flex_bat_mwh", "co2_cost_eur",
            "independent_carbon_damage_eur", "cooling_mwh",
            "recovered_heat_it_mwhth", "recovered_heat_cooling_mwhth",
            "storage_loss_mwh",
        }
        assert required.issubset(ops_result_24h.keys())
        banned = ("operator", "settlement", "unitprofit", "total_profit")
        assert not any(any(token in key.lower() for token in banned) for key in ops_result_24h)

    def test_golden_values(self, ops_result_24h):
        golden_path = os.path.join(GOLDEN_DIR, "solve_ops_scenario1_24h.json")
        if not os.path.exists(golden_path):
            pytest.skip("Golden file not yet created. Run: python tests/capture_golden.py")
        with open(golden_path) as f:
            golden = json.load(f)
        for key in ["PUE", "siPUE", "E_IT", "E_FAC"]:
            assert ops_result_24h[key] == pytest.approx(golden[key], rel=1e-3), \
                f"Mismatch in {key}: got {ops_result_24h[key]}, expected {golden[key]}"
        for key in ["planning_objective_cost_eur", "market_planning_cost_npv_eur"]:
            if key in golden:
                assert ops_result_24h[key] == pytest.approx(golden[key], rel=1e-3), \
                    f"Mismatch in {key}"


class TestSolveOpsZeroCapacities:
    def test_zero_capacities_still_optimal(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        all_sc = make_scenarios(cfg_test)
        apply_global_heat_assumptions(all_sc, cfg_test)
        sc1 = all_sc[0]

        random.seed(cfg_test.seed)
        prof = make_hourly_profiles(sc1, cfg_test.T, cfg_test)
        prof = apply_services_mode(sc1, prof, cfg_test)

        ops = solve_ops_given_design(
            sc1, prof, cfg_test,
            K_HX=0.0, K_FLEX=0.0, K_BAT_E=0.0, K_BAT_P=0.0,
            eff_reduction=0.0,
        )
        assert ops["status"] == "Optimal"
        assert ops["heat_MWhth"] == pytest.approx(0.0, abs=0.01)
