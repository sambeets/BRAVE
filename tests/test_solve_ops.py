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

    k_hx, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(sc1, prof, cfg_test)
    ops = solve_ops_given_design(sc1, prof, cfg_test, k_hx, k_flex, k_bat_e, k_bat_p, eff)
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

    def test_output_keys_present(self, ops_result_24h):
        required = {
            "status", "total_cost", "total_profit", "E_IT", "E_FAC", "E_GRID",
            "PUE", "siPUE", "credit_eur", "heat_rev_eur", "flex_rev_eur",
            "land_m2", "land_cost_eur", "heat_MWhth",
        }
        assert required.issubset(ops_result_24h.keys())

    def test_golden_values(self, ops_result_24h):
        golden_path = os.path.join(GOLDEN_DIR, "solve_ops_scenario1_24h.json")
        if not os.path.exists(golden_path):
            pytest.skip("Golden file not yet created. Run: python tests/capture_golden.py")
        with open(golden_path) as f:
            golden = json.load(f)
        for key in ["PUE", "siPUE", "E_IT", "E_FAC"]:
            assert ops_result_24h[key] == pytest.approx(golden[key], rel=1e-3), \
                f"Mismatch in {key}: got {ops_result_24h[key]}, expected {golden[key]}"
        for key in ["total_profit", "heat_rev_eur", "flex_rev_eur"]:
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
