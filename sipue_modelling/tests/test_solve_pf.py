import json
import os
import random
import pytest
from main import (
    Config, Scenario, make_scenarios, apply_global_heat_assumptions,
    make_hourly_profiles, apply_services_mode,
    solve_perfect_foresight, _ensure_network_defaults,
)

pytestmark = [pytest.mark.integration]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture
def pf_result_24h(cfg_test):
    """Solve perfect foresight for scenario 1 at T=24."""
    _ensure_network_defaults(cfg_test)
    all_sc = make_scenarios(cfg_test)
    apply_global_heat_assumptions(all_sc, cfg_test)
    sc1 = all_sc[0]

    random.seed(cfg_test.seed)
    prof = make_hourly_profiles(sc1, cfg_test.T, cfg_test)
    prof = apply_services_mode(sc1, prof, cfg_test)

    cstar, design = solve_perfect_foresight(sc1, prof, cfg_test)
    return cstar, design


class TestSolvePerfectForesight:
    def test_status_optimal(self, pf_result_24h):
        _, design = pf_result_24h
        assert design["status"] == "Optimal"

    def test_design_values_nonnegative(self, pf_result_24h):
        _, design = pf_result_24h
        assert design["K_HX"] >= -1e-6
        assert design["K_HP"] >= -1e-6
        assert design["K_FLEX"] >= -1e-6
        assert design["K_BAT_E"] >= -1e-6
        assert design["K_BAT_P"] >= -1e-6

    def test_design_within_bounds(self, pf_result_24h, cfg_test):
        _, design = pf_result_24h
        assert design["K_HX"] <= cfg_test.K_HX_max + 1e-6
        assert design["K_HP"] <= cfg_test.K_HP_max + 1e-6
        assert design["K_FLEX"] <= cfg_test.K_FLEX_max + 1e-6
        assert design["K_BAT_E"] <= cfg_test.K_BAT_E_max + 1e-6
        assert design["K_BAT_P"] <= cfg_test.K_BAT_P_max + 1e-6

    def test_cstar_is_finite(self, pf_result_24h):
        cstar, _ = pf_result_24h
        assert isinstance(cstar, float)
        assert cstar != float("inf")
        assert cstar != float("-inf")

    def test_golden_values(self, pf_result_24h):
        golden_path = os.path.join(GOLDEN_DIR, "solve_pf_scenario1_24h.json")
        if not os.path.exists(golden_path):
            pytest.skip("Golden file not yet created. Run: python tests/capture_golden.py")
        with open(golden_path) as f:
            golden = json.load(f)
        cstar, design = pf_result_24h
        assert cstar == pytest.approx(golden["cstar"], rel=1e-3)
        for key in ["K_HX", "K_HP", "K_FLEX", "K_BAT_E", "K_BAT_P"]:
            if key in golden:
                assert design[key] == pytest.approx(golden[key], rel=1e-3), \
                    f"Mismatch in {key}: got {design[key]}, expected {golden[key]}"
