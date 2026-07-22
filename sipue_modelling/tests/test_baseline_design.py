import pytest
from main import (
    Config, Scenario, baseline_design_for_scenario, compute_land_m2,
)

pytestmark = pytest.mark.unit


class TestHyperscaleBaseline:
    def test_hyperscale_values(self, cfg_test, scenario_hyperscale):
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(
            scenario_hyperscale, prof, cfg_test
        )
        # hyperscale type_scales are all 1.0
        assert k_hx == pytest.approx(cfg_test.baseline_hx_mwth_per_it_mw * 40.0)
        assert k_hp == pytest.approx(k_hx / (scenario_hyperscale.hp_cop - 1.0))
        assert k_flex == pytest.approx(cfg_test.baseline_flex_mw_per_it_mw * 40.0)
        assert k_bat_e == pytest.approx(cfg_test.baseline_bat_e_hours * 40.0)
        assert k_bat_p == pytest.approx(cfg_test.baseline_bat_p_mw_per_it_mw * 40.0)
        assert eff == pytest.approx(0.15)

    def test_returns_six_values(self, cfg_test, scenario_hyperscale):
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        result = baseline_design_for_scenario(scenario_hyperscale, prof, cfg_test)
        assert len(result) == 6


class TestAiBaseline:
    def test_ai_flex_scale_is_recalibrated(self, cfg_test, scenario_nordic):
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        _, _, k_flex, _, _, _ = baseline_design_for_scenario(scenario_nordic, prof, cfg_test)
        assert k_flex == pytest.approx(cfg_test.baseline_flex_mw_per_it_mw * 1.50 * 25.0)


@pytest.mark.parametrize("dc_type, hx_scale", [
    ("hyperscale", 1.00),
    ("streaming", 0.70),
    ("ai", 0.55),
    ("colocation", 0.40),
])
class TestTypeScales:
    def test_hx_scaling(self, cfg_test, dc_type, hx_scale):
        sc = Scenario(
            sid=99, name="test", dc_type=dc_type, location="Netherlands",
            base_it_mw=10, flex_frac=0.15, base_pue=1.18,
            heat_recovery_frac=0.45, hp_cop=3.5,
            el_price_level=58, el_price_vol=20,
            heat_price_level=30, heat_demand_level=65,
            flex_price_level=14, carbon_factor=0.3, carbon_price=65,
            land_limit_m2=5000,  # Large enough to not constrain
        )
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        k_hx, _, _, _, _, _ = baseline_design_for_scenario(sc, prof, cfg_test)
        assert k_hx == pytest.approx(cfg_test.baseline_hx_mwth_per_it_mw * hx_scale * 10.0)


class TestDisabledServices:
    def test_heat_disabled(self, cfg_test, scenario_hyperscale):
        prof = {"allow_heat": False, "allow_flex": True, "allow_battery": True}
        k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(
            scenario_hyperscale, prof, cfg_test
        )
        assert k_hx == 0.0
        assert k_hp == 0.0
        assert k_flex > 0.0

    def test_flex_disabled(self, cfg_test, scenario_hyperscale):
        prof = {"allow_heat": True, "allow_flex": False, "allow_battery": True}
        k_hx, _, k_flex, _, _, _ = baseline_design_for_scenario(
            scenario_hyperscale, prof, cfg_test
        )
        assert k_hx > 0.0
        assert k_flex == 0.0

    def test_battery_disabled(self, cfg_test, scenario_hyperscale):
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": False}
        _, _, _, k_bat_e, k_bat_p, _ = baseline_design_for_scenario(
            scenario_hyperscale, prof, cfg_test
        )
        assert k_bat_e == 0.0
        assert k_bat_p == 0.0


class TestLandLimitScaling:
    def test_tight_land_limit_scales_down(self, cfg_test):
        sc = Scenario(
            sid=1, name="test", dc_type="hyperscale", location="Netherlands",
            base_it_mw=40, flex_frac=0.15, base_pue=1.18,
            heat_recovery_frac=0.45, hp_cop=3.5,
            el_price_level=58, el_price_vol=20,
            heat_price_level=30, heat_demand_level=65,
            flex_price_level=14, carbon_factor=0.3, carbon_price=65,
            land_limit_m2=500,  # Tight after existing core and cooling footprint.
        )
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(sc, prof, cfg_test)
        land = compute_land_m2(
            k_hx, k_flex, k_bat_e, k_bat_p, eff, cfg_test,
            K_HP=k_hp, base_it_mw=sc.base_it_mw, K_COOL=(sc.base_pue - 1.0) * sc.base_it_mw,
        )
        assert land <= 500.0 + 0.01

    def test_no_land_limit(self, cfg_test):
        sc = Scenario(
            sid=1, name="test", dc_type="hyperscale", location="Netherlands",
            base_it_mw=40, flex_frac=0.15, base_pue=1.18,
            heat_recovery_frac=0.45, hp_cop=3.5,
            el_price_level=58, el_price_vol=20,
            heat_price_level=30, heat_demand_level=65,
            flex_price_level=14, carbon_factor=0.3, carbon_price=65,
            land_limit_m2=None,
        )
        cfg_test_no_land = Config(
            enforce_locked_calibration=False, T=24, land_limit_m2=None,
        )
        prof = {"allow_heat": True, "allow_flex": True, "allow_battery": True}
        k_hx, _, _, _, _, _ = baseline_design_for_scenario(sc, prof, cfg_test_no_land)
        # Should use full baseline without scaling
        assert k_hx == pytest.approx(cfg_test_no_land.baseline_hx_mwth_per_it_mw * 40.0)
