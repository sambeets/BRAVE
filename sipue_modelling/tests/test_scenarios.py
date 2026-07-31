import pytest
from main import Config, Scenario, make_scenarios, apply_global_heat_assumptions
from brave.scenario import DATA_CENTER_ARCHETYPES, make_study_scenarios

pytestmark = pytest.mark.component


class TestMakeScenarios:
    def test_returns_10(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        assert len(scenarios) == 10

    def test_unique_sids(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        sids = [sc.sid for sc in scenarios]
        assert len(set(sids)) == 10
        assert sids == list(range(1, 11))

    def test_dc_types_present(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        types = {sc.dc_type for sc in scenarios}
        assert types == {"hyperscale", "ai", "streaming", "colocation"}

    def test_locations(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        locations = {sc.location for sc in scenarios}
        assert "Norway" in locations
        assert "Netherlands" in locations
        assert "United Kingdom" in locations
        assert "Germany" in locations

    def test_positive_fields(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        for sc in scenarios:
            assert sc.base_it_mw > 0
            assert sc.base_pue >= 1.0
            assert sc.hp_cop > 0
            assert sc.el_price_level > 0
            assert sc.heat_price_level > 0
            assert sc.carbon_price > 0

    def test_scenario_names_unique(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        names = [sc.name for sc in scenarios]
        assert len(set(names)) == 10

    def test_first_scenario_is_hyperscale_netherlands(self):
        cfg = Config(enforce_locked_calibration=False)
        sc = make_scenarios(cfg)[0]
        assert sc.sid == 1
        assert sc.dc_type == "hyperscale"
        assert sc.location == "Netherlands"
        assert sc.base_it_mw == 40


class TestApplyGlobalHeatAssumptions:
    def test_caps_recovery_at_050(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios = make_scenarios(cfg)
        apply_global_heat_assumptions(scenarios, cfg)
        for sc in scenarios:
            assert sc.heat_recovery_frac <= 0.50

    def test_flex_frac_scaled_by_090(self):
        cfg = Config(enforce_locked_calibration=False)
        scenarios_orig = make_scenarios(cfg)
        original_flex = [sc.flex_frac for sc in scenarios_orig]
        apply_global_heat_assumptions(scenarios_orig, cfg)
        for sc, orig in zip(scenarios_orig, original_flex):
            assert sc.flex_frac == pytest.approx(orig * 0.90)

    def test_override_heat_recovery(self):
        cfg = Config(
            enforce_locked_calibration=False,
            override_heat_recovery_frac=0.30,
        )
        scenarios = make_scenarios(cfg)
        apply_global_heat_assumptions(scenarios, cfg)
        for sc in scenarios:
            assert sc.heat_recovery_frac == pytest.approx(0.30)

    def test_override_hp_cop(self):
        cfg = Config(
            enforce_locked_calibration=False,
            override_hp_cop=4.0,
        )
        scenarios = make_scenarios(cfg)
        apply_global_heat_assumptions(scenarios, cfg)
        for sc in scenarios:
            assert sc.hp_cop == pytest.approx(4.0)


class TestPublicationStudyScenarios:
    def test_four_archetypes_share_ten_uncertainty_states(self):
        study = make_study_scenarios(Config(enforce_locked_calibration=False))
        assert set(study) == {item.key for item in DATA_CENTER_ARCHETYPES}
        assert all(len(group) == 10 for group in study.values())
        assert all({sc.uncertainty_id for sc in group} == set(range(1, 11)) for group in study.values())

    def test_archetype_sizes_match_study_design(self):
        study = make_study_scenarios(Config(enforce_locked_calibration=False))
        assert {sc.base_it_mw for sc in study["ai_hpc"]} == {40.0}
        assert {sc.base_it_mw for sc in study["hyperscale"]} == {60.0}
        assert {sc.base_it_mw for sc in study["colocation"]} == {25.0}
        assert {sc.base_it_mw for sc in study["enterprise_edge"]} == {8.0}

    def test_base_heat_inputs_are_separate_from_state_adjusted_values(self):
        bases = {item.key: item.heat_recovery_frac for item in DATA_CENTER_ARCHETYPES}
        assert bases["ai_hpc"] == pytest.approx(0.50)
        assert bases["hyperscale"] == pytest.approx(0.45)
        assert bases["colocation"] == pytest.approx(0.32)
        assert bases["enterprise_edge"] == pytest.approx(0.20)

        study = make_study_scenarios(Config(enforce_locked_calibration=False))
        assert max(sc.heat_recovery_frac for sc in study["ai_hpc"]) == pytest.approx(0.575)
        assert max(sc.heat_recovery_frac for sc in study["hyperscale"]) == pytest.approx(0.5175)
        assert max(sc.heat_recovery_frac for sc in study["colocation"]) <= 0.50
        assert max(sc.heat_recovery_frac for sc in study["enterprise_edge"]) <= 0.50
