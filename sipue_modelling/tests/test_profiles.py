import random
import pytest
from main import (
    Config, Scenario, make_hourly_profiles, _ensure_network_defaults,
    build_representative_days, load_network_timeseries,
)
from brave.scenario import DATA_CENTER_ARCHETYPES, make_archetype_scenarios

pytestmark = pytest.mark.component


class TestMakeHourlyProfiles:
    def test_profile_length(self, cfg_test, scenario_hyperscale):
        random.seed(42)
        _ensure_network_defaults(cfg_test)
        prof = make_hourly_profiles(scenario_hyperscale, 24, cfg_test)
        assert len(prof["it_base"]) == 24
        assert len(prof["p_heat"]) == 24
        assert len(prof["dh_cap"]) == 24
        assert len(prof["p_flex"]) == 24
        assert len(prof["cop_t"]) == 24

    def test_profile_keys(self, cfg_test, scenario_hyperscale):
        random.seed(42)
        _ensure_network_defaults(cfg_test)
        prof = make_hourly_profiles(scenario_hyperscale, 24, cfg_test)
        required_keys = {
            "it_base", "it_demand", "p_el", "p_heat", "dh_cap",
            "p_flex", "cop_t", "dc_node",
            "carbon_factor", "p_co2",
        }
        assert required_keys.issubset(prof.keys())

    def test_it_base_positive(self, cfg_test, scenario_hyperscale):
        random.seed(42)
        _ensure_network_defaults(cfg_test)
        prof = make_hourly_profiles(scenario_hyperscale, 24, cfg_test)
        assert all(v > 0 for v in prof["it_base"])

    def test_deterministic_with_seed(self, cfg_test, scenario_hyperscale):
        _ensure_network_defaults(cfg_test)
        random.seed(42)
        prof1 = make_hourly_profiles(scenario_hyperscale, 24, cfg_test)
        random.seed(42)
        prof2 = make_hourly_profiles(scenario_hyperscale, 24, cfg_test)
        assert prof1["it_base"] == prof2["it_base"]
        assert prof1["p_flex"] == prof2["p_flex"]
        assert prof1["p_heat"] == prof2["p_heat"]

    def test_dc_node_assignment_norway(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        sc_no = Scenario(
            sid=99, name="test", dc_type="ai", location="Norway",
            base_it_mw=10, flex_frac=0.10, base_pue=1.12,
            heat_recovery_frac=0.50, hp_cop=3.6,
            el_price_level=42, el_price_vol=15,
            heat_price_level=30, heat_demand_level=45,
            flex_price_level=10, carbon_factor=0.029, carbon_price=65,
        )
        random.seed(42)
        prof = make_hourly_profiles(sc_no, 24, cfg_test)
        assert prof["dc_node"] == "N1"

    def test_dc_node_assignment_germany(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        sc = Scenario(
            sid=99, name="test", dc_type="colocation", location="Germany",
            base_it_mw=10, flex_frac=0.10, base_pue=1.35,
            heat_recovery_frac=0.35, hp_cop=3.1,
            el_price_level=62, el_price_vol=22,
            heat_price_level=30, heat_demand_level=42,
            flex_price_level=10, carbon_factor=0.385, carbon_price=65,
        )
        random.seed(42)
        prof = make_hourly_profiles(sc, 24, cfg_test)
        assert prof["dc_node"] == "N2"

    def test_dc_node_assignment_uk(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        sc = Scenario(
            sid=99, name="test", dc_type="streaming", location="United Kingdom",
            base_it_mw=10, flex_frac=0.10, base_pue=1.35,
            heat_recovery_frac=0.30, hp_cop=2.8,
            el_price_level=82, el_price_vol=30,
            heat_price_level=30, heat_demand_level=20,
            flex_price_level=10, carbon_factor=0.207, carbon_price=65,
        )
        random.seed(42)
        prof = make_hourly_profiles(sc, 24, cfg_test)
        assert prof["dc_node"] == "N3"

    def test_uncertainty_states_set_fuel_and_ets_price_moments(self, cfg_test):
        cfg = Config(
            T=24,
            enforce_locked_calibration=False,
            network_timeseries_csv=cfg_test.network_timeseries_csv,
        )
        scenarios = make_archetype_scenarios(cfg, DATA_CENTER_ARCHETYPES[0])
        low = make_hourly_profiles(scenarios[1], cfg.T, cfg)
        high = make_hourly_profiles(scenarios[2], cfg.T, cfg)
        low_ets = sum(low["p_co2"]) / cfg.T
        high_ets = sum(high["p_co2"]) / cfg.T
        assert low_ets == pytest.approx(scenarios[1].carbon_price)
        assert high_ets == pytest.approx(scenarios[2].carbon_price)
        assert high_ets > low_ets

        low_n2_mc = sum(
            fuel / cfg.th_eff_N2 + cfg.th_vom_N2 for fuel in low["fuel_price"]
        ) / cfg.T
        high_n2_mc = sum(
            fuel / cfg.th_eff_N2 + cfg.th_vom_N2 for fuel in high["fuel_price"]
        ) / cfg.T
        assert low_n2_mc == pytest.approx(scenarios[1].el_price_level, abs=1e-3)
        assert high_n2_mc == pytest.approx(scenarios[2].el_price_level, abs=1e-3)
        assert high_n2_mc > low_n2_mc


class TestBuildRepresentativeDays:
    def test_7_rep_days(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        ts = load_network_timeseries(cfg_test, T=8760)
        cfg = Config(
            enforce_locked_calibration=False,
            network_timeseries_csv=cfg_test.network_timeseries_csv,
            representative_day_count=7,
        )
        indices, weights, starts, total_days = build_representative_days(ts, cfg)
        assert len(starts) == cfg.representative_day_count
        assert len(indices) == cfg.representative_day_count * 24
        assert len(weights) == cfg.representative_day_count * 24
        assert total_days == 365

    def test_weights_sum(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        ts = load_network_timeseries(cfg_test, T=8760)
        cfg = Config(
            enforce_locked_calibration=False,
            network_timeseries_csv=cfg_test.network_timeseries_csv,
            representative_day_count=7,
        )
        indices, weights, starts, total_days = build_representative_days(ts, cfg)
        day_weights = [weights[s] for s in starts]
        assert sum(day_weights) == pytest.approx(total_days, rel=1e-9)

    def test_selection_is_deterministic(self, cfg_test):
        _ensure_network_defaults(cfg_test)
        ts = load_network_timeseries(cfg_test, T=8760)
        cfg = Config(
            enforce_locked_calibration=False,
            network_timeseries_csv=cfg_test.network_timeseries_csv,
            representative_day_count=7,
        )
        indices_a, weights_a, starts_a, total_days_a = build_representative_days(ts, cfg)
        indices_b, weights_b, starts_b, total_days_b = build_representative_days(ts, cfg)
        assert indices_a == indices_b
        assert weights_a == weights_b
        assert starts_a == starts_b
        assert total_days_a == total_days_b
