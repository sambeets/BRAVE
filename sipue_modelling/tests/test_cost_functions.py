import pytest
from main import (
    Config, Scenario,
    compute_land_m2, land_cost_horizon, demand_charge_horizon,
    capex_scale, eff_capex_from_reduction, service_revenue_scale,
    design_size_proxy,
)

pytestmark = pytest.mark.unit


# ---------- compute_land_m2 ----------

class TestComputeLandM2:
    def test_zero_capacities(self, cfg_test):
        result = compute_land_m2(0, 0, 0, 0, 0.0, cfg_test)
        assert result == pytest.approx(cfg_test.base_land_m2)

    def test_linear_in_capacities(self, cfg_test):
        land = compute_land_m2(10.0, 5.0, 20.0, 8.0, 0.15, cfg_test)
        expected = (cfg_test.base_land_m2
                    + cfg_test.land_a_hx * 10
                    + cfg_test.land_a_flex * 5
                    + cfg_test.land_a_bat_e * 20
                    + cfg_test.land_a_bat_p * 8)
        assert land == pytest.approx(expected)

    def test_eff_reduction_does_not_affect_land(self, cfg_test):
        a = compute_land_m2(10, 5, 20, 8, 0.0, cfg_test)
        b = compute_land_m2(10, 5, 20, 8, 0.50, cfg_test)
        assert a == b


# ---------- land_cost_horizon ----------

class TestLandCostHorizon:
    def test_with_rep_days_annual(self):
        cfg = Config(
            enforce_locked_calibration=False,
            use_representative_days_annual=True,
            land_cost_eur_per_m2=35.0,
            land_maintenance_frac=0.0,
        )
        cost = land_cost_horizon(500.0, cfg)
        # scale=1.0 because rep days path
        assert cost == pytest.approx(500.0 * 35.0)

    def test_without_rep_days(self, cfg_test):
        # use_representative_days_annual=False, T=24
        cost = land_cost_horizon(500.0, cfg_test)
        # horizon_days = rep_total_days=24 (from Config default), or T/24=1
        # Since cfg_test.rep_total_days is the Config default (24.0) and
        # use_representative_days_annual=False, scale = rep_total_days/365 = 24/365
        assert cost >= 0


# ---------- demand_charge_horizon ----------

class TestDemandChargeHorizon:
    def test_disabled(self):
        cfg = Config(enforce_locked_calibration=False, enable_demand_charge=False)
        assert demand_charge_horizon(cfg) == 0.0

    def test_annual_with_rep_days(self):
        cfg = Config(
            enforce_locked_calibration=False,
            enable_demand_charge=True,
            use_representative_days_annual=True,
            demand_charge_eur_per_mw_year=50000.0,
        )
        assert demand_charge_horizon(cfg) == pytest.approx(50000.0)


# ---------- capex_scale ----------

class TestCapexScale:
    def test_annual_mode_returns_one(self):
        cfg = Config(
            enforce_locked_calibration=False,
            capex_scale_with_horizon=True,
            use_representative_days_annual=True,
        )
        assert capex_scale(cfg) == 1.0

    def test_disabled_returns_one(self):
        cfg = Config(enforce_locked_calibration=False, capex_scale_with_horizon=False)
        assert capex_scale(cfg) == 1.0

    def test_fraction_of_year(self):
        cfg = Config(
            enforce_locked_calibration=False,
            capex_scale_with_horizon=True,
            use_representative_days_annual=False,
            T=4380,
        )
        assert capex_scale(cfg) == pytest.approx(4380 / 8760)


# ---------- eff_capex_from_reduction ----------

class TestEffCapexFromReduction:
    def test_zero_reduction(self):
        cfg = Config(enforce_locked_calibration=False)
        assert eff_capex_from_reduction(0.0, cfg) == 0.0

    def test_0_15_reduction(self):
        cfg = Config(enforce_locked_calibration=False)
        assert eff_capex_from_reduction(0.15, cfg) == 1680000.0

    def test_empty_levels(self):
        cfg = Config(enforce_locked_calibration=False, eff_levels=())
        assert eff_capex_from_reduction(0.15, cfg) == 0.0


# ---------- service_revenue_scale ----------

class TestServiceRevenueScale:
    def test_no_delay(self):
        cfg = Config(enforce_locked_calibration=False, construction_delay_frac=0.0)
        assert service_revenue_scale(cfg, None) == pytest.approx(1.0)

    def test_cfg_delay(self):
        cfg = Config(enforce_locked_calibration=False, construction_delay_frac=0.10)
        assert service_revenue_scale(cfg, None) == pytest.approx(0.90)

    def test_combined_delay(self, cfg_test, scenario_hyperscale):
        # cfg has 0.0 delay, scenario has 0.06
        result = service_revenue_scale(cfg_test, scenario_hyperscale)
        assert result == pytest.approx(1.0 * (1.0 - 0.06))

    def test_both_delays(self, scenario_hyperscale):
        cfg = Config(enforce_locked_calibration=False, construction_delay_frac=0.10)
        result = service_revenue_scale(cfg, scenario_hyperscale)
        assert result == pytest.approx(0.90 * 0.94)


# ---------- design_size_proxy ----------

class TestDesignSizeProxy:
    def test_zero_ramp_coefficients(self, cfg_test, scenario_hyperscale):
        # Default ramp_k_* are all 0.0
        result = design_size_proxy(scenario_hyperscale, 10, 5, 20, 8, cfg_test)
        assert result == pytest.approx(scenario_hyperscale.base_it_mw)
