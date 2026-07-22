from dataclasses import replace

import pulp
import pytest

from brave.config import Config
from brave.network import calibrate_no_dc_network, solve_no_dc_network
from brave.optimization import _extract_constraint_duals
from brave.runners import _no_services_ops, _prepare_profiles
from brave.scenario import DATA_CENTER_ARCHETYPES, apply_global_heat_assumptions, make_archetype_scenarios
from brave.solver import make_solver_lp
from brave.timeseries import make_hourly_profiles


pytestmark = pytest.mark.component


def test_no_dc_calibration_is_physical_and_excludes_load_shedding():
    cfg = Config(
        T=24,
        use_representative_days_annual=False,
        enforce_locked_calibration=False,
    )
    scenario = make_archetype_scenarios(cfg, DATA_CENTER_ARCHETYPES[0])[0]
    profile = make_hourly_profiles(scenario, cfg.T, cfg)
    calibration = calibrate_no_dc_network(profile, cfg)
    assert calibration["reference_price_eur_per_mwh"] > 0.0
    assert calibration["load_shedding_mwh"] == pytest.approx(0.0)
    assert calibration["congestion_fraction"] >= cfg.congestion_target_min_fraction
    assert calibration["congestion_fraction"] <= cfg.congestion_target_max_fraction + 1.0 / cfg.T


def test_three_bus_flows_obey_dc_angle_cycle():
    cfg = Config(T=24, use_representative_days_annual=False, enforce_locked_calibration=False)
    scenario = make_archetype_scenarios(cfg, DATA_CENTER_ARCHETYPES[0])[0]
    profile = make_hourly_profiles(scenario, cfg.T, cfg)
    calibration = calibrate_no_dc_network(profile, cfg)
    dispatch = solve_no_dc_network(
        profile,
        cfg,
        (
            calibration["line_cap_12_mw"],
            calibration["line_cap_23_mw"],
            calibration["line_cap_13_mw"],
        ),
    )
    flows = dispatch["flows"]
    for t in range(cfg.T):
        lhs = (
            flows["12"][t] / cfg.line_susceptance_12_mw_per_rad
            + flows["23"][t] / cfg.line_susceptance_23_mw_per_rad
        )
        rhs = flows["13"][t] / cfg.line_susceptance_13_mw_per_rad
        assert lhs == pytest.approx(rhs, abs=1e-7)


def test_highs_native_dual_extraction_matches_cbc_on_non_degenerate_lp():
    duals = {}
    for solver_name in ("highs", "cbc"):
        model = pulp.LpProblem(f"dual-{solver_name}", pulp.LpMinimize)
        x = pulp.LpVariable(f"x-{solver_name}", lowBound=0.0)
        binding = x >= 2.0
        model += binding, "binding"
        model += x
        model.solve(make_solver_lp(Config(solver_lp=solver_name, solver_msg=False)))
        duals[solver_name] = _extract_constraint_duals(model)[id(binding)]
    assert duals["highs"] == pytest.approx(1.0)
    assert duals["highs"] == pytest.approx(duals["cbc"])


def test_binding_congested_network_has_positive_directional_marginal_value():
    cfg = Config(
        T=24,
        use_representative_days_annual=False,
        enforce_locked_calibration=False,
        export_csv=False,
        export_plots=False,
    )
    scenarios = make_archetype_scenarios(cfg, DATA_CENTER_ARCHETYPES[0])
    apply_global_heat_assumptions(scenarios, cfg)
    calibrated, profiles = _prepare_profiles(cfg, scenarios)
    ops = _no_services_ops(scenarios[0], profiles[scenarios[0].sid], calibrated)
    assert ops["congestion_hours"] > 0.0
    assert ops["congestion_marginal_value_eur_per_1pct"] > 0.0

    # The derivative scaled to 1% should bound the explicit 1% value because
    # the marginal congestion benefit decreases as constraints relax.
    one_percent = replace(
        calibrated,
        auto_calibrate_line_limits=False,
        auto_calibrate_sipue_reference_price=False,
        line_cap_12=calibrated.line_cap_12 * 1.01,
        line_cap_23=calibrated.line_cap_23 * 1.01,
        line_cap_13=calibrated.line_cap_13 * 1.01,
    )
    relaxed = _no_services_ops(scenarios[0], profiles[scenarios[0].sid], one_percent)
    explicit_value = ops["market_planning_cost_npv_eur"] - relaxed["market_planning_cost_npv_eur"]
    assert explicit_value > 0.0
    assert explicit_value <= ops["congestion_marginal_value_eur_per_1pct"] + 1e-5
