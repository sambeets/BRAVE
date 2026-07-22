from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pulp

from brave.config import Config
from brave.scenario import Scenario
from brave.metrics import (
    compute_sipue,
    compute_land_components,
    compute_land_m2,
    effective_land_limit_m2,
    land_cost_horizon,
    demand_charge_horizon,
    compute_independent_system_cost,
    design_size_proxy,
    eff_capex_from_reduction,
    capex_scale,
)
from brave.accounting import (
    build_carbon_ledgers,
    lifecycle_market_value,
    lifecycle_operating_factor,
    present_value_annuity_factor,
)
from brave.solver import make_solver_lp
from brave.timeseries import make_overhead_alpha_base


def _hp_source_to_delivered_ratio(cop: float) -> float:
    """Convert recovered source heat to delivered heat for a heat pump."""
    return max(1.000001, cop) / max(1e-9, max(1.000001, cop) - 1.0)


def _hp_source_to_power_ratio(cop: float) -> float:
    """Convert recovered source heat to heat-pump electric input."""
    return 1.0 / max(1e-9, max(1.000001, cop) - 1.0)


def _commissioning_delay_months(cfg: Config, sc: Scenario) -> float:
    cfg_months = max(0.0, float(getattr(cfg, "construction_delay_months", 0.0)))
    sc_months = max(0.0, float(getattr(sc, "construction_delay_months", 0.0)))
    if cfg_months <= 1e-9:
        cfg_months = 12.0 * max(0.0, float(getattr(cfg, "construction_delay_frac", 0.0)))
    if sc_months <= 1e-9:
        sc_months = 12.0 * max(0.0, float(getattr(sc, "construction_delay_frac", 0.0)))
    return cfg_months + sc_months


def _lifecycle_factors(cfg: Config, sc: Scenario) -> Tuple[float, float]:
    # A full 8,760-hour trace is annual even when representative-day
    # weighting is disabled. Short unweighted horizons remain test/smoke runs.
    if not cfg.use_representative_days_annual and cfg.T != 8760:
        return 1.0, 1.0
    operating = lifecycle_operating_factor(
        cfg.lifecycle_years,
        cfg.real_discount_rate,
        _commissioning_delay_months(cfg, sc),
    )
    land = present_value_annuity_factor(cfg.lifecycle_years, cfg.real_discount_rate)
    return operating, land


def _existing_cooling_capacity_mw(sc: Scenario, alpha_base: List[float]) -> float:
    return max(0.0, sc.base_it_mw * max(alpha_base, default=max(0.0, sc.base_pue - 1.0)))


def _network_line_parameters(sc: Scenario, cfg: Config) -> Tuple[float, float, float, float, float, float]:
    multiplier = max(1e-6, float(getattr(sc, "congestion_multiplier", 1.0)))
    return (
        cfg.line_cap_12 * multiplier,
        cfg.line_cap_23 * multiplier,
        cfg.line_cap_13 * multiplier,
        cfg.line_susceptance_12_mw_per_rad,
        cfg.line_susceptance_23_mw_per_rad,
        cfg.line_susceptance_13_mw_per_rad,
    )


def _extract_constraint_duals(model: pulp.LpProblem) -> Dict[int, float]:
    """Return solver row duals keyed by the corresponding constraint object."""
    constraints = list(model.constraints.values())
    direct = [getattr(constraint, "pi", None) for constraint in constraints]
    if direct and all(value is not None for value in direct):
        return {id(constraint): float(value) for constraint, value in zip(constraints, direct)}

    # PuLP 2.9's HiGHS adapter leaves ``constraint.pi`` unset even though
    # highspy returns row duals. HiGHS builds rows in model constraint order.
    solver_model = getattr(model, "solverModel", None)
    if solver_model is not None and hasattr(solver_model, "getSolution"):
        native = list(getattr(solver_model.getSolution(), "row_dual", []))
        if len(native) == len(constraints):
            return {
                id(constraint): float(value)
                for constraint, value in zip(constraints, native)
            }
    return {}


def baseline_design_for_scenario(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Tuple[float, float, float, float, float, float]:
    """
    Baseline ("incumbent") design, scaled by scenario IT size and DC type.
    Type-specific scaling reflects realistic industry practice:
      - hyperscale: large campuses, invested in heat recovery & flexibility
      - streaming:  moderate service infrastructure
      - ai:         compute-focused, minimal heat recovery investment
      - colocation: tenant-driven, minimal service infrastructure
    Returns: (K_HX, K_HP, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction)
    """
    # Type-specific multipliers on baseline allocations
    type_scales = {
        "hyperscale": {"hx": 1.00, "flex": 1.00, "bat_e": 1.00, "bat_p": 1.00},
        "streaming":  {"hx": 0.70, "flex": 0.85, "bat_e": 0.80, "bat_p": 0.80},
        "ai":         {"hx": 0.55, "flex": 1.50, "bat_e": 0.70, "bat_p": 0.70},
        "colocation": {"hx": 0.40, "flex": 0.55, "bat_e": 0.60, "bat_p": 0.60},
        "enterprise_edge": {"hx": 0.20, "flex": 0.30, "bat_e": 0.35, "bat_p": 0.35},
    }
    ts = type_scales.get(sc.dc_type, {"hx": 1.0, "flex": 1.0, "bat_e": 1.0, "bat_p": 1.0})

    k_hx = max(0.0, cfg.baseline_hx_mwth_per_it_mw * ts["hx"]) * sc.base_it_mw
    k_hp = k_hx * _hp_source_to_power_ratio(sc.hp_cop)
    k_flex = max(0.0, cfg.baseline_flex_mw_per_it_mw * ts["flex"]) * sc.base_it_mw
    k_bat_e = max(0.0, cfg.baseline_bat_e_hours * ts["bat_e"]) * sc.base_it_mw
    k_bat_p = max(0.0, cfg.baseline_bat_p_mw_per_it_mw * ts["bat_p"]) * sc.base_it_mw
    eff_red = max(0.0, min(0.90, float(cfg.baseline_eff_reduction)))

    if not bool(prof.get("allow_heat", True)):
        k_hx = 0.0
        k_hp = 0.0
    if not bool(prof.get("allow_flex", True)):
        k_flex = 0.0
    if not bool(prof.get("allow_battery", True)):
        k_bat_e = 0.0
        k_bat_p = 0.0

    k_hx = min(cfg.K_HX_max, k_hx)
    k_hp = min(cfg.K_HP_max, k_hx * _hp_source_to_power_ratio(sc.hp_cop))
    k_flex = min(cfg.K_FLEX_max, k_flex)
    k_bat_e = min(cfg.K_BAT_E_max, k_bat_e)
    k_bat_p = min(cfg.K_BAT_P_max, k_bat_p)

    # Respect per-scenario land limits by scaling down proportionally
    alpha_base = prof.get("alpha_base")
    if alpha_base is None:
        alpha_base = (
            make_overhead_alpha_base(sc, len(prof["it_base"]), cfg)
            if "it_base" in prof
            else [max(0.0, sc.base_pue - 1.0)]
        )
    cooling_capacity = _existing_cooling_capacity_mw(sc, alpha_base)
    land_limit = effective_land_limit_m2(sc, cfg)
    if land_limit is not None:
        components = compute_land_components(
            k_hx, k_flex, k_bat_e, k_bat_p, eff_red, cfg,
            K_HP=k_hp, base_it_mw=sc.base_it_mw, K_COOL=cooling_capacity,
        )
        land_used = sum(components.values())
        if land_used > land_limit:
            service_land = components["integration_m2"]
            if service_land > 0:
                existing_land = components["existing_core_m2"] + components["existing_cooling_m2"]
                scale = max(0.0, (land_limit - existing_land) / service_land)
                k_hx *= scale
                k_hp *= scale
                k_flex *= scale
                k_bat_e *= scale
                k_bat_p *= scale

    return k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff_red


def solve_ops_given_design(
    sc: Scenario,
    prof: Dict[str, List[float]],
    cfg: Config,
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    eff_reduction: float,  # NEW: chosen efficiency level in [0,1)
    K_HP: Optional[float] = None,
    _compute_congestion_marginal: bool = True,
) -> Dict:
    T = len(prof["it_base"])
    dt = 1.0
    w = prof.get("w", [1.0] * T)
    if len(w) != T:
        raise ValueError("Profile weights length does not match horizon length.")
    rep_day_starts = prof.get("rep_day_starts", [])

    it_base = prof["it_base"]
    it_demand = prof.get("it_demand", it_base)
    p_heat = prof["p_heat"]
    p_flex = prof["p_flex"]
    dh_cap = prof["dh_cap"]
    cf = prof["carbon_factor"]
    p_co2 = prof["p_co2"]
    cop_t = prof.get("cop_t", [sc.hp_cop] * T)
    node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
    dc_node = prof.get("dc_node", cfg.network_nodes[0])
    cap12, cap23, cap13, b12, b23, b13 = _network_line_parameters(sc, cfg)

    allow_heat = bool(prof.get("allow_heat", True))
    allow_flex = bool(prof.get("allow_flex", True))
    allow_battery = bool(prof.get("allow_battery", True))
    allow_it_shortfall = False

    alpha_base = prof.get("alpha_base", None)
    if alpha_base is None:
        alpha_base = make_overhead_alpha_base(sc, T, cfg)

    eff_reduction = max(0.0, min(0.90, eff_reduction))
    if K_HP is None:
        K_HP = min(cfg.K_HP_max, max(0.0, K_HX) * _hp_source_to_power_ratio(sc.hp_cop))
    K_HP = max(0.0, min(cfg.K_HP_max, float(K_HP)))
    alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

    p_heat_eff = p_heat
    dh_cap_eff = dh_cap

    m = pulp.LpProblem(f"Ops_{sc.sid}", pulp.LpMinimize)

    IT_SERV = pulp.LpVariable.dicts("IT_SERV", range(T), lowBound=0)
    COOL = pulp.LpVariable.dicts("COOL", range(T), lowBound=0)
    H_REC = pulp.LpVariable.dicts("H_REC", range(T), lowBound=0)
    H_REC_IT = pulp.LpVariable.dicts("H_REC_IT", range(T), lowBound=0)
    H_REC_COOL = pulp.LpVariable.dicts("H_REC_COOL", range(T), lowBound=0)
    H_FLOW = pulp.LpVariable.dicts("H_FLOW", range(T), lowBound=0)
    H_IN = pulp.LpVariable.dicts("H_IN", range(T), lowBound=0)
    H_SOLD = pulp.LpVariable.dicts("H_SOLD", range(T), lowBound=0)
    P_HP = pulp.LpVariable.dicts("P_HP", range(T), lowBound=0)

    K = int(cfg.dh_segments)
    if K <= 0:
        raise ValueError("dh_segments must be >= 1")
    if len(cfg.dh_seg_fracs) != K or len(cfg.dh_value_multipliers) != K:
        raise ValueError("dh_seg_fracs and dh_value_multipliers must have length dh_segments")
    if abs(sum(cfg.dh_seg_fracs) - 1.0) > 1e-6:
        raise ValueError("dh_seg_fracs must sum to 1.0")
    Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}", range(T), lowBound=0) for k in range(K)}

    RF_LOAD = pulp.LpVariable.dicts("RF_LOAD", range(T), lowBound=0)
    RF_BAT = pulp.LpVariable.dicts("RF_BAT", range(T), lowBound=0)
    RF = pulp.LpVariable.dicts("RF", range(T), lowBound=0)
    DC_LOAD = pulp.LpVariable.dicts("DC_LOAD", range(T), lowBound=0)
    P_PEAK = pulp.LpVariable("P_PEAK", lowBound=0)

    CH = pulp.LpVariable.dicts("CH", range(T), lowBound=0)
    DIS = pulp.LpVariable.dicts("DIS", range(T), lowBound=0)
    SOC = pulp.LpVariable.dicts("SOC", range(T + 1), lowBound=0)
    SLA_SHORT = pulp.LpVariable.dicts("SLA_SHORT", range(T), lowBound=0)
    SLA_SHORT_EFF = pulp.LpVariable.dicts("SLA_SHORT_EFF", range(T), lowBound=0)

    if cfg.enable_thermal_storage:
        CH_TH = pulp.LpVariable.dicts("CH_TH", range(T), lowBound=0)
        DIS_TH = pulp.LpVariable.dicts("DIS_TH", range(T), lowBound=0)
        S_TH = pulp.LpVariable.dicts("S_TH", range(T + 1), lowBound=0)
        K_TH_STORE = cfg.k_th_store_share_of_daily_peak * sc.heat_demand_level
        P_TH_CH = cfg.th_store_p_ch_share * sc.heat_demand_level
        P_TH_DIS = cfg.th_store_p_dis_share * sc.heat_demand_level
    else:
        CH_TH = {t: 0 for t in range(T)}
        DIS_TH = {t: 0 for t in range(T)}
        S_TH = {t: 0 for t in range(T + 1)}
        K_TH_STORE = 0.0
        P_TH_CH = 0.0
        P_TH_DIS = 0.0

    G_REN = {n: pulp.LpVariable.dicts(f"G_REN_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
    G_TH = {n: pulp.LpVariable.dicts(f"G_TH_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
    F12 = pulp.LpVariable.dicts("F12", range(T), lowBound=None)
    F23 = pulp.LpVariable.dicts("F23", range(T), lowBound=None)
    F13 = pulp.LpVariable.dicts("F13", range(T), lowBound=None)
    THETA = {n: pulp.LpVariable.dicts(f"THETA_{n}", range(T), lowBound=None) for n in cfg.network_nodes}
    balance_constraints = {n: [] for n in cfg.network_nodes}
    line_limit_constraints = {line: [] for line in ("12+", "12-", "23+", "23-", "13+", "13-")}

    if cfg.transmission_cost_eur_per_mwh > 0:
        AF12 = pulp.LpVariable.dicts("AF12", range(T), lowBound=0)
        AF23 = pulp.LpVariable.dicts("AF23", range(T), lowBound=0)
        AF13 = pulp.LpVariable.dicts("AF13", range(T), lowBound=0)
    else:
        AF12 = {t: 0 for t in range(T)}
        AF23 = {t: 0 for t in range(T)}
        AF13 = {t: 0 for t in range(T)}

    U_GRID = pulp.LpVariable.dicts("U_GRID", range(T), lowBound=0, upBound=1)

    if not cfg.fix_it_profile:
        m += pulp.lpSum([IT_SERV[t] * dt for t in range(T)]) == sum(it_base) * dt, "WorkConservation"

    m += SOC[0] == 0.5 * K_BAT_E, "SOC_start"
    m += SOC[T] == SOC[0], "SOC_cycle"

    if cfg.enable_thermal_storage:
        m += S_TH[0] == 0.0, "S_TH_start"
        m += S_TH[T] == 0.0, "S_TH_end"

    n1, n2, n3 = cfg.network_nodes
    force_it_flex = bool(prof.get("force_it_flex", False))
    flex_frac_eff = min(0.30, sc.flex_frac * (cfg.it_flex_scale_services_active if allow_flex else 1.0))
    for t in range(T):
        m += THETA[n1][t] == 0.0
        m += F12[t] == b12 * (THETA[n1][t] - THETA[n2][t])
        m += F23[t] == b23 * (THETA[n2][t] - THETA[n3][t])
        m += F13[t] == b13 * (THETA[n1][t] - THETA[n3][t])
        for key, constraint in (
            ("12+", F12[t] <= cap12), ("12-", -F12[t] <= cap12),
            ("23+", F23[t] <= cap23), ("23-", -F23[t] <= cap23),
            ("13+", F13[t] <= cap13), ("13-", -F13[t] <= cap13),
        ):
            m += constraint
            line_limit_constraints[key].append(constraint)
        if allow_it_shortfall:
            m += IT_SERV[t] <= it_demand[t]
            m += IT_SERV[t] >= (1.0 - cfg.max_it_curtail_frac) * it_demand[t]
            it_ub = it_demand[t]
        elif force_it_flex:
            lb = (1 - flex_frac_eff) * it_base[t]
            ub = (1 + flex_frac_eff) * it_base[t]
            m += IT_SERV[t] >= lb
            m += IT_SERV[t] <= ub
            it_ub = ub
        elif cfg.fix_it_profile:
            m += IT_SERV[t] == it_base[t]
            it_ub = it_base[t]
        elif not cfg.fix_it_profile:
            lb = (1 - flex_frac_eff) * it_base[t]
            ub = (1 + flex_frac_eff) * it_base[t]
            m += IT_SERV[t] >= lb
            m += IT_SERV[t] <= ub
            it_ub = ub
        else:
            it_cap = sc.base_it_mw * cfg.it_capacity_multiplier
            ramp_share = max(0.0, min(1.0, cfg.ramp_base_share - cfg.ramp_tau_per_mw * design_size_proxy(sc, K_HX, K_FLEX, K_BAT_E, K_BAT_P, cfg, K_HP=K_HP)))
            m += IT_SERV[t] <= it_demand[t] * ramp_share
            m += IT_SERV[t] <= it_cap
            it_ub = it_demand[t] * ramp_share

        m += COOL[t] == alpha_eff[t] * IT_SERV[t]

        if allow_heat and K_HX > 0 and K_HP > 0:
            pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
            delivered_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_delivered_ratio(cop_t[t])
            hp_power_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_power_ratio(cop_t[t])
            m += H_REC[t] == H_REC_IT[t] + H_REC_COOL[t]
            m += H_REC_IT[t] == H_REC[t] * (1.0 / max(1e-9, 1.0 + alpha_eff[t]))
            m += H_REC_COOL[t] == H_REC[t] * alpha_eff[t] / max(1e-9, 1.0 + alpha_eff[t])
            m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
            m += H_REC[t] <= K_HX
            m += H_FLOW[t] <= H_REC[t]
            m += H_FLOW[t] <= pipe_cap
            m += H_IN[t] == delivered_ratio * H_FLOW[t]
            m += P_HP[t] == hp_power_ratio * H_FLOW[t]
            m += P_HP[t] <= K_HP

            m += H_SOLD[t] == pulp.lpSum(Q_DH[k][t] for k in range(K))
            for k in range(K):
                m += Q_DH[k][t] <= cfg.dh_seg_fracs[k] * dh_cap_eff[t]

            if cfg.enable_thermal_storage:
                m += S_TH[t] <= K_TH_STORE
                m += S_TH[t + 1] <= K_TH_STORE
                m += CH_TH[t] <= P_TH_CH
                m += DIS_TH[t] <= P_TH_DIS
                m += S_TH[t + 1] == (1.0 - cfg.th_store_loss_per_hour) * S_TH[t] + cfg.eta_th_ch * CH_TH[t] * dt - (1.0 / cfg.eta_th_dis) * DIS_TH[t] * dt
                m += H_IN[t] + DIS_TH[t] == H_SOLD[t] + CH_TH[t]
            else:
                m += H_IN[t] == H_SOLD[t]
        else:
            m += H_REC[t] == 0
            m += H_REC_IT[t] == 0
            m += H_REC_COOL[t] == 0
            m += H_FLOW[t] == 0
            m += H_IN[t] == 0
            m += H_SOLD[t] == 0
            m += P_HP[t] == 0
            for k in range(K):
                m += Q_DH[k][t] == 0
            if cfg.enable_thermal_storage:
                m += CH_TH[t] == 0
                m += DIS_TH[t] == 0
                m += S_TH[t + 1] == S_TH[t]

        flex_reference = it_demand[t]
        m += SOC[t] <= K_BAT_E
        m += SOC[t + 1] <= K_BAT_E

        if allow_battery and (K_BAT_E > 0 or K_BAT_P > 0):
            m += CH[t] <= K_BAT_P
            m += DIS[t] <= K_BAT_P
            m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1 / cfg.eta_dis) * DIS[t] * dt
            reserve_duration = max(1e-9, cfg.battery_flex_reserve_duration_h)
            ups_reserve = max(0.0, min(1.0, cfg.battery_ups_reserve_fraction)) * K_BAT_E
            m += SOC[t] >= ups_reserve
            m += SOC[t + 1] >= ups_reserve
            m += RF_BAT[t] <= K_BAT_P - DIS[t]
            m += RF_BAT[t] <= (SOC[t] - ups_reserve) / reserve_duration
        else:
            m += CH[t] == 0
            m += DIS[t] == 0
            m += SOC[t + 1] == SOC[t]
            m += RF_BAT[t] == 0

        if allow_flex and K_FLEX > 0:
            m += RF_LOAD[t] <= K_FLEX
            m += RF_LOAD[t] <= flex_frac_eff * flex_reference
            m += IT_SERV[t] + RF_LOAD[t] <= flex_reference
        else:
            m += RF_LOAD[t] == 0
        if not allow_flex:
            m += RF_BAT[t] == 0
        m += RF[t] == RF_LOAD[t] + RF_BAT[t]

        P_SERV = (
            cfg.service_parasitic_heat_frac * H_SOLD[t]
            + cfg.service_parasitic_flex_frac * RF[t]
            + cfg.service_parasitic_hx_mw_per_mwth * K_HX
            + cfg.service_parasitic_flex_mw_per_mw * K_FLEX
        )

        m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP[t] + P_SERV + CH[t] - DIS[t]
        m += P_PEAK >= DC_LOAD[t]
        if allow_it_shortfall:
            dc_grid_cap_t = cfg.dc_grid_cap_share * it_demand[t] * (1.0 + alpha_eff[t])
            m += DC_LOAD[t] <= dc_grid_cap_t

        if allow_it_shortfall:
            m += SLA_SHORT[t] >= it_demand[t] - IT_SERV[t]
            m += SLA_SHORT[t] <= it_demand[t]
            if cfg.sla_penalty_only_when_congested:
                m += SLA_SHORT_EFF[t] <= SLA_SHORT[t]
                m += SLA_SHORT_EFF[t] <= it_demand[t] * U_GRID[t]
                m += SLA_SHORT_EFF[t] >= SLA_SHORT[t] - it_demand[t] * (1.0 - U_GRID[t])
            else:
                m += SLA_SHORT_EFF[t] == SLA_SHORT[t]
        else:
            m += SLA_SHORT[t] == 0
            m += SLA_SHORT_EFF[t] == 0

        dc_at_n1 = DC_LOAD[t] if dc_node == n1 else 0.0
        dc_at_n2 = DC_LOAD[t] if dc_node == n2 else 0.0
        dc_at_n3 = DC_LOAD[t] if dc_node == n3 else 0.0

        ren_avail = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
        m += G_REN[n1][t] <= ren_avail[n1][t]
        m += G_REN[n2][t] <= ren_avail[n2][t]
        m += G_REN[n3][t] <= ren_avail[n3][t]

        m += G_TH[n1][t] <= cfg.th_cap_N1_mw
        m += G_TH[n2][t] <= cfg.th_cap_N2_mw
        m += G_TH[n3][t] <= cfg.th_cap_N3_mw

        con_n1 = (G_REN[n1][t] + G_TH[n1][t]) - F12[t] - F13[t] == node_loads[n1][t] + dc_at_n1
        con_n2 = (G_REN[n2][t] + G_TH[n2][t]) + F12[t] - F23[t] == node_loads[n2][t] + dc_at_n2
        con_n3 = (G_REN[n3][t] + G_TH[n3][t]) + F13[t] + F23[t] == node_loads[n3][t] + dc_at_n3
        m += con_n1
        m += con_n2
        m += con_n3
        balance_constraints[n1].append(con_n1)
        balance_constraints[n2].append(con_n2)
        balance_constraints[n3].append(con_n3)

        if cfg.transmission_cost_eur_per_mwh > 0:
            m += AF12[t] >= F12[t]
            m += AF12[t] >= -F12[t]
            m += AF23[t] >= F23[t]
            m += AF23[t] >= -F23[t]
            m += AF13[t] >= F13[t]
            m += AF13[t] >= -F13[t]

        m += U_GRID[t] >= F12[t] * (1.0 / cap12)
        m += U_GRID[t] >= -F12[t] * (1.0 / cap12)
        m += U_GRID[t] >= F23[t] * (1.0 / cap23)
        m += U_GRID[t] >= -F23[t] * (1.0 / cap23)
        m += U_GRID[t] >= F13[t] * (1.0 / cap13)
        m += U_GRID[t] >= -F13[t] * (1.0 / cap13)

    for start in rep_day_starts:
        end = start + 24
        if 0 <= start < T and end <= T:
            m += SOC[end] == SOC[start]
            if cfg.enable_thermal_storage:
                m += S_TH[end] == S_TH[start]

    fuel_price = prof.get("fuel_price", [0.0] * T)
    co2_price = prof.get("p_co2", [0.0] * T)
    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    el_cost_expr = pulp.lpSum(
        (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    power_emissions_expr = pulp.lpSum(
        th_ef[n] * G_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    power_co2_cost_expr = pulp.lpSum(
        co2_price[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    trans_cost_expr = 0
    if cfg.transmission_cost_eur_per_mwh > 0:
        trans_cost_expr = pulp.lpSum(
            cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
            for t in range(T)
        )

    avoided_heat_price = prof.get("avoided_heat_cost", p_heat_eff)
    heat_emissions_factor = (
        cfg.independent_heat_boiler_emissions_tco2_per_mwh_fuel
        / max(1e-9, cfg.independent_heat_boiler_efficiency)
    )
    baseline_heat_resource_cost_expr = pulp.lpSum(
        avoided_heat_price[t] * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    baseline_heat_emissions_expr = pulp.lpSum(
        heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    avoided_heat_emissions_expr = pulp.lpSum(
        heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
    )
    emissions_expr = power_emissions_expr + baseline_heat_emissions_expr - avoided_heat_emissions_expr
    baseline_heat_co2_cost_expr = pulp.lpSum(
        co2_price[t] * heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    avoided_heat_co2_cost_expr = pulp.lpSum(
        co2_price[t] * heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
    )
    co2_cost_expr = power_co2_cost_expr + baseline_heat_co2_cost_expr - avoided_heat_co2_cost_expr
    avoided_flex_price = prof.get(
        "avoided_flex_cost",
        [cfg.independent_avoided_flex_cost_eur_per_mw_h] * T,
    )
    heat_revenue_expr = pulp.lpSum(
        p_heat_eff[t] * cfg.dh_value_multipliers[k] * Q_DH[k][t] * dt * w[t]
        for k in range(K) for t in range(T)
    )
    flex_revenue_expr = pulp.lpSum(
        p_flex[t] * RF[t] * dt * w[t] for t in range(T)
    )
    demand_charge_expr = demand_charge_horizon(cfg) * P_PEAK

    if allow_it_shortfall:
        sla_penalty_expr = pulp.lpSum(cfg.sla_penalty_eur_per_mwh * SLA_SHORT_EFF[t] * dt * w[t] for t in range(T))
    else:
        sla_penalty_expr = 0
    E_IT_expr = pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T))
    E_FAC_expr = pulp.lpSum(DC_LOAD[t] * dt * w[t] for t in range(T))
    eff_benefit_expr = eff_reduction * (
        cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC_expr
        + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT_expr
    )

    scale_cap = capex_scale(cfg)
    capex = sc.capex_multiplier * scale_cap * (cfg.capex_HX * K_HX +
                         cfg.capex_HP * K_HP +
                         cfg.capex_FLEX * K_FLEX +
                         cfg.capex_BAT_E * K_BAT_E +
                         cfg.capex_BAT_P * K_BAT_P)
    eff_capex = scale_cap * eff_capex_from_reduction(eff_reduction, cfg)
    cooling_capacity_mw = _existing_cooling_capacity_mw(sc, alpha_base)
    land_components = compute_land_components(
        K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg,
        K_HP=K_HP, base_it_mw=sc.base_it_mw, K_COOL=cooling_capacity_mw,
    )
    land_m2 = sum(land_components.values())
    land_cost = sc.land_cost_multiplier * land_cost_horizon(land_m2, cfg)
    annual_market_operating_cost_expr = (
        el_cost_expr + baseline_heat_resource_cost_expr + co2_cost_expr + trans_cost_expr + demand_charge_expr
        + sla_penalty_expr - heat_revenue_expr - flex_revenue_expr
        - eff_benefit_expr
    )
    operating_pv_factor, land_pv_factor = _lifecycle_factors(cfg, sc)
    lifecycle_market_cost_expr = (
        operating_pv_factor * annual_market_operating_cost_expr
        + capex + eff_capex + land_pv_factor * land_cost
    )
    carbon_cap = (cfg.carbon_emissions_cap_tco2_by_sid or {}).get(sc.sid)
    if cfg.enforce_carbon_cap and carbon_cap is not None:
        m += emissions_expr <= float(carbon_cap), "PhysicalCarbonCap"
    m += lifecycle_market_cost_expr + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))

    m.solve(make_solver_lp(cfg))

    status = pulp.LpStatus[m.status]
    if cfg.validate_solutions and status != "Optimal":
        raise RuntimeError(f"Ops model did not solve to Optimal (status={status}).")

    duals = _extract_constraint_duals(m)
    line_caps_by_key = {
        "12+": cap12, "12-": cap12,
        "23+": cap23, "23-": cap23,
        "13+": cap13, "13-": cap13,
    }
    line_shadow_prices = [
        abs(duals.get(id(constraint), 0.0))
        for constraints in line_limit_constraints.values()
        for constraint in constraints
    ]
    max_line_shadow_price = max(line_shadow_prices, default=0.0)
    relief_fraction = max(0.0, float(cfg.congestion_relief_perturbation_frac))
    native_dual_linearized_value = sum(
        abs(duals.get(id(constraint), 0.0)) * line_caps_by_key[key] * relief_fraction
        for key, constraints in line_limit_constraints.items()
        for constraint in constraints
    )

    ITv = [pulp.value(IT_SERV[t]) for t in range(T)]
    COOLv = [pulp.value(COOL[t]) for t in range(T)]
    DC_LOADv = [pulp.value(DC_LOAD[t]) for t in range(T)]
    G_RENv = {n: [pulp.value(G_REN[n][t]) for t in range(T)] for n in cfg.network_nodes}
    G_THv = {n: [pulp.value(G_TH[n][t]) for t in range(T)] for n in cfg.network_nodes}
    F12v = [pulp.value(F12[t]) for t in range(T)]
    F23v = [pulp.value(F23[t]) for t in range(T)]
    F13v = [pulp.value(F13[t]) for t in range(T)]

    H_RECv = [pulp.value(H_REC[t]) for t in range(T)]
    H_REC_ITv = [pulp.value(H_REC_IT[t]) for t in range(T)]
    H_REC_COOLv = [pulp.value(H_REC_COOL[t]) for t in range(T)]
    H_FLOWv = [pulp.value(H_FLOW[t]) for t in range(T)]
    H_INv = [pulp.value(H_IN[t]) for t in range(T)]
    H_SOLDv = [pulp.value(H_SOLD[t]) for t in range(T)]
    P_HPv = [pulp.value(P_HP[t]) for t in range(T)]
    Q_DHv = {k: [pulp.value(Q_DH[k][t]) for t in range(T)] for k in range(K)}
    if cfg.enable_thermal_storage:
        CH_THv = [pulp.value(CH_TH[t]) for t in range(T)]
        DIS_THv = [pulp.value(DIS_TH[t]) for t in range(T)]
        S_THv = [pulp.value(S_TH[t]) for t in range(T + 1)]
        K_TH_STORE = cfg.k_th_store_share_of_daily_peak * sc.heat_demand_level
        P_TH_CH = cfg.th_store_p_ch_share * sc.heat_demand_level
        P_TH_DIS = cfg.th_store_p_dis_share * sc.heat_demand_level
    else:
        CH_THv = [0.0 for _ in range(T)]
        DIS_THv = [0.0 for _ in range(T)]
        S_THv = [0.0 for _ in range(T + 1)]
        K_TH_STORE = 0.0
        P_TH_CH = 0.0
        P_TH_DIS = 0.0

    U_GRIDv = [pulp.value(U_GRID[t]) for t in range(T)]
    RF_LOADv = [pulp.value(RF_LOAD[t]) for t in range(T)]
    RF_BATv = [pulp.value(RF_BAT[t]) for t in range(T)]
    RFv = [pulp.value(RF[t]) for t in range(T)]
    CHv = [pulp.value(CH[t]) for t in range(T)]
    DISv = [pulp.value(DIS[t]) for t in range(T)]
    SOCv = [pulp.value(SOC[t]) for t in range(T + 1)]
    peak_import_mw = float(pulp.value(P_PEAK))
    util_12 = [abs(F12v[t]) / max(1e-9, cap12) for t in range(T)]
    util_23 = [abs(F23v[t]) / max(1e-9, cap23) for t in range(T)]
    util_13 = [abs(F13v[t]) / max(1e-9, cap13) for t in range(T)]
    util_max = [max(util_12[t], util_23[t], util_13[t]) for t in range(T)]
    congestion_hours = sum(w[t] for t in range(T) if util_max[t] >= 1.0 - cfg.line_binding_tolerance)
    w_sum = sum(w) if w else float(T)

    E_IT = sum(ITv[t] * w[t] for t in range(T)) * dt
    E_GRID = sum(DC_LOADv[t] * w[t] for t in range(T)) * dt
    E_LOAD = sum((DC_LOADv[t] + DISv[t] - CHv[t]) * w[t] * dt for t in range(T))
    E_FAC = E_GRID
    PUE = E_FAC / max(1e-9, E_IT)

    if cfg.validate_solutions:
        tol = float(cfg.validate_tol)

        def _bad(msg: str) -> None:
            raise AssertionError(f"[validate_ops] {msg}")

        for seq_name, seq in [
            ("IT_SERV", ITv),
            ("COOL", COOLv),
            ("DC_LOAD", DC_LOADv),
            ("F12", F12v),
            ("F23", F23v),
            ("F13", F13v),
            ("U_GRID", U_GRIDv),
            ("H_REC", H_RECv),
            ("H_REC_IT", H_REC_ITv),
            ("H_REC_COOL", H_REC_COOLv),
            ("H_FLOW", H_FLOWv),
            ("H_IN", H_INv),
            ("H_SOLD", H_SOLDv),
            ("P_HP", P_HPv),
            ("RF_LOAD", RF_LOADv),
            ("RF_BAT", RF_BATv),
            ("SOC", SOCv),
        ]:
            if any(v is None or (isinstance(v, float) and (v != v)) for v in seq):
                _bad(f"{seq_name} contains None/NaN values.")

        if E_IT <= tol:
            _bad(f"E_IT is too small ({E_IT}); PUE/siPUE may be ill-defined.")
        if not (PUE == PUE and abs(PUE) < 1e12):
            _bad(f"PUE is not finite: {PUE}")
        if cfg.enable_demand_charge:
            if peak_import_mw < -tol:
                _bad(f"Peak import is negative: {peak_import_mw}")
            max_load = max(DC_LOADv) if DC_LOADv else 0.0
            if peak_import_mw + tol < max_load:
                _bad(f"Peak import {peak_import_mw} is below max hourly load {max_load}.")

        ren_avail_prof = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})

        for t in range(T):
            if abs(F12v[t]) > cap12 + tol:
                _bad(f"|F12[{t}]|={abs(F12v[t])} exceeds cap {cap12}.")
            if abs(F23v[t]) > cap23 + tol:
                _bad(f"|F23[{t}]|={abs(F23v[t])} exceeds cap {cap23}.")
            if abs(F13v[t]) > cap13 + tol:
                _bad(f"|F13[{t}]|={abs(F13v[t])} exceeds cap {cap13}.")

            util = util_max[t]
            if U_GRIDv[t] + tol < util:
                _bad(f"U_GRID[{t}]={U_GRIDv[t]} is below max line utilization {util}.")
            if U_GRIDv[t] < -tol or U_GRIDv[t] > 1.0 + tol:
                _bad(f"U_GRID[{t}]={U_GRIDv[t]} is outside [0,1].")
            if cfg.eps_u_grid and cfg.eps_u_grid > 0:
                tight_tol = max(1e-4, 100.0 * tol)
                if U_GRIDv[t] > util + tight_tol:
                    _bad(f"U_GRID[{t}]={U_GRIDv[t]} is not tight vs utilization {util} (tight_tol={tight_tol}).")

            for n in cfg.network_nodes:
                if G_RENv[n][t] < -tol:
                    _bad(f"G_REN[{n},{t}]={G_RENv[n][t]} is negative.")
                if G_THv[n][t] < -tol:
                    _bad(f"G_TH[{n},{t}]={G_THv[n][t]} is negative.")
                if G_RENv[n][t] > ren_avail_prof[n][t] + tol:
                    _bad(f"G_REN[{n},{t}] exceeds renewable availability.")
            if G_THv["N1"][t] > cfg.th_cap_N1_mw + tol:
                _bad(f"G_TH[N1,{t}] exceeds thermal cap.")
            if G_THv["N2"][t] > cfg.th_cap_N2_mw + tol:
                _bad(f"G_TH[N2,{t}] exceeds thermal cap.")
            if G_THv["N3"][t] > cfg.th_cap_N3_mw + tol:
                _bad(f"G_TH[N3,{t}] exceeds thermal cap.")

            dc_at_n1 = DC_LOADv[t] if dc_node == "N1" else 0.0
            dc_at_n2 = DC_LOADv[t] if dc_node == "N2" else 0.0
            dc_at_n3 = DC_LOADv[t] if dc_node == "N3" else 0.0
            res1 = (G_RENv["N1"][t] + G_THv["N1"][t]) - F12v[t] - F13v[t] - (node_loads["N1"][t] + dc_at_n1)
            res2 = (G_RENv["N2"][t] + G_THv["N2"][t]) + F12v[t] - F23v[t] - (node_loads["N2"][t] + dc_at_n2)
            res3 = (G_RENv["N3"][t] + G_THv["N3"][t]) + F13v[t] + F23v[t] - (node_loads["N3"][t] + dc_at_n3)
            if abs(res1) > tol * (1.0 + abs(node_loads["N1"][t]) + abs(dc_at_n1)):
                _bad(f"Balance N1 residual at t={t}: {res1}")
            if abs(res2) > tol * (1.0 + abs(node_loads["N2"][t]) + abs(dc_at_n2)):
                _bad(f"Balance N2 residual at t={t}: {res2}")
            if abs(res3) > tol * (1.0 + abs(node_loads["N3"][t]) + abs(dc_at_n3)):
                _bad(f"Balance N3 residual at t={t}: {res3}")

            if allow_heat and K_HX > 0 and K_HP > 0:
                pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
                delivered_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_delivered_ratio(cop_t[t])
                hp_power_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_power_ratio(cop_t[t])
                if H_RECv[t] > sc.heat_recovery_frac * (ITv[t] + COOLv[t]) + tol:
                    _bad(f"H_REC[{t}] exceeds recoverable heat.")
                if abs(H_RECv[t] - H_REC_ITv[t] - H_REC_COOLv[t]) > tol:
                    _bad(f"Recovered heat source attribution is inconsistent at t={t}.")
                if H_RECv[t] > K_HX + tol:
                    _bad(f"H_REC[{t}] exceeds K_HX.")
                if H_FLOWv[t] > H_RECv[t] + tol:
                    _bad(f"H_FLOW[{t}] exceeds H_REC.")
                if H_FLOWv[t] > pipe_cap + tol:
                    _bad(f"H_FLOW[{t}] exceeds pipe capacity.")
                if abs(H_INv[t] - delivered_ratio * H_FLOWv[t]) > tol * (1.0 + abs(delivered_ratio * H_FLOWv[t])):
                    _bad(f"H_IN[{t}] inconsistent with pipe loss and heat-pump lift.")
                if abs(P_HPv[t] - hp_power_ratio * H_FLOWv[t]) > tol * (1.0 + abs(hp_power_ratio * H_FLOWv[t])):
                    _bad(f"P_HP[{t}] inconsistent with recovered heat and COP.")
                if P_HPv[t] > K_HP + tol:
                    _bad(f"P_HP[{t}] exceeds K_HP.")

                sold = sum(Q_DHv[k][t] for k in range(K))
                if abs(H_SOLDv[t] - sold) > tol * (1.0 + abs(sold)):
                    _bad(f"H_SOLD[{t}] not equal to sum(Q_DH).")
                for k in range(K):
                    if Q_DHv[k][t] < -tol:
                        _bad(f"Q_DH[{k},{t}] is negative.")
                    if Q_DHv[k][t] > cfg.dh_seg_fracs[k] * dh_cap_eff[t] + tol:
                        _bad(f"Q_DH[{k},{t}] exceeds segment cap.")

                if cfg.enable_thermal_storage:
                    if S_THv[t] < -tol or S_THv[t] > K_TH_STORE + tol:
                        _bad(f"S_TH[{t}] outside bounds.")
                    if CH_THv[t] < -tol or CH_THv[t] > P_TH_CH + tol:
                        _bad(f"CH_TH[{t}] outside bounds.")
                    if DIS_THv[t] < -tol or DIS_THv[t] > P_TH_DIS + tol:
                        _bad(f"DIS_TH[{t}] outside bounds.")
                    dyn_rhs = (1.0 - cfg.th_store_loss_per_hour) * S_THv[t] + cfg.eta_th_ch * CH_THv[t] * dt - (1.0 / cfg.eta_th_dis) * DIS_THv[t] * dt
                    if abs(S_THv[t + 1] - dyn_rhs) > tol * (1.0 + abs(dyn_rhs)):
                        _bad(f"S_TH dynamics violated at t={t}.")
                    bal = (H_INv[t] + DIS_THv[t]) - (H_SOLDv[t] + CH_THv[t])
                    if abs(bal) > tol * (1.0 + abs(H_INv[t]) + abs(H_SOLDv[t])):
                        _bad(f"Heat balance violated at t={t}: {bal}.")
                else:
                    if abs(H_INv[t] - H_SOLDv[t]) > tol * (1.0 + abs(H_INv[t])):
                        _bad(f"Heat balance (no storage) violated at t={t}.")
            else:
                if abs(H_RECv[t]) > tol or abs(H_FLOWv[t]) > tol or abs(H_INv[t]) > tol or abs(H_SOLDv[t]) > tol or abs(P_HPv[t]) > tol:
                    _bad(f"Heat vars nonzero while heat disabled at t={t}.")
                for k in range(K):
                    if abs(Q_DHv[k][t]) > tol:
                        _bad(f"Q_DH nonzero while heat disabled at t={t}.")
                if cfg.enable_thermal_storage:
                    if abs(CH_THv[t]) > tol or abs(DIS_THv[t]) > tol:
                        _bad(f"Thermal storage flows nonzero while heat disabled at t={t}.")

            flex_reference = it_demand[t]
            if abs(RFv[t] - RF_LOADv[t] - RF_BATv[t]) > tol * (1.0 + abs(RFv[t])):
                _bad(f"RF[{t}] is not equal to RF_LOAD + RF_BAT.")
            if allow_flex and K_FLEX > 0:
                if RF_LOADv[t] > K_FLEX + tol:
                    _bad(f"RF_LOAD[{t}] exceeds K_FLEX.")
                if RF_LOADv[t] > flex_frac_eff * flex_reference + tol:
                    _bad(f"RF_LOAD[{t}] exceeds IT flexibility fraction.")
                if ITv[t] + RF_LOADv[t] > flex_reference + tol:
                    _bad(f"IT_SERV[{t}] + RF_LOAD[{t}] exceeds flex reference.")
            elif abs(RF_LOADv[t]) > tol:
                _bad(f"RF_LOAD nonzero while flex disabled at t={t}.")
            if allow_flex and allow_battery and (K_BAT_E > 0 or K_BAT_P > 0):
                reserve_duration = max(1e-9, cfg.battery_flex_reserve_duration_h)
                ups_reserve = max(0.0, min(1.0, cfg.battery_ups_reserve_fraction)) * K_BAT_E
                if SOCv[t] < ups_reserve - tol or SOCv[t + 1] < ups_reserve - tol:
                    _bad(f"SOC[{t}] violates UPS reserve.")
                if RF_BATv[t] > K_BAT_P - DISv[t] + tol:
                    _bad(f"RF_BAT[{t}] exceeds unused battery inverter power.")
                if RF_BATv[t] > (SOCv[t] - ups_reserve) / reserve_duration + tol:
                    _bad(f"RF_BAT[{t}] exceeds stored energy above UPS reserve.")
            elif abs(RF_BATv[t]) > tol:
                _bad(f"RF_BAT nonzero without active flex-capable battery at t={t}.")

    fuel_price = prof.get("fuel_price", [0.0] * T)
    co2_price = prof.get("p_co2", [0.0] * T)
    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    el_cost_val = sum(
        (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_THv[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    power_co2_cost_val = sum(
        co2_price[t] * th_ef[n] * G_THv[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    trans_cost_val = 0.0
    if cfg.transmission_cost_eur_per_mwh > 0:
        trans_cost_val = cfg.transmission_cost_eur_per_mwh * sum(
            (abs(F12v[t]) + abs(F23v[t]) + abs(F13v[t])) * dt * w[t] for t in range(T)
        )
    flex_load_mwh = sum(RF_LOADv[t] * w[t] for t in range(T)) * dt
    flex_bat_mwh = sum(RF_BATv[t] * w[t] for t in range(T)) * dt
    if allow_it_shortfall:
        cong_factor_v = U_GRIDv if cfg.sla_penalty_only_when_congested else [1.0] * T
        sla_penalty_val = sum(
            (cfg.sla_penalty_eur_per_mwh * cong_factor_v[t])
            * max(0.0, it_demand[t] - ITv[t]) * dt * w[t]
            for t in range(T)
        )
    else:
        sla_penalty_val = 0.0
    demand_charge_val = demand_charge_horizon(cfg) * peak_import_mw

    bat_charge_mwh = sum(CHv[t] * w[t] for t in range(T)) * dt
    bat_discharge_mwh = sum(DISv[t] * w[t] for t in range(T)) * dt

    power_emissions_tco2 = sum(
        th_ef[n] * G_THv[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    avoided_heat_price = prof.get("avoided_heat_cost", p_heat_eff)
    heat_emissions_factor = (
        cfg.independent_heat_boiler_emissions_tco2_per_mwh_fuel
        / max(1e-9, cfg.independent_heat_boiler_efficiency)
    )
    baseline_heat_resource_cost = sum(
        avoided_heat_price[t] * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    baseline_heat_emissions_tco2 = sum(
        heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    avoided_heat_emissions_tco2 = sum(
        heat_emissions_factor * H_SOLDv[t] * dt * w[t] for t in range(T)
    )
    baseline_heat_co2_cost = sum(
        co2_price[t] * heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
    )
    avoided_heat_co2_cost = sum(
        co2_price[t] * heat_emissions_factor * H_SOLDv[t] * dt * w[t] for t in range(T)
    )
    emissions_tco2 = power_emissions_tco2 + baseline_heat_emissions_tco2 - avoided_heat_emissions_tco2
    co2_cost_val = power_co2_cost_val + baseline_heat_co2_cost - avoided_heat_co2_cost

    eff_benefit_val = eff_reduction * (
        cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC
        + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT
    )

    cue_tco2_per_mwh_it = emissions_tco2 / max(1e-9, E_IT)
    heat_delivered_mwhth = sum(H_SOLDv[t] * w[t] for t in range(T)) * dt
    heat_revenue_val = sum(
        p_heat_eff[t] * cfg.dh_value_multipliers[k] * Q_DHv[k][t] * dt * w[t]
        for k in range(K) for t in range(T)
    )
    flex_revenue_val = sum(p_flex[t] * RFv[t] * dt * w[t] for t in range(T))
    (
        independent_system_cost,
        independent_avoided_heat_cost,
        independent_avoided_flex_cost,
        independent_carbon_damage,
        independent_peak_capacity_cost,
    ) = compute_independent_system_cost(
        power_resource_cost_eur=el_cost_val + baseline_heat_resource_cost,
        transmission_cost_eur=trans_cost_val,
        emissions_tco2=emissions_tco2,
        heat_delivered_mwhth=heat_delivered_mwhth,
        peak_grid_mw=peak_import_mw,
        cfg=cfg,
        avoided_heat_cost_eur_per_mwhth=prof.get("avoided_heat_cost"),
        heat_delivered_mwth=H_SOLDv,
        avoided_flex_cost_eur_per_mw_h=prof.get("avoided_flex_cost"),
        flex_delivered_mw=RFv,
        weights=w,
        dt=dt,
        carbon_damage_eur_per_tco2=sc.carbon_damage_eur_per_tco2,
    )
    # An isolated dispatch has no counterfactual, so its siPUE is neutral.
    # Scenario exports overwrite this with no-services avoided system cost.
    siPUE, system_credit_mwh = compute_sipue(
        E_FAC=E_FAC,
        E_IT=E_IT,
        net_system_benefit_eur=0.0,
        reference_price_eur_per_mwh=cfg.sipue_system_ref_price_eur_per_mwh,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
    )
    land_components_val = compute_land_components(
        K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg,
        K_HP=K_HP, base_it_mw=sc.base_it_mw, K_COOL=cooling_capacity_mw,
    )
    land_m2_val = sum(land_components_val.values())
    land_cost_val = sc.land_cost_multiplier * land_cost_horizon(land_m2_val, cfg)
    land_limit_val = effective_land_limit_m2(sc, cfg)
    land_utilization_pct = (
        100.0 * land_m2_val / land_limit_val if land_limit_val is not None and land_limit_val > 0.0 else None
    )
    land_binding = (
        land_limit_val is not None
        and abs(land_limit_val - land_m2_val) <= cfg.land_binding_tolerance_m2
    )
    land_feasible = land_limit_val is None or land_m2_val <= land_limit_val + cfg.validate_tol
    eff_capex_val = sc.capex_multiplier * capex_scale(cfg) * eff_capex_from_reduction(eff_reduction, cfg)
    total_capex_val = capex + eff_capex_val
    annual_market_operating_cost = (
        el_cost_val + baseline_heat_resource_cost + co2_cost_val + trans_cost_val + demand_charge_val + sla_penalty_val
        - heat_revenue_val - flex_revenue_val - eff_benefit_val
    )
    ledgers = build_carbon_ledgers(
        resource_cost_eur=el_cost_val + baseline_heat_resource_cost,
        transmission_cost_eur=trans_cost_val,
        carbon_market_cost_eur=co2_cost_val,
        carbon_damage_eur=independent_carbon_damage,
        market_grid_peak_cost_eur=demand_charge_val,
        independent_peak_capacity_cost_eur=independent_peak_capacity_cost,
        sla_cost_eur=sla_penalty_val,
        capex_eur=total_capex_val,
        land_cost_eur=land_cost_val,
        heat_revenue_eur=heat_revenue_val,
        flex_revenue_eur=flex_revenue_val,
        avoided_heat_cost_eur=independent_avoided_heat_cost,
        avoided_flex_cost_eur=independent_avoided_flex_cost,
        efficiency_benefit_eur=eff_benefit_val,
    )
    lifecycle = lifecycle_market_value(
        annual_market_operating_cost_eur=annual_market_operating_cost,
        upfront_capex_eur=total_capex_val,
        annual_land_cost_eur=land_cost_val,
        years=cfg.lifecycle_years,
        real_discount_rate=cfg.real_discount_rate,
        commissioning_delay_months=_commissioning_delay_months(cfg, sc),
    )
    planning_objective_cost_eur = lifecycle["market_cost_npv_eur"]
    recovered_heat_it_mwhth = sum(H_REC_ITv[t] * w[t] for t in range(T)) * dt
    recovered_heat_cooling_mwhth = sum(H_REC_COOLv[t] * w[t] for t in range(T)) * dt
    cooling_mwh = sum(COOLv[t] * w[t] for t in range(T)) * dt
    storage_loss_mwh = max(0.0, bat_charge_mwh - bat_discharge_mwh)
    thermal_generation_mwh = sum(
        G_THv[n][t] * dt * w[t] for n in cfg.network_nodes for t in range(T)
    )
    renewable_available_mwh = sum(
        prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    renewable_used_mwh = sum(
        G_RENv[n][t] * dt * w[t] for n in cfg.network_nodes for t in range(T)
    )
    average_resource_cost = el_cost_val / max(1e-9, thermal_generation_mwh)
    cooling_resource_cost = min(el_cost_val, average_resource_cost * cooling_mwh)
    storage_resource_cost = min(
        max(0.0, el_cost_val - cooling_resource_cost),
        average_resource_cost * storage_loss_mwh,
    )
    other_resource_cost = el_cost_val - cooling_resource_cost - storage_resource_cost

    result = {
        "status": pulp.LpStatus[m.status],
        "planning_objective_cost_eur": planning_objective_cost_eur,
        "market_planning_cost_eur": ledgers.market_cost_eur,
        "market_planning_value_npv_eur": lifecycle["market_value_npv_eur"],
        "market_planning_cost_npv_eur": lifecycle["market_cost_npv_eur"],
        "independent_system_cost_eur": ledgers.independent_cost_eur,
        "lifecycle_operating_factor": lifecycle["operating_factor"],
        "lifecycle_availability_factor": lifecycle["availability_factor"],
        "operating_cost_npv_eur": lifecycle["operating_cost_npv_eur"],
        "capex_npv_eur": lifecycle["capex_npv_eur"],
        "land_cost_npv_eur": lifecycle["land_cost_npv_eur"],
        "E_IT": E_IT,
        "base_it_mw": sc.base_it_mw,
        "E_FAC": E_FAC,
        "E_GRID": E_GRID,
        "E_LOAD": E_LOAD,
        "PUE": PUE,
        "siPUE": siPUE,
        "CUE_tCO2_per_MWh_IT": cue_tco2_per_mwh_it,
        "system_benefit_eur": 0.0,
        "system_credit_mwh": system_credit_mwh,
        "grid_util_series": util_max,
        "dc_load_series": DC_LOADv,
        "emissions_tco2": emissions_tco2,
        "power_emissions_tco2": power_emissions_tco2,
        "baseline_heat_emissions_tco2": baseline_heat_emissions_tco2,
        "avoided_heat_emissions_tco2": avoided_heat_emissions_tco2,

        "K_HX": K_HX,
        "K_HP": K_HP,
        "K_FLEX": K_FLEX,
        "K_BAT_E": K_BAT_E,
        "K_BAT_P": K_BAT_P,
        "hp_electric_mw_peak": max(P_HPv) if P_HPv else 0.0,
        "hp_electric_mwh": sum(P_HPv[t] * w[t] for t in range(T)) * dt,
        "cooling_mwh": cooling_mwh,
        "storage_loss_mwh": storage_loss_mwh,
        "recovered_heat_it_mwhth": recovered_heat_it_mwhth,
        "recovered_heat_cooling_mwhth": recovered_heat_cooling_mwhth,
        "heat_MWhth": heat_delivered_mwhth,
        "flex_cap_hours": flex_load_mwh + flex_bat_mwh,
        "flex_load_mwh": flex_load_mwh,
        "flex_bat_mwh": flex_bat_mwh,
        "grid_util_avg": sum(util_max[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "grid_util_max": max(util_max) if util_max else float("nan"),
        "line_util_12_avg": sum(util_12[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_23_avg": sum(util_23[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_13_avg": sum(util_13[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_12_max": max(util_12) if util_12 else float("nan"),
        "line_util_23_max": max(util_23) if util_23 else float("nan"),
        "line_util_13_max": max(util_13) if util_13 else float("nan"),
        "congestion_hours": congestion_hours,
        "congestion_native_dual_max_eur_per_mw": max_line_shadow_price,
        "congestion_native_dual_linearized_eur": native_dual_linearized_value,
        "congestion_marginal_value_eur_per_1pct": 0.0,
        "renewable_curtailment_mwh": max(0.0, renewable_available_mwh - renewable_used_mwh),
        "thermal_generation_mwh": thermal_generation_mwh,
        "line_cap_12_mw": cap12,
        "line_cap_23_mw": cap23,
        "line_cap_13_mw": cap13,
        "heat_pipe_util_avg": (
            sum((H_FLOWv[t] / max(1e-9, cfg.heat_pipe_cap_share * sc.heat_demand_level)) * w[t] for t in range(T)) / max(1e-9, w_sum)
        ),
        "heat_pipe_util_max": (
            max((H_FLOWv[t] / max(1e-9, cfg.heat_pipe_cap_share * sc.heat_demand_level)) for t in range(T)) if T > 0 else float("nan")
        ),

        "el_cost_eur": el_cost_val,
        "baseline_heat_resource_cost_eur": baseline_heat_resource_cost,
        "resource_cost_other_eur": other_resource_cost,
        "cooling_resource_cost_eur": cooling_resource_cost,
        "storage_loss_resource_cost_eur": storage_resource_cost,
        "congestion_direct_cost_eur": 0.0,
        "co2_cost_eur": co2_cost_val,
        "power_co2_cost_eur": power_co2_cost_val,
        "baseline_heat_co2_cost_eur": baseline_heat_co2_cost,
        "avoided_heat_co2_cost_eur": avoided_heat_co2_cost,
        "trans_cost_eur": trans_cost_val,
        "demand_charge_eur": demand_charge_val,
        "independent_avoided_heat_cost_eur": independent_avoided_heat_cost,
        "independent_avoided_flex_cost_eur": independent_avoided_flex_cost,
        "heat_revenue_eur": heat_revenue_val,
        "flex_revenue_eur": flex_revenue_val,
        "independent_carbon_damage_eur": independent_carbon_damage,
        "independent_peak_capacity_cost_eur": independent_peak_capacity_cost,
        "peak_import_mw": peak_import_mw,
        "annual_market_operating_cost_eur": annual_market_operating_cost,
        "avoided_heat_value_eur": independent_avoided_heat_cost,
        "avoided_flex_value_eur": independent_avoided_flex_cost,
        "sla_penalty_eur": sla_penalty_val,
        "land_m2": land_m2_val,
        "existing_core_land_m2": land_components_val["existing_core_m2"],
        "existing_cooling_land_m2": land_components_val["existing_cooling_m2"],
        "integration_land_m2": land_components_val["integration_m2"],
        "land_limit_m2": land_limit_val,
        "land_utilization_pct": land_utilization_pct,
        "land_constraint_binding": land_binding,
        "land_feasible": land_feasible,
        "land_cost_eur": land_cost_val,
        "capex_eur": capex,
        "eff_capex_eur": eff_capex_val,
        "eff_benefit_eur": eff_benefit_val,
        "bat_charge_mwh": bat_charge_mwh,
        "bat_discharge_mwh": bat_discharge_mwh,

        "allow_heat": allow_heat,
        "allow_flex": allow_flex,
        "allow_battery": allow_battery,
        "eff_reduction": eff_reduction,
    }
    if _compute_congestion_marginal and relief_fraction > 0.0 and congestion_hours > 0.0:
        relaxed_cfg = replace(
            cfg,
            auto_calibrate_line_limits=False,
            auto_calibrate_sipue_reference_price=False,
            line_cap_12=cfg.line_cap_12 * (1.0 + relief_fraction),
            line_cap_23=cfg.line_cap_23 * (1.0 + relief_fraction),
            line_cap_13=cfg.line_cap_13 * (1.0 + relief_fraction),
        )
        relaxed = solve_ops_given_design(
            sc, prof, relaxed_cfg,
            K_HX, K_FLEX, K_BAT_E, K_BAT_P,
            eff_reduction=eff_reduction,
            K_HP=K_HP,
            _compute_congestion_marginal=False,
        )
        marginal_value = max(
            0.0,
            float(result["market_planning_cost_npv_eur"])
            - float(relaxed["market_planning_cost_npv_eur"]),
        ) * (0.01 / relief_fraction)
        result["congestion_marginal_value_eur_per_1pct"] = marginal_value
    return result



def solve_perfect_foresight(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Tuple[float, Dict]:
    T = len(prof["it_base"])
    dt = 1.0
    w = prof.get("w", [1.0] * T)
    if len(w) != T:
        raise ValueError("Profile weights length does not match horizon length.")
    rep_day_starts = prof.get("rep_day_starts", [])

    it_base = prof["it_base"]
    it_demand = prof.get("it_demand", it_base)
    p_heat = prof["p_heat"]
    p_flex = prof["p_flex"]
    dh_cap = prof["dh_cap"]
    p_co2 = prof["p_co2"]
    cop_t = prof.get("cop_t", [sc.hp_cop] * T)
    node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
    ren_avail = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
    fuel_price = prof.get("fuel_price", [0.0] * T)
    dc_node = prof.get("dc_node", cfg.network_nodes[0])
    cap12, cap23, cap13, b12, b23, b13 = _network_line_parameters(sc, cfg)

    allow_heat = bool(prof.get("allow_heat", True))
    allow_flex = bool(prof.get("allow_flex", True))
    allow_battery = bool(prof.get("allow_battery", True))
    allow_it_shortfall_pf = False

    alpha_base = prof.get("alpha_base", None)
    if alpha_base is None:
        alpha_base = make_overhead_alpha_base(sc, T, cfg)

    p_heat_eff = p_heat
    dh_cap_eff = dh_cap  # interpreted as DH market size in the endogenous value model

    K = int(cfg.dh_segments)
    if K <= 0:
        raise ValueError("dh_segments must be >= 1")
    if len(cfg.dh_seg_fracs) != K or len(cfg.dh_value_multipliers) != K:
        raise ValueError("dh_seg_fracs and dh_value_multipliers must have length dh_segments")
    if abs(sum(cfg.dh_seg_fracs) - 1.0) > 1e-6:
        raise ValueError("dh_seg_fracs must sum to 1.0")

    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    def solve_fixed_eff(eff_reduction: float) -> Tuple[float, Dict]:
        eff_reduction = max(0.0, min(0.90, eff_reduction))
        m = pulp.LpProblem(f"PF_{sc.sid}_eff{eff_reduction:.2f}", pulp.LpMinimize)

        K_HX = pulp.LpVariable("K_HX", lowBound=0, upBound=cfg.K_HX_max)
        K_HP = pulp.LpVariable("K_HP", lowBound=0, upBound=cfg.K_HP_max)
        K_FLEX = pulp.LpVariable("K_FLEX", lowBound=0, upBound=cfg.K_FLEX_max)
        bat_e_bound = (
            cfg.battery_capacity_headroom_multiplier * cfg.min_bat_e_hours * sc.base_it_mw
            if cfg.min_bat_e_hours > 0 else cfg.K_BAT_E_max
        )
        bat_p_bound = (
            cfg.battery_capacity_headroom_multiplier * cfg.min_bat_p_mw_per_it_mw * sc.base_it_mw
            if cfg.min_bat_p_mw_per_it_mw > 0 else cfg.K_BAT_P_max
        )
        K_BAT_E = pulp.LpVariable("K_BAT_E", lowBound=0, upBound=bat_e_bound)
        K_BAT_P = pulp.LpVariable("K_BAT_P", lowBound=0, upBound=bat_p_bound)
        if allow_heat and cfg.min_hx_mwth_per_it_mw > 0:
            min_hx = cfg.min_hx_mwth_per_it_mw * sc.base_it_mw
            m += K_HX >= min_hx, "MinHX"
            m += K_HP >= min_hx * _hp_source_to_power_ratio(sc.hp_cop), "MinHP"
        if allow_flex and cfg.min_flex_mw_per_it_mw > 0:
            min_flex = cfg.min_flex_mw_per_it_mw * sc.base_it_mw
            m += K_FLEX >= min_flex, "MinFlex"
        if allow_battery:
            if cfg.min_bat_e_hours > 0:
                min_bat_e = cfg.min_bat_e_hours * sc.base_it_mw
                m += K_BAT_E >= min_bat_e, "MinBatE"
            if cfg.min_bat_p_mw_per_it_mw > 0:
                min_bat_p = cfg.min_bat_p_mw_per_it_mw * sc.base_it_mw
                m += K_BAT_P >= min_bat_p, "MinBatP"

        IT_SERV = pulp.LpVariable.dicts("IT_SERV", range(T), lowBound=0)
        COOL = pulp.LpVariable.dicts("COOL", range(T), lowBound=0)

        H_REC = pulp.LpVariable.dicts("H_REC", range(T), lowBound=0)
        H_REC_IT = pulp.LpVariable.dicts("H_REC_IT", range(T), lowBound=0)
        H_REC_COOL = pulp.LpVariable.dicts("H_REC_COOL", range(T), lowBound=0)
        H_FLOW = pulp.LpVariable.dicts("H_FLOW", range(T), lowBound=0)
        H_IN = pulp.LpVariable.dicts("H_IN", range(T), lowBound=0)
        H_SOLD = pulp.LpVariable.dicts("H_SOLD", range(T), lowBound=0)
        P_HP = pulp.LpVariable.dicts("P_HP", range(T), lowBound=0)
        Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}", range(T), lowBound=0) for k in range(K)}

        RF_LOAD = pulp.LpVariable.dicts("RF_LOAD", range(T), lowBound=0)
        RF_BAT = pulp.LpVariable.dicts("RF_BAT", range(T), lowBound=0)
        RF = pulp.LpVariable.dicts("RF", range(T), lowBound=0)
        DC_LOAD = pulp.LpVariable.dicts("DC_LOAD", range(T), lowBound=0)
        P_PEAK = pulp.LpVariable("P_PEAK", lowBound=0)

        CH = pulp.LpVariable.dicts("CH", range(T), lowBound=0)
        DIS = pulp.LpVariable.dicts("DIS", range(T), lowBound=0)
        SOC = pulp.LpVariable.dicts("SOC", range(T + 1), lowBound=0)
        SLA_SHORT = pulp.LpVariable.dicts("SLA_SHORT", range(T), lowBound=0)
        SLA_SHORT_EFF = pulp.LpVariable.dicts("SLA_SHORT_EFF", range(T), lowBound=0)

        if cfg.enable_thermal_storage:
            CH_TH = pulp.LpVariable.dicts("CH_TH", range(T), lowBound=0)
            DIS_TH = pulp.LpVariable.dicts("DIS_TH", range(T), lowBound=0)
            S_TH = pulp.LpVariable.dicts("S_TH", range(T + 1), lowBound=0)
            K_TH_STORE = cfg.k_th_store_share_of_daily_peak * sc.heat_demand_level
            P_TH_CH = cfg.th_store_p_ch_share * sc.heat_demand_level
            P_TH_DIS = cfg.th_store_p_dis_share * sc.heat_demand_level
        else:
            CH_TH = {t: 0 for t in range(T)}
            DIS_TH = {t: 0 for t in range(T)}
            S_TH = {t: 0 for t in range(T + 1)}
            K_TH_STORE = 0.0
            P_TH_CH = 0.0
            P_TH_DIS = 0.0

        G_REN = {n: pulp.LpVariable.dicts(f"G_REN_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
        G_TH = {n: pulp.LpVariable.dicts(f"G_TH_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
        F12 = pulp.LpVariable.dicts("F12", range(T), lowBound=None)
        F23 = pulp.LpVariable.dicts("F23", range(T), lowBound=None)
        F13 = pulp.LpVariable.dicts("F13", range(T), lowBound=None)
        THETA = {n: pulp.LpVariable.dicts(f"THETA_{n}", range(T), lowBound=None) for n in cfg.network_nodes}
        U_GRID = pulp.LpVariable.dicts("U_GRID", range(T), lowBound=0, upBound=1)
        if cfg.transmission_cost_eur_per_mwh > 0:
            AF12 = pulp.LpVariable.dicts("AF12", range(T), lowBound=0)
            AF23 = pulp.LpVariable.dicts("AF23", range(T), lowBound=0)
            AF13 = pulp.LpVariable.dicts("AF13", range(T), lowBound=0)
        else:
            AF12 = {t: 0 for t in range(T)}
            AF23 = {t: 0 for t in range(T)}
            AF13 = {t: 0 for t in range(T)}

        ramp_share = pulp.LpVariable("ramp_share", lowBound=0, upBound=1)

        if not allow_heat:
            m += K_HX == 0, "DisableHeat"
            m += K_HP == 0, "DisableHeatPump"
        if not allow_flex:
            m += K_FLEX == 0, "DisableFlex"
        if not allow_battery:
            m += K_BAT_E == 0, "DisableBatteryE"
            m += K_BAT_P == 0, "DisableBatteryP"

        if not cfg.fix_it_profile:
            m += pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T)) == sum(it_base[t] * dt * w[t] for t in range(T)), "WorkConservation"

        m += SOC[0] == 0.5 * K_BAT_E, "SOC_start"
        m += SOC[T] == SOC[0], "SOC_cycle"

        for start in rep_day_starts:
            end = start + 24
            if 0 <= start < T and end <= T:
                m += SOC[end] == SOC[start]
                if cfg.enable_thermal_storage:
                    m += S_TH[end] == S_TH[start]

        if cfg.enable_thermal_storage:
            m += S_TH[0] == 0.0, "S_TH_start"
            m += S_TH[T] == 0.0, "S_TH_end"

        n1, n2, n3 = cfg.network_nodes

        m += K_BAT_P <= K_BAT_E, "BatteryPowerLeqEnergy"

        size_proxy_expr = (
            sc.base_it_mw
            + cfg.ramp_k_hx * K_HX
            + cfg.ramp_k_hp * K_HP
            + cfg.ramp_k_flex * K_FLEX
            + cfg.ramp_k_bat_e * K_BAT_E
            + cfg.ramp_k_bat_p * K_BAT_P
        )
        m += ramp_share <= cfg.ramp_base_share - cfg.ramp_tau_per_mw * size_proxy_expr, "RampShareUb"

        alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

        cooling_capacity_mw = _existing_cooling_capacity_mw(sc, alpha_base)
        land_m2_expr = compute_land_m2(
            K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg,
            K_HP=K_HP, base_it_mw=sc.base_it_mw, K_COOL=cooling_capacity_mw,
        )
        land_limit = effective_land_limit_m2(sc, cfg)
        if land_limit is not None:
            m += land_m2_expr <= land_limit, "LandLimit"
        land_cost_expr = sc.land_cost_multiplier * land_cost_horizon(land_m2_expr, cfg)
        force_it_flex = bool(prof.get("force_it_flex", False))
        flex_frac_eff = min(0.30, sc.flex_frac * (cfg.it_flex_scale_services_active if allow_flex else 1.0))
        for t in range(T):
            m += THETA[n1][t] == 0.0
            m += F12[t] == b12 * (THETA[n1][t] - THETA[n2][t])
            m += F23[t] == b23 * (THETA[n2][t] - THETA[n3][t])
            m += F13[t] == b13 * (THETA[n1][t] - THETA[n3][t])
            m += F12[t] <= cap12
            m += -F12[t] <= cap12
            m += F23[t] <= cap23
            m += -F23[t] <= cap23
            m += F13[t] <= cap13
            m += -F13[t] <= cap13
            if allow_it_shortfall_pf:
                m += IT_SERV[t] <= it_demand[t]
                m += IT_SERV[t] >= (1.0 - cfg.max_it_curtail_frac) * it_demand[t]
            elif force_it_flex:
                lb = (1 - flex_frac_eff) * it_base[t]
                ub = (1 + flex_frac_eff) * it_base[t]
                m += IT_SERV[t] >= lb
                m += IT_SERV[t] <= ub
            elif cfg.fix_it_profile:
                m += IT_SERV[t] == it_base[t]
            elif not cfg.fix_it_profile:
                lb = (1 - flex_frac_eff) * it_base[t]
                ub = (1 + flex_frac_eff) * it_base[t]
                m += IT_SERV[t] >= lb
                m += IT_SERV[t] <= ub
            else:
                it_cap = sc.base_it_mw * cfg.it_capacity_multiplier
                m += IT_SERV[t] <= it_demand[t] * ramp_share
                m += IT_SERV[t] <= it_cap

            m += COOL[t] == alpha_eff[t] * IT_SERV[t]

            if allow_heat:
                pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
                delivered_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_delivered_ratio(cop_t[t])
                hp_power_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_power_ratio(cop_t[t])
                m += H_REC[t] == H_REC_IT[t] + H_REC_COOL[t]
                m += H_REC_IT[t] == H_REC[t] * (1.0 / max(1e-9, 1.0 + alpha_eff[t]))
                m += H_REC_COOL[t] == H_REC[t] * alpha_eff[t] / max(1e-9, 1.0 + alpha_eff[t])
                m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
                m += H_REC[t] <= K_HX
                m += H_FLOW[t] <= H_REC[t]
                m += H_FLOW[t] <= pipe_cap
                m += H_IN[t] == delivered_ratio * H_FLOW[t]
                m += P_HP[t] == hp_power_ratio * H_FLOW[t]
                m += P_HP[t] <= K_HP

                m += H_SOLD[t] == pulp.lpSum(Q_DH[k][t] for k in range(K))
                for k in range(K):
                    m += Q_DH[k][t] <= cfg.dh_seg_fracs[k] * dh_cap_eff[t]

                if cfg.enable_thermal_storage:
                    m += S_TH[t] <= K_TH_STORE
                    m += S_TH[t + 1] <= K_TH_STORE
                    m += CH_TH[t] <= P_TH_CH
                    m += DIS_TH[t] <= P_TH_DIS
                    m += (
                        S_TH[t + 1]
                        == (1.0 - cfg.th_store_loss_per_hour) * S_TH[t]
                        + cfg.eta_th_ch * CH_TH[t] * dt
                        - (1.0 / cfg.eta_th_dis) * DIS_TH[t] * dt
                    )
                    m += H_IN[t] + DIS_TH[t] == H_SOLD[t] + CH_TH[t]
                else:
                    m += H_IN[t] == H_SOLD[t]
            else:
                m += H_REC[t] == 0
                m += H_REC_IT[t] == 0
                m += H_REC_COOL[t] == 0
                m += H_FLOW[t] == 0
                m += H_IN[t] == 0
                m += H_SOLD[t] == 0
                m += P_HP[t] == 0
                for k in range(K):
                    m += Q_DH[k][t] == 0
                if cfg.enable_thermal_storage:
                    m += CH_TH[t] == 0
                    m += DIS_TH[t] == 0
                    m += S_TH[t + 1] == S_TH[t]

            flex_reference = it_demand[t]
            m += SOC[t] <= K_BAT_E
            m += SOC[t + 1] <= K_BAT_E
            if allow_battery:
                m += CH[t] <= K_BAT_P
                m += DIS[t] <= K_BAT_P
                m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1.0 / cfg.eta_dis) * DIS[t] * dt
                reserve_duration = max(1e-9, cfg.battery_flex_reserve_duration_h)
                ups_reserve = max(0.0, min(1.0, cfg.battery_ups_reserve_fraction)) * K_BAT_E
                m += SOC[t] >= ups_reserve
                m += SOC[t + 1] >= ups_reserve
                m += RF_BAT[t] <= K_BAT_P - DIS[t]
                m += RF_BAT[t] <= (SOC[t] - ups_reserve) / reserve_duration
            else:
                m += CH[t] == 0
                m += DIS[t] == 0
                m += SOC[t + 1] == SOC[t]
                m += RF_BAT[t] == 0

            if allow_flex:
                m += RF_LOAD[t] <= K_FLEX
                m += RF_LOAD[t] <= flex_frac_eff * flex_reference
                m += IT_SERV[t] + RF_LOAD[t] <= flex_reference
            else:
                m += RF_LOAD[t] == 0
            if not allow_flex:
                m += RF_BAT[t] == 0
            m += RF[t] == RF_LOAD[t] + RF_BAT[t]

            P_SERV = (
                cfg.service_parasitic_heat_frac * H_SOLD[t]
                + cfg.service_parasitic_flex_frac * RF[t]
                + cfg.service_parasitic_hx_mw_per_mwth * K_HX
                + cfg.service_parasitic_flex_mw_per_mw * K_FLEX
            )

            m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP[t] + P_SERV + CH[t] - DIS[t]
            if allow_it_shortfall_pf:
                dc_grid_cap_t = cfg.dc_grid_cap_share * it_demand[t] * (1.0 + alpha_eff[t])
                m += DC_LOAD[t] <= dc_grid_cap_t

            if allow_it_shortfall_pf:
                m += SLA_SHORT[t] >= it_demand[t] - IT_SERV[t]
                m += SLA_SHORT[t] <= it_demand[t]
                if cfg.sla_penalty_only_when_congested:
                    m += SLA_SHORT_EFF[t] <= SLA_SHORT[t]
                    m += SLA_SHORT_EFF[t] <= it_demand[t] * U_GRID[t]
                    m += SLA_SHORT_EFF[t] >= SLA_SHORT[t] - it_demand[t] * (1.0 - U_GRID[t])
                else:
                    m += SLA_SHORT_EFF[t] == SLA_SHORT[t]
            else:
                m += SLA_SHORT[t] == 0
                m += SLA_SHORT_EFF[t] == 0
            m += P_PEAK >= DC_LOAD[t]

            dc_at_n1 = DC_LOAD[t] if dc_node == n1 else 0.0
            dc_at_n2 = DC_LOAD[t] if dc_node == n2 else 0.0
            dc_at_n3 = DC_LOAD[t] if dc_node == n3 else 0.0

            m += G_REN[n1][t] <= ren_avail[n1][t]
            m += G_REN[n2][t] <= ren_avail[n2][t]
            m += G_REN[n3][t] <= ren_avail[n3][t]

            m += G_TH[n1][t] <= cfg.th_cap_N1_mw
            m += G_TH[n2][t] <= cfg.th_cap_N2_mw
            m += G_TH[n3][t] <= cfg.th_cap_N3_mw

            m += (G_REN[n1][t] + G_TH[n1][t]) - F12[t] - F13[t] == node_loads[n1][t] + dc_at_n1
            m += (G_REN[n2][t] + G_TH[n2][t]) + F12[t] - F23[t] == node_loads[n2][t] + dc_at_n2
            m += (G_REN[n3][t] + G_TH[n3][t]) + F13[t] + F23[t] == node_loads[n3][t] + dc_at_n3

            if cfg.transmission_cost_eur_per_mwh > 0:
                m += AF12[t] >= F12[t]
                m += AF12[t] >= -F12[t]
                m += AF23[t] >= F23[t]
                m += AF23[t] >= -F23[t]
                m += AF13[t] >= F13[t]
                m += AF13[t] >= -F13[t]

            m += U_GRID[t] >= F12[t] * (1.0 / cap12)
            m += U_GRID[t] >= -F12[t] * (1.0 / cap12)
            m += U_GRID[t] >= F23[t] * (1.0 / cap23)
            m += U_GRID[t] >= -F23[t] * (1.0 / cap23)
            m += U_GRID[t] >= F13[t] * (1.0 / cap13)
            m += U_GRID[t] >= -F13[t] * (1.0 / cap13)

        el_cost_expr = pulp.lpSum(
            (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
            for n in cfg.network_nodes for t in range(T)
        )
        power_emissions_expr = pulp.lpSum(
            th_ef[n] * G_TH[n][t] * dt * w[t]
            for n in cfg.network_nodes for t in range(T)
        )
        power_co2_cost_expr = pulp.lpSum(
            p_co2[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
            for n in cfg.network_nodes for t in range(T)
        )
        trans_cost_expr = 0
        if cfg.transmission_cost_eur_per_mwh > 0:
            trans_cost_expr = pulp.lpSum(
                cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
                for t in range(T)
            )

        avoided_heat_price = prof.get("avoided_heat_cost", p_heat_eff)
        heat_emissions_factor = (
            cfg.independent_heat_boiler_emissions_tco2_per_mwh_fuel
            / max(1e-9, cfg.independent_heat_boiler_efficiency)
        )
        baseline_heat_resource_cost_expr = pulp.lpSum(
            avoided_heat_price[t] * dh_cap_eff[t] * dt * w[t] for t in range(T)
        )
        baseline_heat_emissions_expr = pulp.lpSum(
            heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
        )
        avoided_heat_emissions_expr = pulp.lpSum(
            heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
        )
        emissions_expr = power_emissions_expr + baseline_heat_emissions_expr - avoided_heat_emissions_expr
        baseline_heat_co2_cost_expr = pulp.lpSum(
            p_co2[t] * heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
        )
        avoided_heat_co2_cost_expr = pulp.lpSum(
            p_co2[t] * heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
        )
        co2_cost_expr = power_co2_cost_expr + baseline_heat_co2_cost_expr - avoided_heat_co2_cost_expr
        avoided_flex_price = prof.get(
            "avoided_flex_cost",
            [cfg.independent_avoided_flex_cost_eur_per_mw_h] * T,
        )
        heat_revenue_expr = pulp.lpSum(
            p_heat_eff[t] * cfg.dh_value_multipliers[k] * Q_DH[k][t] * dt * w[t]
            for k in range(K) for t in range(T)
        )
        flex_revenue_expr = pulp.lpSum(
            p_flex[t] * RF[t] * dt * w[t] for t in range(T)
        )
        demand_charge_expr = demand_charge_horizon(cfg) * P_PEAK

        if allow_it_shortfall_pf:
            sla_penalty_expr = pulp.lpSum(cfg.sla_penalty_eur_per_mwh * SLA_SHORT_EFF[t] * dt * w[t] for t in range(T))
        else:
            sla_penalty_expr = 0
        E_IT_expr = pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T))
        E_FAC_expr = pulp.lpSum(DC_LOAD[t] * dt * w[t] for t in range(T))
        eff_benefit_expr = eff_reduction * (
            cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC_expr
            + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT_expr
        )

        capex_expr = sc.capex_multiplier * capex_scale(cfg) * (
            cfg.capex_HX * K_HX
            + cfg.capex_HP * K_HP
            + cfg.capex_FLEX * K_FLEX
            + cfg.capex_BAT_E * K_BAT_E
            + cfg.capex_BAT_P * K_BAT_P
            + eff_capex_from_reduction(eff_reduction, cfg)
        )

        operating_pv_factor, land_pv_factor = _lifecycle_factors(cfg, sc)
        lifecycle_cost_expr = (
            operating_pv_factor * (
                el_cost_expr + baseline_heat_resource_cost_expr + co2_cost_expr + trans_cost_expr + demand_charge_expr
                + sla_penalty_expr - heat_revenue_expr - flex_revenue_expr
                - eff_benefit_expr
            )
            + capex_expr + land_pv_factor * land_cost_expr
        )
        carbon_cap = (cfg.carbon_emissions_cap_tco2_by_sid or {}).get(sc.sid)
        if cfg.enforce_carbon_cap and carbon_cap is not None:
            m += emissions_expr <= float(carbon_cap), "PhysicalCarbonCap"
        m += lifecycle_cost_expr + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))

        m.solve(make_solver_lp(cfg))

        return pulp.value(m.objective), {
            "K_HX": pulp.value(K_HX),
            "K_HP": pulp.value(K_HP),
            "K_FLEX": pulp.value(K_FLEX),
            "K_BAT_E": pulp.value(K_BAT_E),
            "K_BAT_P": pulp.value(K_BAT_P),
            "status": pulp.LpStatus[m.status],
        }

    best_obj = float("inf")
    best = {}
    best_idx = 0
    if cfg.fixed_eff_reduction is not None:
        eff = float(cfg.fixed_eff_reduction)
        obj, des = solve_fixed_eff(eff)
        best_obj = obj
        best = des
        best_idx = min(range(len(cfg.eff_levels)), key=lambda i: abs(cfg.eff_levels[i] - eff))
    else:
        for i, eff in enumerate(cfg.eff_levels):
            obj, des = solve_fixed_eff(eff)
            if obj < best_obj:
                best_obj = obj
                best = des
                best_idx = i

    pf_ops = solve_ops_given_design(
        sc, prof, cfg,
        best["K_HX"],
        best["K_FLEX"],
        best["K_BAT_E"],
        best["K_BAT_P"],
        eff_reduction=cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx],
        K_HP=best["K_HP"],
    )

    return best_obj, {
        "K_HX": best["K_HX"],
        "K_HP": best["K_HP"],
        "K_FLEX": best["K_FLEX"],
        "K_BAT_E": best["K_BAT_E"],
        "K_BAT_P": best["K_BAT_P"],
        "eff_level_index": best_idx,
        "eff_reduction": cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx],
        "eff_capex": eff_capex_from_reduction(
            cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx], cfg
        ),
        "status": best["status"],
        "market_planning_cost_npv_eur": pf_ops.get("market_planning_cost_npv_eur"),
    }


def solve_minmax_regret(
    scenarios: List[Scenario],
    profiles: Dict[int, Dict[str, List[float]]],
    C_star: Dict[int, float],
    cfg: Config,
) -> Dict:
    if cfg.robust_objective not in {
        "normalized_lexicographic", "lexicographic_minmax", "pure_minmax", "expected_value"
    }:
        raise ValueError(
            "robust_objective must be normalized lexicographic, absolute lexicographic, pure min-max, or expected value."
        )
    K = int(cfg.dh_segments)
    if K <= 0:
        raise ValueError("dh_segments must be >= 1")
    if len(cfg.dh_seg_fracs) != K or len(cfg.dh_value_multipliers) != K:
        raise ValueError("dh_seg_fracs and dh_value_multipliers must have length dh_segments")
    if abs(sum(cfg.dh_seg_fracs) - 1.0) > 1e-6:
        raise ValueError("dh_seg_fracs must sum to 1.0")

    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    if not scenarios:
        raise ValueError("Min-max regret requires at least one scenario.")
    max_it_mw = max(sc.base_it_mw for sc in scenarios)
    heat_enabled = any(bool(profiles[sc.sid].get("allow_heat", True)) for sc in scenarios)
    flex_enabled = any(bool(profiles[sc.sid].get("allow_flex", True)) for sc in scenarios)
    battery_enabled = any(bool(profiles[sc.sid].get("allow_battery", True)) for sc in scenarios)
    # One installed heat-pump rating must remain feasible in every uncertainty
    # state. The lowest COP requires the greatest electrical input capacity.
    min_cop = min(
        (sc.hp_cop for sc in scenarios if bool(profiles[sc.sid].get("allow_heat", True))),
        default=1.1,
    )

    def solve_fixed_eff(eff_reduction: float) -> Dict:
        eff_reduction = max(0.0, min(0.90, eff_reduction))
        m = pulp.LpProblem(f"MinMaxRegret_T{cfg.T}_eff{eff_reduction:.2f}", pulp.LpMinimize)

        k_HX = pulp.LpVariable("k_HX_per_MW_IT", lowBound=0, upBound=cfg.K_HX_max / max_it_mw)
        k_HP = pulp.LpVariable("k_HP_per_MW_IT", lowBound=0, upBound=cfg.K_HP_max / max_it_mw)
        k_FLEX = pulp.LpVariable("k_FLEX_per_MW_IT", lowBound=0, upBound=cfg.K_FLEX_max / max_it_mw)
        # Bounded by a headroom multiplier on the min_bat_* floor (both already
        # per-MW-IT ratios) rather than an absolute cap divided by max_it_mw --
        # the latter gives smaller archetypes a *larger* per-MW-IT ceiling,
        # backwards from what a facility-size-invariant bound should do.
        k_bat_e_bound = (
            cfg.battery_capacity_headroom_multiplier * cfg.min_bat_e_hours
            if cfg.min_bat_e_hours > 0 else cfg.K_BAT_E_max / max_it_mw
        )
        k_bat_p_bound = (
            cfg.battery_capacity_headroom_multiplier * cfg.min_bat_p_mw_per_it_mw
            if cfg.min_bat_p_mw_per_it_mw > 0 else cfg.K_BAT_P_max / max_it_mw
        )
        k_BAT_E = pulp.LpVariable("k_BAT_E_per_MW_IT", lowBound=0, upBound=k_bat_e_bound)
        k_BAT_P = pulp.LpVariable("k_BAT_P_per_MW_IT", lowBound=0, upBound=k_bat_p_bound)

        if heat_enabled and cfg.min_hx_mwth_per_it_mw > 0:
            m += k_HX >= cfg.min_hx_mwth_per_it_mw, "MinHXIntensity"
            m += k_HP >= cfg.min_hx_mwth_per_it_mw * _hp_source_to_power_ratio(min_cop), "MinHPIntensity"
        if flex_enabled and cfg.min_flex_mw_per_it_mw > 0:
            m += k_FLEX >= cfg.min_flex_mw_per_it_mw, "MinFlexIntensity"
        if battery_enabled and cfg.min_bat_e_hours > 0:
            m += k_BAT_E >= cfg.min_bat_e_hours, "MinBatEIntensity"
        if battery_enabled and cfg.min_bat_p_mw_per_it_mw > 0:
            m += k_BAT_P >= cfg.min_bat_p_mw_per_it_mw, "MinBatPIntensity"

        Z = pulp.LpVariable("Z", lowBound=0)

        m += k_BAT_P <= k_BAT_E, "BatteryPowerLeqEnergyIntensity"

        Cost_s_vars = {}

        for sc in scenarios:
            prof = profiles[sc.sid]
            T = len(prof["it_base"])
            dt = 1.0
            w = prof.get("w", [1.0] * T)
            if len(w) != T:
                raise ValueError("Profile weights length does not match horizon length.")
            rep_day_starts = prof.get("rep_day_starts", [])

            it_base = prof["it_base"]
            it_demand = prof.get("it_demand", it_base)
            p_heat = prof["p_heat"]
            p_flex = prof["p_flex"]
            dh_cap = prof["dh_cap"]
            p_co2 = prof["p_co2"]
            cop_t = prof.get("cop_t", [sc.hp_cop] * T)
            node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
            ren_avail = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
            fuel_price = prof.get("fuel_price", [0.0] * T)
            dc_node = prof.get("dc_node", cfg.network_nodes[0])
            cap12, cap23, cap13, b12, b23, b13 = _network_line_parameters(sc, cfg)

            allow_heat = bool(prof.get("allow_heat", True))
            allow_flex = bool(prof.get("allow_flex", True))
            allow_battery = bool(prof.get("allow_battery", True))
            allow_it_shortfall_sc = False

            alpha_base = prof.get("alpha_base", None)
            if alpha_base is None:
                alpha_base = make_overhead_alpha_base(sc, T, cfg)
            alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

            K_HX_s = k_HX * sc.base_it_mw
            K_HP_s = k_HP * sc.base_it_mw
            K_FLEX_s = k_FLEX * sc.base_it_mw
            K_BAT_E_s = k_BAT_E * sc.base_it_mw
            K_BAT_P_s = k_BAT_P * sc.base_it_mw
            cooling_capacity_mw = _existing_cooling_capacity_mw(sc, alpha_base)
            land_m2_expr = compute_land_m2(
                K_HX_s, K_FLEX_s, K_BAT_E_s, K_BAT_P_s, eff_reduction, cfg,
                K_HP=K_HP_s, base_it_mw=sc.base_it_mw, K_COOL=cooling_capacity_mw,
            )
            land_limit = effective_land_limit_m2(sc, cfg)
            if land_limit is not None:
                m += land_m2_expr <= land_limit, f"LandLimit_s{sc.sid}"
            land_cost_expr = sc.land_cost_multiplier * land_cost_horizon(land_m2_expr, cfg)

            p_heat_eff = p_heat
            dh_cap_eff = dh_cap

            IT_SERV = pulp.LpVariable.dicts(f"IT_SERV_s{sc.sid}", range(T), lowBound=0)
            COOL = pulp.LpVariable.dicts(f"COOL_s{sc.sid}", range(T), lowBound=0)

            H_REC = pulp.LpVariable.dicts(f"H_REC_s{sc.sid}", range(T), lowBound=0)
            H_REC_IT = pulp.LpVariable.dicts(f"H_REC_IT_s{sc.sid}", range(T), lowBound=0)
            H_REC_COOL = pulp.LpVariable.dicts(f"H_REC_COOL_s{sc.sid}", range(T), lowBound=0)
            H_FLOW = pulp.LpVariable.dicts(f"H_FLOW_s{sc.sid}", range(T), lowBound=0)
            H_IN = pulp.LpVariable.dicts(f"H_IN_s{sc.sid}", range(T), lowBound=0)
            H_SOLD = pulp.LpVariable.dicts(f"H_SOLD_s{sc.sid}", range(T), lowBound=0)
            P_HP = pulp.LpVariable.dicts(f"P_HP_s{sc.sid}", range(T), lowBound=0)
            Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}_s{sc.sid}", range(T), lowBound=0) for k in range(K)}

            RF_LOAD = pulp.LpVariable.dicts(f"RF_LOAD_s{sc.sid}", range(T), lowBound=0)
            RF_BAT = pulp.LpVariable.dicts(f"RF_BAT_s{sc.sid}", range(T), lowBound=0)
            RF = pulp.LpVariable.dicts(f"RF_s{sc.sid}", range(T), lowBound=0)
            DC_LOAD = pulp.LpVariable.dicts(f"DC_LOAD_s{sc.sid}", range(T), lowBound=0)
            P_PEAK = pulp.LpVariable(f"P_PEAK_s{sc.sid}", lowBound=0)

            CH = pulp.LpVariable.dicts(f"CH_s{sc.sid}", range(T), lowBound=0)
            DIS = pulp.LpVariable.dicts(f"DIS_s{sc.sid}", range(T), lowBound=0)
            SOC = pulp.LpVariable.dicts(f"SOC_s{sc.sid}", range(T + 1), lowBound=0)
            SLA_SHORT = pulp.LpVariable.dicts(f"SLA_SHORT_s{sc.sid}", range(T), lowBound=0)
            SLA_SHORT_EFF = pulp.LpVariable.dicts(f"SLA_SHORT_EFF_s{sc.sid}", range(T), lowBound=0)

            if cfg.enable_thermal_storage:
                CH_TH = pulp.LpVariable.dicts(f"CH_TH_s{sc.sid}", range(T), lowBound=0)
                DIS_TH = pulp.LpVariable.dicts(f"DIS_TH_s{sc.sid}", range(T), lowBound=0)
                S_TH = pulp.LpVariable.dicts(f"S_TH_s{sc.sid}", range(T + 1), lowBound=0)
                K_TH_STORE = cfg.k_th_store_share_of_daily_peak * sc.heat_demand_level
                P_TH_CH = cfg.th_store_p_ch_share * sc.heat_demand_level
                P_TH_DIS = cfg.th_store_p_dis_share * sc.heat_demand_level
            else:
                CH_TH = {t: 0 for t in range(T)}
                DIS_TH = {t: 0 for t in range(T)}
                S_TH = {t: 0 for t in range(T + 1)}
                K_TH_STORE = 0.0
                P_TH_CH = 0.0
                P_TH_DIS = 0.0

            G_REN = {n: pulp.LpVariable.dicts(f"G_REN_{n}_s{sc.sid}", range(T), lowBound=0) for n in cfg.network_nodes}
            G_TH = {n: pulp.LpVariable.dicts(f"G_TH_{n}_s{sc.sid}", range(T), lowBound=0) for n in cfg.network_nodes}
            F12 = pulp.LpVariable.dicts(f"F12_s{sc.sid}", range(T), lowBound=None)
            F23 = pulp.LpVariable.dicts(f"F23_s{sc.sid}", range(T), lowBound=None)
            F13 = pulp.LpVariable.dicts(f"F13_s{sc.sid}", range(T), lowBound=None)
            THETA = {n: pulp.LpVariable.dicts(f"THETA_{n}_s{sc.sid}", range(T), lowBound=None) for n in cfg.network_nodes}
            U_GRID = pulp.LpVariable.dicts(f"U_GRID_s{sc.sid}", range(T), lowBound=0, upBound=1)
            if cfg.transmission_cost_eur_per_mwh > 0:
                AF12 = pulp.LpVariable.dicts(f"AF12_s{sc.sid}", range(T), lowBound=0)
                AF23 = pulp.LpVariable.dicts(f"AF23_s{sc.sid}", range(T), lowBound=0)
                AF13 = pulp.LpVariable.dicts(f"AF13_s{sc.sid}", range(T), lowBound=0)
            else:
                AF12 = {t: 0 for t in range(T)}
                AF23 = {t: 0 for t in range(T)}
                AF13 = {t: 0 for t in range(T)}

            ramp_share = pulp.LpVariable(f"ramp_share_s{sc.sid}", lowBound=0, upBound=1)

            Cost = pulp.LpVariable(f"Cost_s{sc.sid}", lowBound=None)
            Cost_s_vars[sc.sid] = Cost

            if not cfg.fix_it_profile:
                m += pulp.lpSum(IT_SERV[t] * dt for t in range(T)) == sum(it_base) * dt

            m += SOC[0] == 0.5 * K_BAT_E_s
            m += SOC[T] == SOC[0]

            if cfg.enable_thermal_storage:
                m += S_TH[0] == 0.0
                m += S_TH[T] == 0.0

            n1, n2, n3 = cfg.network_nodes

            size_proxy_expr = (
                sc.base_it_mw
                + cfg.ramp_k_hx * K_HX_s
                + cfg.ramp_k_hp * K_HP_s
                + cfg.ramp_k_flex * K_FLEX_s
                + cfg.ramp_k_bat_e * K_BAT_E_s
                + cfg.ramp_k_bat_p * K_BAT_P_s
            )
            m += ramp_share <= cfg.ramp_base_share - cfg.ramp_tau_per_mw * size_proxy_expr

            force_it_flex = bool(prof.get("force_it_flex", False))
            flex_frac_eff = min(0.30, sc.flex_frac * (cfg.it_flex_scale_services_active if allow_flex else 1.0))
            for t in range(T):
                m += THETA[n1][t] == 0.0
                m += F12[t] == b12 * (THETA[n1][t] - THETA[n2][t])
                m += F23[t] == b23 * (THETA[n2][t] - THETA[n3][t])
                m += F13[t] == b13 * (THETA[n1][t] - THETA[n3][t])
                m += F12[t] <= cap12
                m += -F12[t] <= cap12
                m += F23[t] <= cap23
                m += -F23[t] <= cap23
                m += F13[t] <= cap13
                m += -F13[t] <= cap13
                if allow_it_shortfall_sc:
                    m += IT_SERV[t] <= it_demand[t]
                    m += IT_SERV[t] >= (1.0 - cfg.max_it_curtail_frac) * it_demand[t]
                elif force_it_flex:
                    lb = (1 - flex_frac_eff) * it_base[t]
                    ub = (1 + flex_frac_eff) * it_base[t]
                    m += IT_SERV[t] >= lb
                    m += IT_SERV[t] <= ub
                elif cfg.fix_it_profile:
                    m += IT_SERV[t] == it_base[t]
                elif not cfg.fix_it_profile:
                    lb = (1 - flex_frac_eff) * it_base[t]
                    ub = (1 + flex_frac_eff) * it_base[t]
                    m += IT_SERV[t] >= lb
                    m += IT_SERV[t] <= ub
                else:
                    it_cap = sc.base_it_mw * cfg.it_capacity_multiplier
                    m += IT_SERV[t] <= it_demand[t] * ramp_share
                    m += IT_SERV[t] <= it_cap

                m += COOL[t] == alpha_eff[t] * IT_SERV[t]

                if allow_heat:
                    pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
                    delivered_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_delivered_ratio(cop_t[t])
                    hp_power_ratio = (1.0 - cfg.heat_pipe_loss_frac) * _hp_source_to_power_ratio(cop_t[t])
                    m += H_REC[t] == H_REC_IT[t] + H_REC_COOL[t]
                    m += H_REC_IT[t] == H_REC[t] * (1.0 / max(1e-9, 1.0 + alpha_eff[t]))
                    m += H_REC_COOL[t] == H_REC[t] * alpha_eff[t] / max(1e-9, 1.0 + alpha_eff[t])
                    m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
                    m += H_REC[t] <= K_HX_s
                    m += H_FLOW[t] <= H_REC[t]
                    m += H_FLOW[t] <= pipe_cap
                    m += H_IN[t] == delivered_ratio * H_FLOW[t]
                    m += P_HP[t] == hp_power_ratio * H_FLOW[t]
                    m += P_HP[t] <= K_HP_s

                    m += H_SOLD[t] == pulp.lpSum(Q_DH[k][t] for k in range(K))
                    for k in range(K):
                        m += Q_DH[k][t] <= cfg.dh_seg_fracs[k] * dh_cap_eff[t]

                    if cfg.enable_thermal_storage:
                        m += S_TH[t] <= K_TH_STORE
                        m += S_TH[t + 1] <= K_TH_STORE
                        m += CH_TH[t] <= P_TH_CH
                        m += DIS_TH[t] <= P_TH_DIS
                        m += S_TH[t + 1] == (1.0 - cfg.th_store_loss_per_hour) * S_TH[t] + cfg.eta_th_ch * CH_TH[t] * dt - (1.0 / cfg.eta_th_dis) * DIS_TH[t] * dt
                        m += H_IN[t] + DIS_TH[t] == H_SOLD[t] + CH_TH[t]
                    else:
                        m += H_IN[t] == H_SOLD[t]
                else:
                    m += H_REC[t] == 0
                    m += H_REC_IT[t] == 0
                    m += H_REC_COOL[t] == 0
                    m += H_FLOW[t] == 0
                    m += H_IN[t] == 0
                    m += H_SOLD[t] == 0
                    m += P_HP[t] == 0
                    for k in range(K):
                        m += Q_DH[k][t] == 0
                    if cfg.enable_thermal_storage:
                        m += CH_TH[t] == 0
                        m += DIS_TH[t] == 0
                        m += S_TH[t + 1] == S_TH[t]

                flex_reference = it_demand[t]
                m += SOC[t] <= K_BAT_E_s
                m += SOC[t + 1] <= K_BAT_E_s
                if allow_battery:
                    m += CH[t] <= K_BAT_P_s
                    m += DIS[t] <= K_BAT_P_s
                    m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1.0 / cfg.eta_dis) * DIS[t] * dt
                    reserve_duration = max(1e-9, cfg.battery_flex_reserve_duration_h)
                    ups_reserve = max(0.0, min(1.0, cfg.battery_ups_reserve_fraction)) * K_BAT_E_s
                    m += SOC[t] >= ups_reserve
                    m += SOC[t + 1] >= ups_reserve
                    m += RF_BAT[t] <= K_BAT_P_s - DIS[t]
                    m += RF_BAT[t] <= (SOC[t] - ups_reserve) / reserve_duration
                else:
                    m += CH[t] == 0
                    m += DIS[t] == 0
                    m += SOC[t + 1] == SOC[t]
                    m += RF_BAT[t] == 0

                if allow_flex:
                    m += RF_LOAD[t] <= K_FLEX_s
                    m += RF_LOAD[t] <= flex_frac_eff * flex_reference
                    m += IT_SERV[t] + RF_LOAD[t] <= flex_reference
                else:
                    m += RF_LOAD[t] == 0
                if not allow_flex:
                    m += RF_BAT[t] == 0
                m += RF[t] == RF_LOAD[t] + RF_BAT[t]

                P_SERV = (
                    cfg.service_parasitic_heat_frac * H_SOLD[t]
                    + cfg.service_parasitic_flex_frac * RF[t]
                    + cfg.service_parasitic_hx_mw_per_mwth * K_HX_s
                    + cfg.service_parasitic_flex_mw_per_mw * K_FLEX_s
                )
                m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP[t] + P_SERV + CH[t] - DIS[t]
                m += P_PEAK >= DC_LOAD[t]
                if allow_it_shortfall_sc:
                    dc_grid_cap_t = cfg.dc_grid_cap_share * it_demand[t] * (1.0 + alpha_eff[t])
                    m += DC_LOAD[t] <= dc_grid_cap_t

                if allow_it_shortfall_sc:
                    m += SLA_SHORT[t] >= it_demand[t] - IT_SERV[t]
                    m += SLA_SHORT[t] <= it_demand[t]
                    if cfg.sla_penalty_only_when_congested:
                        m += SLA_SHORT_EFF[t] <= SLA_SHORT[t]
                        m += SLA_SHORT_EFF[t] <= it_demand[t] * U_GRID[t]
                        m += SLA_SHORT_EFF[t] >= SLA_SHORT[t] - it_demand[t] * (1.0 - U_GRID[t])
                    else:
                        m += SLA_SHORT_EFF[t] == SLA_SHORT[t]
                else:
                    m += SLA_SHORT[t] == 0
                    m += SLA_SHORT_EFF[t] == 0

                dc_at_n1 = DC_LOAD[t] if dc_node == n1 else 0.0
                dc_at_n2 = DC_LOAD[t] if dc_node == n2 else 0.0
                dc_at_n3 = DC_LOAD[t] if dc_node == n3 else 0.0

                m += G_REN[n1][t] <= ren_avail[n1][t]
                m += G_REN[n2][t] <= ren_avail[n2][t]
                m += G_REN[n3][t] <= ren_avail[n3][t]

                m += G_TH[n1][t] <= cfg.th_cap_N1_mw
                m += G_TH[n2][t] <= cfg.th_cap_N2_mw
                m += G_TH[n3][t] <= cfg.th_cap_N3_mw

                m += (G_REN[n1][t] + G_TH[n1][t]) - F12[t] - F13[t] == node_loads[n1][t] + dc_at_n1
                m += (G_REN[n2][t] + G_TH[n2][t]) + F12[t] - F23[t] == node_loads[n2][t] + dc_at_n2
                m += (G_REN[n3][t] + G_TH[n3][t]) + F13[t] + F23[t] == node_loads[n3][t] + dc_at_n3

                if cfg.transmission_cost_eur_per_mwh > 0:
                    m += AF12[t] >= F12[t]
                    m += AF12[t] >= -F12[t]
                    m += AF23[t] >= F23[t]
                    m += AF23[t] >= -F23[t]
                    m += AF13[t] >= F13[t]
                    m += AF13[t] >= -F13[t]

                m += U_GRID[t] >= F12[t] * (1.0 / cap12)
                m += U_GRID[t] >= -F12[t] * (1.0 / cap12)
                m += U_GRID[t] >= F23[t] * (1.0 / cap23)
                m += U_GRID[t] >= -F23[t] * (1.0 / cap23)
                m += U_GRID[t] >= F13[t] * (1.0 / cap13)
                m += U_GRID[t] >= -F13[t] * (1.0 / cap13)

            el_cost = pulp.lpSum(
                (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
                for n in cfg.network_nodes for t in range(T)
            )
            power_emissions = pulp.lpSum(
                th_ef[n] * G_TH[n][t] * dt * w[t]
                for n in cfg.network_nodes for t in range(T)
            )
            power_co2_cost = pulp.lpSum(
                p_co2[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
                for n in cfg.network_nodes for t in range(T)
            )
            trans_cost = 0
            if cfg.transmission_cost_eur_per_mwh > 0:
                trans_cost = pulp.lpSum(
                    cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
                    for t in range(T)
                )
            avoided_heat_price = prof.get("avoided_heat_cost", p_heat_eff)
            heat_emissions_factor = (
                cfg.independent_heat_boiler_emissions_tco2_per_mwh_fuel
                / max(1e-9, cfg.independent_heat_boiler_efficiency)
            )
            baseline_heat_resource_cost = pulp.lpSum(
                avoided_heat_price[t] * dh_cap_eff[t] * dt * w[t] for t in range(T)
            )
            baseline_heat_emissions = pulp.lpSum(
                heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
            )
            avoided_heat_emissions = pulp.lpSum(
                heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
            )
            emissions = power_emissions + baseline_heat_emissions - avoided_heat_emissions
            baseline_heat_co2_cost = pulp.lpSum(
                p_co2[t] * heat_emissions_factor * dh_cap_eff[t] * dt * w[t] for t in range(T)
            )
            avoided_heat_co2_cost = pulp.lpSum(
                p_co2[t] * heat_emissions_factor * H_SOLD[t] * dt * w[t] for t in range(T)
            )
            co2_cost = power_co2_cost + baseline_heat_co2_cost - avoided_heat_co2_cost
            avoided_flex_price = prof.get(
                "avoided_flex_cost",
                [cfg.independent_avoided_flex_cost_eur_per_mw_h] * T,
            )
            heat_revenue = pulp.lpSum(
                p_heat_eff[t] * cfg.dh_value_multipliers[k] * Q_DH[k][t] * dt * w[t]
                for k in range(K) for t in range(T)
            )
            flex_revenue = pulp.lpSum(
                p_flex[t] * RF[t] * dt * w[t] for t in range(T)
            )
            demand_charge = demand_charge_horizon(cfg) * P_PEAK
            if allow_it_shortfall_sc:
                sla_penalty = pulp.lpSum(cfg.sla_penalty_eur_per_mwh * SLA_SHORT_EFF[t] * dt * w[t] for t in range(T))
            else:
                sla_penalty = 0
            E_IT_expr = pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T))
            E_FAC_expr = pulp.lpSum(DC_LOAD[t] * dt * w[t] for t in range(T))
            eff_benefit = eff_reduction * (
                cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC_expr
                + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT_expr
            )

            capex = sc.capex_multiplier * capex_scale(cfg) * (
                cfg.capex_HX * K_HX_s
                + cfg.capex_HP * K_HP_s
                + cfg.capex_FLEX * K_FLEX_s
                + cfg.capex_BAT_E * K_BAT_E_s
                + cfg.capex_BAT_P * K_BAT_P_s
                + eff_capex_from_reduction(eff_reduction, cfg)
            )

            operating_pv_factor, land_pv_factor = _lifecycle_factors(cfg, sc)
            m += Cost == (
                operating_pv_factor * (
                    el_cost + baseline_heat_resource_cost + co2_cost + trans_cost + demand_charge + sla_penalty
                    - heat_revenue - flex_revenue - eff_benefit
                )
                + capex + land_pv_factor * land_cost_expr
                + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
            )
            carbon_cap = (cfg.carbon_emissions_cap_tco2_by_sid or {}).get(sc.sid)
            if cfg.enforce_carbon_cap and carbon_cap is not None:
                m += emissions <= float(carbon_cap), f"PhysicalCarbonCap_s{sc.sid}"
            discounted_it_mwh = max(
                1e-9,
                sum(it_base[t] * dt * w[t] for t in range(T)) * operating_pv_factor,
            )
            if cfg.robust_objective == "normalized_lexicographic":
                m += Z >= (Cost - C_star[sc.sid]) / discounted_it_mwh
            else:
                m += Z >= Cost - C_star[sc.sid]

        n_s = max(1, len(Cost_s_vars))
        avg_cost = pulp.lpSum(Cost_s_vars[sid] for sid in Cost_s_vars) * (1.0 / n_s)
        if cfg.robust_objective == "expected_value":
            m += avg_cost
            m.solve(make_solver_lp(cfg))
        else:
            m += Z
            m.solve(make_solver_lp(cfg))
            z_star = float(pulp.value(Z)) if pulp.LpStatus[m.status] == "Optimal" else float("nan")
            if cfg.robust_objective in {"normalized_lexicographic", "lexicographic_minmax"} and pulp.LpStatus[m.status] == "Optimal":
                tolerance = max(1e-6, abs(z_star) * cfg.lexicographic_regret_tolerance)
                m += Z <= z_star + tolerance, "LexicographicRegretLimit"
                m.setObjective(avg_cost)
                m.solve(make_solver_lp(cfg))

        realized_costs = {sid: float(pulp.value(value)) for sid, value in Cost_s_vars.items()}
        realized_regrets = {sid: realized_costs[sid] - C_star[sid] for sid in realized_costs}
        worst_regret = max(realized_regrets.values())
        scenario_by_sid = {sc.sid: sc for sc in scenarios}
        normalized_regrets = {}
        for sid, regret in realized_regrets.items():
            sc = scenario_by_sid[sid]
            prof = profiles[sid]
            scenario_weights = prof.get("w", [1.0] * len(prof["it_base"]))
            operating_factor, _ = _lifecycle_factors(cfg, sc)
            discounted_it = max(
                1e-9,
                sum(
                    prof["it_base"][t] * scenario_weights[t]
                    for t in range(len(prof["it_base"]))
                ) * operating_factor,
            )
            normalized_regrets[sid] = regret / discounted_it
        worst_normalized_regret = max(normalized_regrets.values())
        reference_it_mw = sum(sc.base_it_mw for sc in scenarios) / len(scenarios)

        return {
            "status": pulp.LpStatus[m.status],
            "K_HX": pulp.value(k_HX) * reference_it_mw,
            "K_HP": pulp.value(k_HP) * reference_it_mw,
            "K_FLEX": pulp.value(k_FLEX) * reference_it_mw,
            "K_BAT_E": pulp.value(k_BAT_E) * reference_it_mw,
            "K_BAT_P": pulp.value(k_BAT_P) * reference_it_mw,
            "K_HX_per_MW_IT": pulp.value(k_HX),
            "K_HP_per_MW_IT": pulp.value(k_HP),
            "K_FLEX_per_MW_IT": pulp.value(k_FLEX),
            "K_BAT_E_per_MW_IT": pulp.value(k_BAT_E),
            "K_BAT_P_per_MW_IT": pulp.value(k_BAT_P),
            "capacity_reference_it_mw": reference_it_mw,
            "design_normalization": "capacity_per_MW_IT",
            "eff_reduction": eff_reduction,
            "eff_capex": capex_scale(cfg) * eff_capex_from_reduction(eff_reduction, cfg),
            "Z": worst_normalized_regret if cfg.robust_objective == "normalized_lexicographic" else worst_regret,
            "worst_regret_eur": worst_regret,
            "worst_regret_eur_per_mwh_it": worst_normalized_regret,
            "minimum_worst_regret": worst_regret if cfg.robust_objective == "expected_value" else z_star,
            "scalarized_objective": pulp.value(m.objective),
            "avg_cost": sum(pulp.value(v) for v in Cost_s_vars.values()) / max(1, len(Cost_s_vars)),
            "robust_objective": cfg.robust_objective,
            "min_hx_it_mw_reference": reference_it_mw,
            "Cost_s": realized_costs,
            "Regret_s": realized_regrets,
            "NormalizedRegret_s": normalized_regrets,
        }

    best = None
    best_idx = 0
    if cfg.fixed_eff_reduction is not None:
        eff = float(cfg.fixed_eff_reduction)
        res = solve_fixed_eff(eff)
        best = res
        best_idx = min(range(len(cfg.eff_levels)), key=lambda i: abs(cfg.eff_levels[i] - eff))
    else:
        for i, eff in enumerate(cfg.eff_levels):
            res = solve_fixed_eff(eff)
            if best is None or (res["Z"], res["avg_cost"]) < (best["Z"], best["avg_cost"]):
                best = res
                best_idx = i

    assert best is not None
    best["eff_level_index"] = best_idx
    if cfg.fixed_eff_reduction is not None:
        best["eff_reduction"] = cfg.fixed_eff_reduction
        best["eff_capex"] = eff_capex_from_reduction(cfg.fixed_eff_reduction, cfg)
    return best
