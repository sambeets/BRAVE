"""Shared operational LP core.

The three solvers in :mod:`brave.optimization` (fixed-design operations,
perfect-foresight design, and min-max regret design) all build the same
per-period dispatch model: IT service and cooling, heat recovery and district
heat sales, battery and load flexibility reserves, a three-node DC-OPF network,
thermal generation, and the associated cost/emission ledgers.

Only the *outer* structure differs between them:

* fixed-design operations passes numeric capacities and post-processes duals;
* perfect foresight makes the capacities free decision variables;
* min-max regret shares a per-MW-IT design across scenarios.

``add_operational_block`` builds the common core against whatever capacity
handles the caller supplies (Python floats, ``LpVariable`` objects, or
``LpAffineExpression`` objects), so the reserve/heat/battery physics live in
exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pulp

from brave.config import Config
from brave.scenario import Scenario
from brave.metrics import demand_charge_horizon
from brave.timeseries import make_overhead_alpha_base


def _hp_source_to_delivered_ratio(cop: float) -> float:
    """Convert recovered source heat to delivered heat for a heat pump."""
    return max(1.000001, cop) / max(1e-9, max(1.000001, cop) - 1.0)


def _hp_source_to_power_ratio(cop: float) -> float:
    """Convert recovered source heat to heat-pump electric input."""
    return 1.0 / max(1e-9, max(1.000001, cop) - 1.0)


def _network_line_parameters(sc: Scenario, cfg: Config):
    multiplier = max(1e-6, float(getattr(sc, "congestion_multiplier", 1.0)))
    return (
        cfg.line_cap_12 * multiplier,
        cfg.line_cap_23 * multiplier,
        cfg.line_cap_13 * multiplier,
        cfg.line_susceptance_12_mw_per_rad,
        cfg.line_susceptance_23_mw_per_rad,
        cfg.line_susceptance_13_mw_per_rad,
    )


def _existing_cooling_capacity_mw(sc: Scenario, alpha_base: List[float]) -> float:
    return max(0.0, sc.base_it_mw * max(alpha_base, default=max(0.0, sc.base_pue - 1.0)))


@dataclass
class OperationalBlock:
    """Variable handles and cost expressions produced by the operational core."""

    T: int
    dt: float
    w: List[float]
    alpha_eff: List[float]

    IT_SERV: Dict
    COOL: Dict
    H_REC: Dict
    H_REC_IT: Dict
    H_REC_COOL: Dict
    H_FLOW: Dict
    H_IN: Dict
    H_SOLD: Dict
    P_HP: Dict
    Q_DH: Dict
    RF_LOAD: Dict
    RF_BAT: Dict
    RF: Dict
    DC_LOAD: Dict
    P_PEAK: object
    CH: Dict
    DIS: Dict
    SOC: Dict
    SLA_SHORT: Dict
    SLA_SHORT_EFF: Dict
    CH_TH: Dict
    DIS_TH: Dict
    S_TH: Dict
    G_REN: Dict
    G_TH: Dict
    F12: Dict
    F23: Dict
    F13: Dict
    THETA: Dict
    U_GRID: Dict
    AF12: Dict
    AF23: Dict
    AF13: Dict
    RES_TH: Dict
    RES_SHORT: Dict

    el_cost: object
    power_emissions: object
    power_co2_cost: object
    trans_cost: object
    baseline_heat_resource_cost: object
    baseline_heat_emissions: object
    avoided_heat_emissions: object
    emissions: object
    baseline_heat_co2_cost: object
    avoided_heat_co2_cost: object
    co2_cost: object
    heat_revenue: object
    flex_revenue: object
    demand_charge: object
    eff_benefit: object
    reserve_cost: object
    reserve_shortfall_penalty_cost: object
    E_IT_expr: object
    E_FAC_expr: object
    operating_cost: object
    u_grid_sum: object

    balance_constraints: Dict
    line_limit_constraints: Dict

    K_HX: object
    K_HP: object
    K_FLEX: object
    K_BAT_E: object
    K_BAT_P: object
    heat_active: bool
    flex_active: bool
    battery_active: bool
    eff_reduction: float


def add_operational_block(
    m: pulp.LpProblem,
    sc: Scenario,
    prof: Dict[str, List[float]],
    cfg: Config,
    *,
    K_HX,
    K_HP,
    K_FLEX,
    K_BAT_E,
    K_BAT_P,
    eff_reduction: float,
    heat_active: bool,
    flex_active: bool,
    battery_active: bool,
    suffix: str = "",
    rep_day_cycle: bool = True,
    work_conservation_weighted: bool = False,
    track_constraints: bool = False,
) -> OperationalBlock:
    """Add the shared dispatch variables, constraints and cost expressions.

    ``K_*`` capacity handles may be floats, ``LpVariable`` or ``LpAffineExpression``.
    ``heat_active``/``flex_active``/``battery_active`` decide whether the active
    physics or the disabling ``== 0`` constraints are written (the fixed-design
    solver gates these on positive numeric capacities; the design solvers gate
    on the profile ``allow_*`` flags because capacities are decision variables).
    """
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

    allow_flex = bool(prof.get("allow_flex", True))

    alpha_base = prof.get("alpha_base", None)
    if alpha_base is None:
        alpha_base = make_overhead_alpha_base(sc, T, cfg)
    alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

    p_heat_eff = p_heat
    dh_cap_eff = dh_cap

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

    def nm(base: str) -> str:
        return f"{base}{suffix}"

    IT_SERV = pulp.LpVariable.dicts(nm("IT_SERV"), range(T), lowBound=0)
    COOL = pulp.LpVariable.dicts(nm("COOL"), range(T), lowBound=0)
    H_REC = pulp.LpVariable.dicts(nm("H_REC"), range(T), lowBound=0)
    H_REC_IT = pulp.LpVariable.dicts(nm("H_REC_IT"), range(T), lowBound=0)
    H_REC_COOL = pulp.LpVariable.dicts(nm("H_REC_COOL"), range(T), lowBound=0)
    H_FLOW = pulp.LpVariable.dicts(nm("H_FLOW"), range(T), lowBound=0)
    H_IN = pulp.LpVariable.dicts(nm("H_IN"), range(T), lowBound=0)
    H_SOLD = pulp.LpVariable.dicts(nm("H_SOLD"), range(T), lowBound=0)
    P_HP = pulp.LpVariable.dicts(nm("P_HP"), range(T), lowBound=0)
    Q_DH = {k: pulp.LpVariable.dicts(nm(f"Q_DH_{k+1}"), range(T), lowBound=0) for k in range(K)}

    RF_LOAD = pulp.LpVariable.dicts(nm("RF_LOAD"), range(T), lowBound=0)
    RF_BAT = pulp.LpVariable.dicts(nm("RF_BAT"), range(T), lowBound=0)
    RF = pulp.LpVariable.dicts(nm("RF"), range(T), lowBound=0)
    DC_LOAD = pulp.LpVariable.dicts(nm("DC_LOAD"), range(T), lowBound=0)
    P_PEAK = pulp.LpVariable(nm("P_PEAK"), lowBound=0)

    CH = pulp.LpVariable.dicts(nm("CH"), range(T), lowBound=0)
    DIS = pulp.LpVariable.dicts(nm("DIS"), range(T), lowBound=0)
    SOC = pulp.LpVariable.dicts(nm("SOC"), range(T + 1), lowBound=0)
    SLA_SHORT = pulp.LpVariable.dicts(nm("SLA_SHORT"), range(T), lowBound=0)
    SLA_SHORT_EFF = pulp.LpVariable.dicts(nm("SLA_SHORT_EFF"), range(T), lowBound=0)

    if cfg.enable_thermal_storage:
        CH_TH = pulp.LpVariable.dicts(nm("CH_TH"), range(T), lowBound=0)
        DIS_TH = pulp.LpVariable.dicts(nm("DIS_TH"), range(T), lowBound=0)
        S_TH = pulp.LpVariable.dicts(nm("S_TH"), range(T + 1), lowBound=0)
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

    G_REN = {n: pulp.LpVariable.dicts(nm(f"G_REN_{n}"), range(T), lowBound=0) for n in cfg.network_nodes}
    G_TH = {n: pulp.LpVariable.dicts(nm(f"G_TH_{n}"), range(T), lowBound=0) for n in cfg.network_nodes}
    F12 = pulp.LpVariable.dicts(nm("F12"), range(T), lowBound=None)
    F23 = pulp.LpVariable.dicts(nm("F23"), range(T), lowBound=None)
    F13 = pulp.LpVariable.dicts(nm("F13"), range(T), lowBound=None)
    THETA = {n: pulp.LpVariable.dicts(nm(f"THETA_{n}"), range(T), lowBound=None) for n in cfg.network_nodes}
    U_GRID = pulp.LpVariable.dicts(nm("U_GRID"), range(T), lowBound=0, upBound=1)

    if cfg.transmission_cost_eur_per_mwh > 0:
        AF12 = pulp.LpVariable.dicts(nm("AF12"), range(T), lowBound=0)
        AF23 = pulp.LpVariable.dicts(nm("AF23"), range(T), lowBound=0)
        AF13 = pulp.LpVariable.dicts(nm("AF13"), range(T), lowBound=0)
    else:
        AF12 = {t: 0 for t in range(T)}
        AF23 = {t: 0 for t in range(T)}
        AF13 = {t: 0 for t in range(T)}

    th_cap = {"N1": cfg.th_cap_N1_mw, "N2": cfg.th_cap_N2_mw, "N3": cfg.th_cap_N3_mw}
    if cfg.enable_operating_reserve:
        RES_TH = {n: pulp.LpVariable.dicts(nm(f"RES_TH_{n}"), range(T), lowBound=0) for n in cfg.network_nodes}
        RES_SHORT = pulp.LpVariable.dicts(nm("RES_SHORT"), range(T), lowBound=0)
    else:
        RES_TH = {n: {t: 0 for t in range(T)} for n in cfg.network_nodes}
        RES_SHORT = {t: 0 for t in range(T)}

    balance_constraints = {n: [] for n in cfg.network_nodes}
    line_limit_constraints = {line: [] for line in ("12+", "12-", "23+", "23-", "13+", "13-")}

    if not cfg.fix_it_profile:
        if work_conservation_weighted:
            m += pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T)) == sum(
                it_base[t] * dt * w[t] for t in range(T)
            ), nm("WorkConservation")
        else:
            m += pulp.lpSum(IT_SERV[t] * dt for t in range(T)) == sum(it_base) * dt, nm("WorkConservation")

    m += SOC[0] == 0.5 * K_BAT_E, nm("SOC_start")
    m += SOC[T] == SOC[0], nm("SOC_cycle")

    if cfg.enable_thermal_storage:
        m += S_TH[0] == 0.0, nm("S_TH_start")
        m += S_TH[T] == 0.0, nm("S_TH_end")

    if rep_day_cycle:
        for start in rep_day_starts:
            end = start + 24
            if 0 <= start < T and end <= T:
                m += SOC[end] == SOC[start]
                if cfg.enable_thermal_storage:
                    m += S_TH[end] == S_TH[start]

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
            if track_constraints:
                line_limit_constraints[key].append(constraint)

        if force_it_flex or not cfg.fix_it_profile:
            lb = (1 - flex_frac_eff) * it_base[t]
            ub = (1 + flex_frac_eff) * it_base[t]
            m += IT_SERV[t] >= lb
            m += IT_SERV[t] <= ub
        else:
            m += IT_SERV[t] == it_base[t]

        m += COOL[t] == alpha_eff[t] * IT_SERV[t]

        if heat_active:
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

        if battery_active:
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

        if flex_active:
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

        con_n1 = (G_REN[n1][t] + G_TH[n1][t]) - F12[t] - F13[t] == node_loads[n1][t] + dc_at_n1
        con_n2 = (G_REN[n2][t] + G_TH[n2][t]) + F12[t] - F23[t] == node_loads[n2][t] + dc_at_n2
        con_n3 = (G_REN[n3][t] + G_TH[n3][t]) + F13[t] + F23[t] == node_loads[n3][t] + dc_at_n3
        m += con_n1
        m += con_n2
        m += con_n3
        if track_constraints:
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

        if cfg.enable_operating_reserve:
            reserve_requirement_t = cfg.reserve_requirement_frac * DC_LOAD[t]
            for n in cfg.network_nodes:
                m += RES_TH[n][t] <= th_cap[n] - G_TH[n][t]
            m += (
                pulp.lpSum(RES_TH[n][t] for n in cfg.network_nodes) + RF[t] + RES_SHORT[t]
                >= reserve_requirement_t
            )

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
    heat_revenue = pulp.lpSum(
        p_heat_eff[t] * cfg.dh_value_multipliers[k] * Q_DH[k][t] * dt * w[t]
        for k in range(K) for t in range(T)
    )
    flex_revenue = pulp.lpSum(
        p_flex[t] * RF[t] * dt * w[t] for t in range(T)
    )
    demand_charge = demand_charge_horizon(cfg) * P_PEAK

    if cfg.enable_operating_reserve:
        reserve_cost = pulp.lpSum(
            cfg.thermal_reserve_cost_eur_per_mw_h
            * pulp.lpSum(RES_TH[n][t] for n in cfg.network_nodes)
            * dt * w[t]
            for t in range(T)
        )
        reserve_shortfall_penalty_cost = pulp.lpSum(
            cfg.reserve_shortfall_penalty_eur_per_mw_h * RES_SHORT[t] * dt * w[t]
            for t in range(T)
        )
    else:
        reserve_cost = 0
        reserve_shortfall_penalty_cost = 0

    sla_penalty = 0
    E_IT_expr = pulp.lpSum(IT_SERV[t] * dt * w[t] for t in range(T))
    E_FAC_expr = pulp.lpSum(DC_LOAD[t] * dt * w[t] for t in range(T))
    eff_benefit = eff_reduction * (
        cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC_expr
        + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT_expr
    )

    operating_cost = (
        el_cost + baseline_heat_resource_cost + co2_cost + trans_cost + demand_charge
        + reserve_cost + reserve_shortfall_penalty_cost
        + sla_penalty - heat_revenue - flex_revenue - eff_benefit
    )
    u_grid_sum = pulp.lpSum(U_GRID[t] for t in range(T))

    return OperationalBlock(
        T=T, dt=dt, w=w, alpha_eff=alpha_eff,
        IT_SERV=IT_SERV, COOL=COOL, H_REC=H_REC, H_REC_IT=H_REC_IT,
        H_REC_COOL=H_REC_COOL, H_FLOW=H_FLOW, H_IN=H_IN, H_SOLD=H_SOLD, P_HP=P_HP,
        Q_DH=Q_DH, RF_LOAD=RF_LOAD, RF_BAT=RF_BAT, RF=RF, DC_LOAD=DC_LOAD, P_PEAK=P_PEAK,
        CH=CH, DIS=DIS, SOC=SOC, SLA_SHORT=SLA_SHORT, SLA_SHORT_EFF=SLA_SHORT_EFF,
        CH_TH=CH_TH, DIS_TH=DIS_TH, S_TH=S_TH, G_REN=G_REN, G_TH=G_TH,
        F12=F12, F23=F23, F13=F13, THETA=THETA, U_GRID=U_GRID,
        AF12=AF12, AF23=AF23, AF13=AF13, RES_TH=RES_TH, RES_SHORT=RES_SHORT,
        el_cost=el_cost, power_emissions=power_emissions, power_co2_cost=power_co2_cost,
        trans_cost=trans_cost, baseline_heat_resource_cost=baseline_heat_resource_cost,
        baseline_heat_emissions=baseline_heat_emissions,
        avoided_heat_emissions=avoided_heat_emissions, emissions=emissions,
        baseline_heat_co2_cost=baseline_heat_co2_cost,
        avoided_heat_co2_cost=avoided_heat_co2_cost, co2_cost=co2_cost,
        heat_revenue=heat_revenue, flex_revenue=flex_revenue, demand_charge=demand_charge,
        eff_benefit=eff_benefit, reserve_cost=reserve_cost,
        reserve_shortfall_penalty_cost=reserve_shortfall_penalty_cost,
        E_IT_expr=E_IT_expr, E_FAC_expr=E_FAC_expr,
        operating_cost=operating_cost, u_grid_sum=u_grid_sum,
        balance_constraints=balance_constraints, line_limit_constraints=line_limit_constraints,
        K_HX=K_HX, K_HP=K_HP, K_FLEX=K_FLEX, K_BAT_E=K_BAT_E, K_BAT_P=K_BAT_P,
        heat_active=heat_active, flex_active=flex_active, battery_active=battery_active,
        eff_reduction=eff_reduction,
    )
