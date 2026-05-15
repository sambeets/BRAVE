from __future__ import annotations

from typing import Dict, List, Tuple

import pulp

from brave.config import Config
from brave.scenario import Scenario
from brave.metrics import (
    compute_sipue,
    compute_land_m2,
    service_revenue_scale,
    land_cost_horizon,
    demand_charge_horizon,
    design_size_proxy,
    eff_capex_from_reduction,
    capex_scale,
)
from brave.solver import make_solver_lp
from brave.timeseries import make_overhead_alpha_base


def baseline_design_for_scenario(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Tuple[float, float, float, float, float]:
    """
    Baseline ("incumbent") design, scaled by scenario IT size and DC type.
    Type-specific scaling reflects realistic industry practice:
      - hyperscale: large campuses, invested in heat recovery & flexibility
      - streaming:  moderate service infrastructure
      - ai:         compute-focused, minimal heat recovery investment
      - colocation: tenant-driven, minimal service infrastructure
    Returns: (K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction)
    """
    # Type-specific multipliers on baseline allocations
    type_scales = {
        "hyperscale": {"hx": 1.00, "flex": 1.00, "bat_e": 1.00, "bat_p": 1.00},
        "streaming":  {"hx": 0.70, "flex": 0.85, "bat_e": 0.80, "bat_p": 0.80},
        "ai":         {"hx": 0.55, "flex": 1.50, "bat_e": 0.70, "bat_p": 0.70},
        "colocation": {"hx": 0.40, "flex": 0.55, "bat_e": 0.60, "bat_p": 0.60},
    }
    ts = type_scales.get(sc.dc_type, {"hx": 1.0, "flex": 1.0, "bat_e": 1.0, "bat_p": 1.0})

    k_hx = max(0.0, cfg.baseline_hx_mwth_per_it_mw * ts["hx"]) * sc.base_it_mw
    k_flex = max(0.0, cfg.baseline_flex_mw_per_it_mw * ts["flex"]) * sc.base_it_mw
    k_bat_e = max(0.0, cfg.baseline_bat_e_hours * ts["bat_e"]) * sc.base_it_mw
    k_bat_p = max(0.0, cfg.baseline_bat_p_mw_per_it_mw * ts["bat_p"]) * sc.base_it_mw
    eff_red = max(0.0, min(0.90, float(cfg.baseline_eff_reduction)))

    if not bool(prof.get("allow_heat", True)):
        k_hx = 0.0
    if not bool(prof.get("allow_flex", True)):
        k_flex = 0.0
    if not bool(prof.get("allow_battery", True)):
        k_bat_e = 0.0
        k_bat_p = 0.0

    k_hx = min(cfg.K_HX_max, k_hx)
    k_flex = min(cfg.K_FLEX_max, k_flex)
    k_bat_e = min(cfg.K_BAT_E_max, k_bat_e)
    k_bat_p = min(cfg.K_BAT_P_max, k_bat_p)

    # Respect per-scenario land limits by scaling down proportionally
    land_limit = sc.land_limit_m2 if sc.land_limit_m2 is not None else cfg.land_limit_m2
    if land_limit is not None:
        land_used = compute_land_m2(k_hx, k_flex, k_bat_e, k_bat_p, eff_red, cfg)
        if land_used > land_limit:
            # Scale all service capacities down proportionally to fit
            excess = land_used - land_limit
            service_land = land_used - cfg.base_land_m2
            if service_land > 0:
                scale = max(0.0, (land_limit - cfg.base_land_m2) / service_land)
                k_hx *= scale
                k_flex *= scale
                k_bat_e *= scale
                k_bat_p *= scale

    return k_hx, k_flex, k_bat_e, k_bat_p, eff_red


def solve_ops_given_design(
    sc: Scenario,
    prof: Dict[str, List[float]],
    cfg: Config,
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    eff_reduction: float,  # NEW: chosen efficiency level in [0,1)
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
    dh_cap = prof["dh_cap"]
    p_flex = prof["p_flex"]
    p_it = prof.get("p_it", [0.0] * T)
    cf = prof["carbon_factor"]
    p_co2 = prof["p_co2"]
    cop_t = prof.get("cop_t", [sc.hp_cop] * T)
    node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
    dc_node = prof.get("dc_node", cfg.network_nodes[0])

    allow_heat = bool(prof.get("allow_heat", True))
    allow_flex = bool(prof.get("allow_flex", True))
    allow_battery = bool(prof.get("allow_battery", True))
    allow_it_shortfall = (
        cfg.objective_mode == "profit_max"
        and cfg.allow_it_shortfall
        and cfg.sla_penalty_eur_per_mwh > 0
    )

    alpha_base = prof.get("alpha_base", None)
    if alpha_base is None:
        alpha_base = make_overhead_alpha_base(sc, T, cfg)

    eff_reduction = max(0.0, min(0.90, eff_reduction))
    alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

    p_heat_eff = p_heat
    dh_cap_eff = dh_cap

    m = pulp.LpProblem(f"Ops_{sc.sid}", pulp.LpMinimize)

    IT_SERV = pulp.LpVariable.dicts("IT_SERV", range(T), lowBound=0)
    COOL = pulp.LpVariable.dicts("COOL", range(T), lowBound=0)
    H_REC = pulp.LpVariable.dicts("H_REC", range(T), lowBound=0)
    H_FLOW = pulp.LpVariable.dicts("H_FLOW", range(T), lowBound=0)
    H_IN = pulp.LpVariable.dicts("H_IN", range(T), lowBound=0)
    H_SOLD = pulp.LpVariable.dicts("H_SOLD", range(T), lowBound=0)

    K = int(cfg.dh_segments)
    if K <= 0:
        raise ValueError("dh_segments must be >= 1")
    if len(cfg.dh_seg_fracs) != K or len(cfg.dh_value_multipliers) != K:
        raise ValueError("dh_seg_fracs and dh_value_multipliers must have length dh_segments")
    if abs(sum(cfg.dh_seg_fracs) - 1.0) > 1e-6:
        raise ValueError("dh_seg_fracs must sum to 1.0")
    Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}", range(T), lowBound=0) for k in range(K)}

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
    F12 = pulp.LpVariable.dicts("F12", range(T), lowBound=-cfg.line_cap_12, upBound=cfg.line_cap_12)
    F23 = pulp.LpVariable.dicts("F23", range(T), lowBound=-cfg.line_cap_23, upBound=cfg.line_cap_23)
    F13 = pulp.LpVariable.dicts("F13", range(T), lowBound=-cfg.line_cap_13, upBound=cfg.line_cap_13)
    balance_constraints = {n: [] for n in cfg.network_nodes}

    if cfg.transmission_cost_eur_per_mwh > 0:
        AF12 = pulp.LpVariable.dicts("AF12", range(T), lowBound=0)
        AF23 = pulp.LpVariable.dicts("AF23", range(T), lowBound=0)
        AF13 = pulp.LpVariable.dicts("AF13", range(T), lowBound=0)
    else:
        AF12 = {t: 0 for t in range(T)}
        AF23 = {t: 0 for t in range(T)}
        AF13 = {t: 0 for t in range(T)}

    U_GRID = pulp.LpVariable.dicts("U_GRID", range(T), lowBound=0, upBound=1)

    if cfg.objective_mode == "cost_min":
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
        elif cfg.objective_mode == "cost_min":
            lb = (1 - flex_frac_eff) * it_base[t]
            ub = (1 + flex_frac_eff) * it_base[t]
            m += IT_SERV[t] >= lb
            m += IT_SERV[t] <= ub
            it_ub = ub
        else:
            it_cap = sc.base_it_mw * cfg.it_capacity_multiplier
            ramp_share = max(0.0, min(1.0, cfg.ramp_base_share - cfg.ramp_tau_per_mw * design_size_proxy(sc, K_HX, K_FLEX, K_BAT_E, K_BAT_P, cfg)))
            m += IT_SERV[t] <= it_demand[t] * ramp_share
            m += IT_SERV[t] <= it_cap
            it_ub = it_demand[t] * ramp_share

        m += COOL[t] == alpha_eff[t] * IT_SERV[t]

        if allow_heat and K_HX > 0:
            pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
            m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
            m += H_REC[t] <= K_HX
            m += H_FLOW[t] <= H_REC[t]
            m += H_FLOW[t] <= pipe_cap
            m += H_IN[t] == (1.0 - cfg.heat_pipe_loss_frac) * H_FLOW[t]

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
            m += H_FLOW[t] == 0
            m += H_IN[t] == 0
            m += H_SOLD[t] == 0
            for k in range(K):
                m += Q_DH[k][t] == 0
            if cfg.enable_thermal_storage:
                m += CH_TH[t] == 0
                m += DIS_TH[t] == 0
                m += S_TH[t + 1] == S_TH[t]

        min_it = (1 - flex_frac_eff) * it_base[t] if (cfg.fix_it_profile or cfg.objective_mode == "cost_min") else 0.0
        if allow_flex and K_FLEX > 0:
            m += RF[t] <= K_FLEX
            m += RF[t] <= IT_SERV[t] - min_it
        else:
            m += RF[t] == 0

        m += SOC[t] <= K_BAT_E
        m += SOC[t + 1] <= K_BAT_E

        if allow_battery and (K_BAT_E > 0 or K_BAT_P > 0):
            m += CH[t] <= K_BAT_P
            m += DIS[t] <= K_BAT_P
            m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1 / cfg.eta_dis) * DIS[t] * dt
        else:
            m += CH[t] == 0
            m += DIS[t] == 0
            m += SOC[t + 1] == SOC[t]

        P_HP = H_FLOW[t] * (1.0 / cop_t[t])
        P_SERV = (
            cfg.service_parasitic_heat_frac * H_SOLD[t]
            + cfg.service_parasitic_flex_frac * RF[t]
            + cfg.service_parasitic_hx_mw_per_mwth * K_HX
            + cfg.service_parasitic_flex_mw_per_mw * K_FLEX
        )

        m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP + P_SERV + CH[t] - DIS[t]
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

        m += U_GRID[t] >= F12[t] / cfg.line_cap_12
        m += U_GRID[t] >= -F12[t] / cfg.line_cap_12
        m += U_GRID[t] >= F23[t] / cfg.line_cap_23
        m += U_GRID[t] >= -F23[t] / cfg.line_cap_23
        m += U_GRID[t] >= F13[t] / cfg.line_cap_13
        m += U_GRID[t] >= -F13[t] / cfg.line_cap_13

    for start in rep_day_starts:
        end = start + 24
        if 0 <= start < T and end <= T:
            m += SOC[end] == SOC[start]
            if cfg.enable_thermal_storage:
                m += S_TH[end] == S_TH[start]

    srv_scale = service_revenue_scale(cfg, sc)
    fuel_price = prof.get("fuel_price", [0.0] * T)
    co2_price = prof.get("p_co2", [0.0] * T)
    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    el_cost_expr = srv_scale * pulp.lpSum(
        (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    co2_cost_expr = srv_scale * pulp.lpSum(
        co2_price[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    trans_cost_expr = 0
    if cfg.transmission_cost_eur_per_mwh > 0:
        trans_cost_expr = srv_scale * pulp.lpSum(
            cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
            for t in range(T)
        )

    heat_rev_expr = srv_scale * pulp.lpSum(
        (cfg.dh_value_multipliers[k] * p_heat_eff[t]) * Q_DH[k][t] * dt * w[t]
        for k in range(K) for t in range(T)
    )
    flex_rev_expr = srv_scale * pulp.lpSum([p_flex[t] * RF[t] * dt * w[t] for t in range(T)])
    it_rev_expr = srv_scale * pulp.lpSum([p_it[t] * IT_SERV[t] * dt * w[t] for t in range(T)])
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
    capex = scale_cap * (cfg.capex_HX * K_HX +
                         cfg.capex_FLEX * K_FLEX +
                         cfg.capex_BAT_E * K_BAT_E +
                         cfg.capex_BAT_P * K_BAT_P)
    eff_capex = scale_cap * eff_capex_from_reduction(eff_reduction, cfg)
    land_m2 = compute_land_m2(K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg)
    land_cost = land_cost_horizon(land_m2, cfg)
    if cfg.objective_mode == "profit_max":
        profit_expr = (
            it_rev_expr + heat_rev_expr + flex_rev_expr
            - el_cost_expr - co2_cost_expr - trans_cost_expr - demand_charge_expr - sla_penalty_expr
            - capex - eff_capex - land_cost
            + eff_benefit_expr
        )
        m += -profit_expr + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
    else:
        m += (
            el_cost_expr + co2_cost_expr + trans_cost_expr + demand_charge_expr
            - heat_rev_expr - flex_rev_expr
            + capex + eff_capex + land_cost
            - eff_benefit_expr
            + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
        )

    m.solve(make_solver_lp(cfg))

    status = pulp.LpStatus[m.status]
    if cfg.validate_solutions and status != "Optimal":
        raise RuntimeError(f"Ops model did not solve to Optimal (status={status}).")

    ITv = [pulp.value(IT_SERV[t]) for t in range(T)]
    COOLv = [pulp.value(COOL[t]) for t in range(T)]
    DC_LOADv = [pulp.value(DC_LOAD[t]) for t in range(T)]
    G_RENv = {n: [pulp.value(G_REN[n][t]) for t in range(T)] for n in cfg.network_nodes}
    G_THv = {n: [pulp.value(G_TH[n][t]) for t in range(T)] for n in cfg.network_nodes}
    F12v = [pulp.value(F12[t]) for t in range(T)]
    F23v = [pulp.value(F23[t]) for t in range(T)]
    F13v = [pulp.value(F13[t]) for t in range(T)]

    H_RECv = [pulp.value(H_REC[t]) for t in range(T)]
    H_FLOWv = [pulp.value(H_FLOW[t]) for t in range(T)]
    H_INv = [pulp.value(H_IN[t]) for t in range(T)]
    H_SOLDv = [pulp.value(H_SOLD[t]) for t in range(T)]
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
    RFv = [pulp.value(RF[t]) for t in range(T)]
    CHv = [pulp.value(CH[t]) for t in range(T)]
    DISv = [pulp.value(DIS[t]) for t in range(T)]
    peak_import_mw = float(pulp.value(P_PEAK))
    # --- Settlement OPF: cost-minimization with fixed DC load to get true LMPs ---
    n1, n2, n3 = cfg.network_nodes
    ren_avail_settle = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
    ms = pulp.LpProblem("settlement_opf", pulp.LpMinimize)
    sG_TH = {n: [pulp.LpVariable(f"sGTH_{n}_{t}", 0) for t in range(T)] for n in cfg.network_nodes}
    sG_REN = {n: [pulp.LpVariable(f"sGREN_{n}_{t}", 0) for t in range(T)] for n in cfg.network_nodes}
    sF12 = [pulp.LpVariable(f"sF12_{t}") for t in range(T)]
    sF23 = [pulp.LpVariable(f"sF23_{t}") for t in range(T)]
    sF13 = [pulp.LpVariable(f"sF13_{t}") for t in range(T)]
    settle_bal = {n: [] for n in cfg.network_nodes}
    for t in range(T):
        # Renewable and thermal capacity limits
        ms += sG_REN[n1][t] <= ren_avail_settle[n1][t]
        ms += sG_REN[n2][t] <= ren_avail_settle[n2][t]
        ms += sG_REN[n3][t] <= ren_avail_settle[n3][t]
        ms += sG_TH[n1][t] <= cfg.th_cap_N1_mw
        ms += sG_TH[n2][t] <= cfg.th_cap_N2_mw
        ms += sG_TH[n3][t] <= cfg.th_cap_N3_mw
        # Line limits
        ms += sF12[t] <= cfg.line_cap_12
        ms += sF12[t] >= -cfg.line_cap_12
        ms += sF23[t] <= cfg.line_cap_23
        ms += sF23[t] >= -cfg.line_cap_23
        ms += sF13[t] <= cfg.line_cap_13
        ms += sF13[t] >= -cfg.line_cap_13
        # Nodal balance with fixed DC load
        dc_at = {n1: DC_LOADv[t] if dc_node == n1 else 0.0,
                 n2: DC_LOADv[t] if dc_node == n2 else 0.0,
                 n3: DC_LOADv[t] if dc_node == n3 else 0.0}
        c1 = (sG_REN[n1][t] + sG_TH[n1][t]) - sF12[t] - sF13[t] == node_loads[n1][t] + dc_at[n1]
        c2 = (sG_REN[n2][t] + sG_TH[n2][t]) + sF12[t] - sF23[t] == node_loads[n2][t] + dc_at[n2]
        c3 = (sG_REN[n3][t] + sG_TH[n3][t]) + sF13[t] + sF23[t] == node_loads[n3][t] + dc_at[n3]
        ms += c1; ms += c2; ms += c3
        settle_bal[n1].append(c1)
        settle_bal[n2].append(c2)
        settle_bal[n3].append(c3)
    # Objective: minimize total generation cost (fuel + VOM + CO2)
    ms += pulp.lpSum(
        ((fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) + co2_price[t] * th_ef[n])
        * sG_TH[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    ms.solve(make_solver_lp(cfg))
    settle_status = pulp.LpStatus[ms.status]
    if settle_status != "Optimal":
        print(f"  [WARN] Settlement OPF status: {settle_status}, falling back to profit-max duals")
        lmp_node_series = {n: [] for n in cfg.network_nodes}
        for n in cfg.network_nodes:
            for con in balance_constraints[n]:
                dual_val = getattr(con, "pi", None)
                if dual_val is None:
                    dual_val = float("nan")
                lmp_node_series[n].append(dual_val)
    else:
        lmp_node_series = {n: [] for n in cfg.network_nodes}
        for n in cfg.network_nodes:
            for t_idx, con in enumerate(settle_bal[n]):
                dual_val = getattr(con, "pi", None)
                if dual_val is None:
                    if cfg.require_duals:
                        raise RuntimeError("Settlement OPF: solver did not provide dual values.")
                    dual_val = float("nan")
                else:
                    # Duals are scaled by dt*w[t] from the objective; unscale to get EUR/MWh
                    dual_val = dual_val / max(1e-12, dt * w[t_idx])
                lmp_node_series[n].append(dual_val)
    settle_lmp_series = lmp_node_series.get(dc_node, [float("nan")] * T)
    # DEBUG: print settlement LMP stats
    _valid_lmps = [x for x in settle_lmp_series if x == x and abs(x) < 1e12]
    if _valid_lmps:
        print(f"  [DBG] Settlement LMP at {dc_node}: min={min(_valid_lmps):.1f}, avg={sum(_valid_lmps)/len(_valid_lmps):.1f}, max={max(_valid_lmps):.1f} EUR/MWh (n={len(_valid_lmps)})")

    util_12 = [abs(F12v[t]) / max(1e-9, cfg.line_cap_12) for t in range(T)]
    util_23 = [abs(F23v[t]) / max(1e-9, cfg.line_cap_23) for t in range(T)]
    util_13 = [abs(F13v[t]) / max(1e-9, cfg.line_cap_13) for t in range(T)]
    util_max = [max(util_12[t], util_23[t], util_13[t]) for t in range(T)]
    w_sum = sum(w) if w else float(T)

    E_IT = sum(ITv[t] * w[t] for t in range(T)) * dt
    E_GRID = sum(DC_LOADv[t] * w[t] for t in range(T)) * dt
    E_LOAD = sum((DC_LOADv[t] + DISv[t] - CHv[t]) * w[t] * dt for t in range(T))
    E_FAC = E_GRID
    lmp_num = 0.0
    lmp_den = 0.0
    for load, price, wt in zip(DC_LOADv, settle_lmp_series, w):
        lmp_num += load * price * wt
        lmp_den += load * wt
    if lmp_den <= 1e-9:
        lmp_settle_avg_dc = sum(settle_lmp_series) / max(1, len(settle_lmp_series))
    else:
        lmp_settle_avg_dc = lmp_num / lmp_den
    lmp_shadow_avg_dc = lmp_settle_avg_dc
    PUE = E_FAC / max(1e-9, E_IT)

    el_settlement_eur = 0.0
    for load, price, wt in zip(DC_LOADv, settle_lmp_series, w):
        el_settlement_eur += load * price * dt * wt

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
            ("H_FLOW", H_FLOWv),
            ("H_IN", H_INv),
            ("H_SOLD", H_SOLDv),
        ]:
            if any(v is None or (isinstance(v, float) and (v != v)) for v in seq):
                _bad(f"{seq_name} contains None/NaN values.")

        if E_IT <= tol:
            _bad(f"E_IT is too small ({E_IT}); PUE/siPUE may be ill-defined.")
        if not (PUE == PUE and abs(PUE) < 1e12):
            _bad(f"PUE is not finite: {PUE}")
        if not (el_settlement_eur == el_settlement_eur and abs(el_settlement_eur) < 1e18):
            _bad(f"Electricity settlement is not finite: {el_settlement_eur}")
        if cfg.enable_demand_charge:
            if peak_import_mw < -tol:
                _bad(f"Peak import is negative: {peak_import_mw}")
            max_load = max(DC_LOADv) if DC_LOADv else 0.0
            if peak_import_mw + tol < max_load:
                _bad(f"Peak import {peak_import_mw} is below max hourly load {max_load}.")

        ren_avail_prof = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})

        for t in range(T):
            if abs(F12v[t]) > cfg.line_cap_12 + tol:
                _bad(f"|F12[{t}]|={abs(F12v[t])} exceeds cap {cfg.line_cap_12}.")
            if abs(F23v[t]) > cfg.line_cap_23 + tol:
                _bad(f"|F23[{t}]|={abs(F23v[t])} exceeds cap {cfg.line_cap_23}.")
            if abs(F13v[t]) > cfg.line_cap_13 + tol:
                _bad(f"|F13[{t}]|={abs(F13v[t])} exceeds cap {cfg.line_cap_13}.")

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

            if allow_heat and K_HX > 0:
                pipe_cap = cfg.heat_pipe_cap_share * sc.heat_demand_level
                if H_RECv[t] > sc.heat_recovery_frac * (ITv[t] + COOLv[t]) + tol:
                    _bad(f"H_REC[{t}] exceeds recoverable heat.")
                if H_RECv[t] > K_HX + tol:
                    _bad(f"H_REC[{t}] exceeds K_HX.")
                if H_FLOWv[t] > H_RECv[t] + tol:
                    _bad(f"H_FLOW[{t}] exceeds H_REC.")
                if H_FLOWv[t] > pipe_cap + tol:
                    _bad(f"H_FLOW[{t}] exceeds pipe capacity.")
                if abs(H_INv[t] - (1.0 - cfg.heat_pipe_loss_frac) * H_FLOWv[t]) > tol * (1.0 + abs(H_FLOWv[t])):
                    _bad(f"H_IN[{t}] inconsistent with pipe loss.")

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
                if abs(H_RECv[t]) > tol or abs(H_FLOWv[t]) > tol or abs(H_INv[t]) > tol or abs(H_SOLDv[t]) > tol:
                    _bad(f"Heat vars nonzero while heat disabled at t={t}.")
                for k in range(K):
                    if abs(Q_DHv[k][t]) > tol:
                        _bad(f"Q_DH nonzero while heat disabled at t={t}.")
                if cfg.enable_thermal_storage:
                    if abs(CH_THv[t]) > tol or abs(DIS_THv[t]) > tol:
                        _bad(f"Thermal storage flows nonzero while heat disabled at t={t}.")

    fuel_price = prof.get("fuel_price", [0.0] * T)
    co2_price = prof.get("p_co2", [0.0] * T)
    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    el_cost_val = srv_scale * sum(
        (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_THv[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    co2_cost_val = srv_scale * sum(
        co2_price[t] * th_ef[n] * G_THv[n][t] * dt * w[t]
        for n in cfg.network_nodes for t in range(T)
    )
    trans_cost_val = 0.0
    if cfg.transmission_cost_eur_per_mwh > 0:
        trans_cost_val = srv_scale * cfg.transmission_cost_eur_per_mwh * sum(
            (abs(F12v[t]) + abs(F23v[t]) + abs(F13v[t])) * dt * w[t] for t in range(T)
        )
    heat_rev_val = srv_scale * sum(
        (cfg.dh_value_multipliers[k] * p_heat_eff[t]) * pulp.value(Q_DH[k][t]) * dt * w[t]
        for k in range(K) for t in range(T)
    )
    flex_rev_val = srv_scale * sum(p_flex[t] * RFv[t] * dt * w[t] for t in range(T))
    it_rev_val = srv_scale * sum(p_it[t] * ITv[t] * dt * w[t] for t in range(T))
    if allow_it_shortfall:
        cong_factor_v = U_GRIDv if cfg.sla_penalty_only_when_congested else [1.0] * T
        sla_penalty_val = sum(
            (cfg.sla_penalty_eur_per_mwh * cong_factor_v[t])
            * max(0.0, it_demand[t] - ITv[t]) * dt * w[t]
            for t in range(T)
        )
    else:
        sla_penalty_val = 0.0
    credit_eur = heat_rev_val + flex_rev_val
    demand_charge_val = demand_charge_horizon(cfg) * peak_import_mw

    bat_charge_mwh = sum(CHv[t] * w[t] for t in range(T)) * dt
    bat_discharge_mwh = sum(DISv[t] * w[t] for t in range(T)) * dt

    avg_heat_price = sum(p_heat_eff) / max(1, len(p_heat_eff)) if p_heat_eff else 0.0
    avg_flex_price = sum(p_flex) / max(1, len(p_flex)) if p_flex else 0.0

    include_co2_for_sipue = cfg.sipue_include_co2_penalty
    # Compute DC's proportional share of total grid CO2 for siPUE
    # Total grid generation (thermal only — renewables are zero-carbon)
    total_grid_gen_mwh = sum(
        G_THv[n][t] * dt * w[t] for n in cfg.network_nodes for t in range(T)
    )
    dc_co2_share = co2_cost_val * (E_GRID / max(1e-9, total_grid_gen_mwh)) if total_grid_gen_mwh > 1e-9 else 0.0
    siPUE, credit_mwh, credit_mwh_heat, credit_mwh_flex, co2_mwh, price_used, price_source, price_floor_applied = compute_sipue(
        E_FAC=E_FAC,
        E_IT=E_IT,
        credit_eur=credit_eur,
        heat_rev_eur=heat_rev_val,
        flex_rev_eur=flex_rev_val,
        avg_heat_price=avg_heat_price,
        avg_flex_price=avg_flex_price,
        co2_cost_eur=dc_co2_share,
        el_cost=el_settlement_eur,
        el_settlement_eur=el_settlement_eur,
        E_GRID=E_GRID,
        p_el_ref=prof.get("p_el", [0.0] * T),
        dc_load_series=DC_LOADv,
        settle_lmp_series=settle_lmp_series,
        normalization=cfg.sipue_credit_normalization,
        credit_price_mode=cfg.sipue_credit_price_mode,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
        price_floor_eur_per_mwh=cfg.sipue_price_floor_eur_per_mwh,
        include_co2_penalty=include_co2_for_sipue,
        co2_scale=cfg.sipue_co2_scale,
        co2_ref_price_eur_per_mwh=cfg.sipue_co2_ref_price_eur_per_mwh
    )

    eff_benefit_val = eff_reduction * (
        cfg.eff_cost_saving_eur_per_mwh_fac_per_eff * E_FAC
        + cfg.eff_opex_credit_eur_per_mwh_it_per_eff * E_IT
    )

    emissions_tco2 = sum(th_ef[n] * G_THv[n][t] * dt * w[t] for n in cfg.network_nodes for t in range(T))
    total_cost = pulp.value(m.objective)
    land_m2_val = compute_land_m2(K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg)
    land_cost_val = land_cost_horizon(land_m2_val, cfg)
    eff_capex_val = capex_scale(cfg) * eff_capex_from_reduction(eff_reduction, cfg)
    total_profit_system = (
        it_rev_val + heat_rev_val + flex_rev_val
        - el_cost_val - co2_cost_val - trans_cost_val - demand_charge_val - sla_penalty_val
        - capex - eff_capex_val - land_cost_val
        + eff_benefit_val
    )
    total_profit = (
        it_rev_val + heat_rev_val + flex_rev_val
        - srv_scale * el_settlement_eur - demand_charge_val - sla_penalty_val
        - capex - eff_capex_val - land_cost_val
        + eff_benefit_val
    )

    return {
        "status": pulp.LpStatus[m.status],
        "total_cost": total_cost,
        "total_profit": total_profit,
        "total_profit_system": total_profit_system,
        "E_IT": E_IT,
        "E_FAC": E_FAC,
        "E_GRID": E_GRID,
        "E_LOAD": E_LOAD,
        "PUE": PUE,
        "siPUE": siPUE,
        "credit_eur": credit_eur,
        "credit_mwh": credit_mwh,
        "credit_mwh_heat": credit_mwh_heat,
        "credit_mwh_flex": credit_mwh_flex,
        "co2_mwh": co2_mwh,
        "sipue_price_used": price_used,
        "sipue_price_source": price_source,
        "lmp_settle_avg_dc": lmp_settle_avg_dc,
        "lmp_shadow_avg_dc": lmp_shadow_avg_dc,
        "settle_lmp_series": settle_lmp_series,
        "grid_util_series": util_max,
        "dc_load_series": DC_LOADv,
        "emissions_tco2": emissions_tco2,

        "heat_MWhth": sum(H_SOLDv[t] * w[t] for t in range(T)) * dt,
        "flex_cap_hours": sum(RFv[t] * w[t] for t in range(T)) * dt,
        "grid_util_avg": sum(util_max[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "grid_util_max": max(util_max) if util_max else float("nan"),
        "line_util_12_avg": sum(util_12[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_23_avg": sum(util_23[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_13_avg": sum(util_13[t] * w[t] for t in range(T)) / max(1e-9, w_sum),
        "line_util_12_max": max(util_12) if util_12 else float("nan"),
        "line_util_23_max": max(util_23) if util_23 else float("nan"),
        "line_util_13_max": max(util_13) if util_13 else float("nan"),
        "heat_pipe_util_avg": (
            sum((H_FLOWv[t] / max(1e-9, cfg.heat_pipe_cap_share * sc.heat_demand_level)) * w[t] for t in range(T)) / max(1e-9, w_sum)
        ),
        "heat_pipe_util_max": (
            max((H_FLOWv[t] / max(1e-9, cfg.heat_pipe_cap_share * sc.heat_demand_level)) for t in range(T)) if T > 0 else float("nan")
        ),

        "el_cost_eur": el_cost_val,
        "el_settlement_eur": el_settlement_eur,
        "co2_cost_eur": co2_cost_val,
        "trans_cost_eur": trans_cost_val,
        "demand_charge_eur": demand_charge_val,
        "peak_import_mw": peak_import_mw,
        "heat_rev_eur": heat_rev_val,
        "flex_rev_eur": flex_rev_val,
        "it_rev_eur": it_rev_val,
        "sla_penalty_eur": sla_penalty_val,
        "land_m2": land_m2_val,
        "land_cost_eur": land_cost_val,
        "eff_capex_eur": eff_capex_val,
        "eff_benefit_eur": eff_benefit_val,
        "bat_charge_mwh": bat_charge_mwh,
        "bat_discharge_mwh": bat_discharge_mwh,

        "allow_heat": allow_heat,
        "allow_flex": allow_flex,
        "allow_battery": allow_battery,
        "eff_reduction": eff_reduction,
        "sipue_price_used": price_used,
        "sipue_price_floor_applied": price_floor_applied,
    }



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
    dh_cap = prof["dh_cap"]
    p_flex = prof["p_flex"]
    p_it = prof.get("p_it", [0.0] * T)
    p_co2 = prof["p_co2"]
    cop_t = prof.get("cop_t", [sc.hp_cop] * T)
    node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
    ren_avail = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
    fuel_price = prof.get("fuel_price", [0.0] * T)
    dc_node = prof.get("dc_node", cfg.network_nodes[0])

    allow_heat = bool(prof.get("allow_heat", True))
    allow_flex = bool(prof.get("allow_flex", True))
    allow_battery = bool(prof.get("allow_battery", True))
    allow_it_shortfall_pf = (
        cfg.objective_mode == "profit_max"
        and cfg.allow_it_shortfall
        and cfg.sla_penalty_eur_per_mwh > 0
    )

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
        K_FLEX = pulp.LpVariable("K_FLEX", lowBound=0, upBound=cfg.K_FLEX_max)
        K_BAT_E = pulp.LpVariable("K_BAT_E", lowBound=0, upBound=cfg.K_BAT_E_max)
        K_BAT_P = pulp.LpVariable("K_BAT_P", lowBound=0, upBound=cfg.K_BAT_P_max)
        if allow_heat and cfg.min_hx_mwth_per_it_mw > 0:
            min_hx = cfg.min_hx_mwth_per_it_mw * sc.base_it_mw
            m += K_HX >= min_hx, "MinHX"
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
        H_FLOW = pulp.LpVariable.dicts("H_FLOW", range(T), lowBound=0)
        H_IN = pulp.LpVariable.dicts("H_IN", range(T), lowBound=0)
        H_SOLD = pulp.LpVariable.dicts("H_SOLD", range(T), lowBound=0)
        Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}", range(T), lowBound=0) for k in range(K)}

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
        F12 = pulp.LpVariable.dicts("F12", range(T), lowBound=-cfg.line_cap_12, upBound=cfg.line_cap_12)
        F23 = pulp.LpVariable.dicts("F23", range(T), lowBound=-cfg.line_cap_23, upBound=cfg.line_cap_23)
        F13 = pulp.LpVariable.dicts("F13", range(T), lowBound=-cfg.line_cap_13, upBound=cfg.line_cap_13)
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
        if not allow_flex:
            m += K_FLEX == 0, "DisableFlex"
        if not allow_battery:
            m += K_BAT_E == 0, "DisableBatteryE"
            m += K_BAT_P == 0, "DisableBatteryP"

        if cfg.objective_mode == "cost_min":
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
            + cfg.ramp_k_flex * K_FLEX
            + cfg.ramp_k_bat_e * K_BAT_E
            + cfg.ramp_k_bat_p * K_BAT_P
        )
        m += ramp_share <= cfg.ramp_base_share - cfg.ramp_tau_per_mw * size_proxy_expr, "RampShareUb"

        alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

        land_m2_expr = compute_land_m2(K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg)
        land_limit = sc.land_limit_m2 if sc.land_limit_m2 is not None else cfg.land_limit_m2
        if land_limit is not None:
            m += land_m2_expr <= land_limit, "LandLimit"
        land_cost_expr = land_cost_horizon(land_m2_expr, cfg)
        force_it_flex = bool(prof.get("force_it_flex", False))
        flex_frac_eff = min(0.30, sc.flex_frac * (cfg.it_flex_scale_services_active if allow_flex else 1.0))
        for t in range(T):
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
            elif cfg.objective_mode == "cost_min":
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
                m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
                m += H_REC[t] <= K_HX
                m += H_FLOW[t] <= H_REC[t]
                m += H_FLOW[t] <= pipe_cap
                m += H_IN[t] == (1.0 - cfg.heat_pipe_loss_frac) * H_FLOW[t]

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
                m += H_FLOW[t] == 0
                m += H_IN[t] == 0
                m += H_SOLD[t] == 0
                for k in range(K):
                    m += Q_DH[k][t] == 0
                if cfg.enable_thermal_storage:
                    m += CH_TH[t] == 0
                    m += DIS_TH[t] == 0
                    m += S_TH[t + 1] == S_TH[t]

            min_it = (1 - flex_frac_eff) * it_base[t] if (cfg.fix_it_profile or cfg.objective_mode == "cost_min") else 0.0
            if allow_flex:
                m += RF[t] <= K_FLEX
                m += RF[t] <= IT_SERV[t] - min_it
            else:
                m += RF[t] == 0

            m += SOC[t] <= K_BAT_E
            m += SOC[t + 1] <= K_BAT_E
            if allow_battery:
                m += CH[t] <= K_BAT_P
                m += DIS[t] <= K_BAT_P
                m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1.0 / cfg.eta_dis) * DIS[t] * dt
            else:
                m += CH[t] == 0
                m += DIS[t] == 0
                m += SOC[t + 1] == SOC[t]

            P_HP = H_FLOW[t] * (1.0 / cop_t[t])
            P_SERV = (
                cfg.service_parasitic_heat_frac * H_SOLD[t]
                + cfg.service_parasitic_flex_frac * RF[t]
                + cfg.service_parasitic_hx_mw_per_mwth * K_HX
                + cfg.service_parasitic_flex_mw_per_mw * K_FLEX
            )

            m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP + P_SERV + CH[t] - DIS[t]
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

            m += U_GRID[t] >= F12[t] / cfg.line_cap_12
            m += U_GRID[t] >= -F12[t] / cfg.line_cap_12
            m += U_GRID[t] >= F23[t] / cfg.line_cap_23
            m += U_GRID[t] >= -F23[t] / cfg.line_cap_23
            m += U_GRID[t] >= F13[t] / cfg.line_cap_13
            m += U_GRID[t] >= -F13[t] / cfg.line_cap_13

        srv_scale = service_revenue_scale(cfg, sc)

        el_cost_expr = srv_scale * pulp.lpSum(
            (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
            for n in cfg.network_nodes for t in range(T)
        )
        co2_cost_expr = srv_scale * pulp.lpSum(
            p_co2[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
            for n in cfg.network_nodes for t in range(T)
        )
        trans_cost_expr = 0
        if cfg.transmission_cost_eur_per_mwh > 0:
            trans_cost_expr = srv_scale * pulp.lpSum(
                cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
                for t in range(T)
            )

        heat_rev_expr = srv_scale * pulp.lpSum(
            (cfg.dh_value_multipliers[k] * p_heat_eff[t]) * Q_DH[k][t] * dt * w[t]
            for k in range(K) for t in range(T)
        )
        flex_rev_expr = srv_scale * pulp.lpSum(p_flex[t] * RF[t] * dt * w[t] for t in range(T))
        it_rev_expr = srv_scale * pulp.lpSum(p_it[t] * IT_SERV[t] * dt * w[t] for t in range(T))
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

        capex_expr = capex_scale(cfg) * (
            cfg.capex_HX * K_HX
            + cfg.capex_FLEX * K_FLEX
            + cfg.capex_BAT_E * K_BAT_E
            + cfg.capex_BAT_P * K_BAT_P
            + eff_capex_from_reduction(eff_reduction, cfg)
        )

        if cfg.objective_mode == "profit_max":
            profit_expr = (
                it_rev_expr + heat_rev_expr + flex_rev_expr
                - el_cost_expr - co2_cost_expr - trans_cost_expr - demand_charge_expr - sla_penalty_expr
                - capex_expr - land_cost_expr
                + eff_benefit_expr
            )
            m += -profit_expr + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
        else:
            m += (
                el_cost_expr + co2_cost_expr + trans_cost_expr + demand_charge_expr
                - heat_rev_expr - flex_rev_expr
                + capex_expr + land_cost_expr
                - eff_benefit_expr
                + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
            )

        m.solve(make_solver_lp(cfg))

        return pulp.value(m.objective), {
            "K_HX": pulp.value(K_HX),
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
        eff_reduction=cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx]
    )

    return best_obj, {
        "K_HX": best["K_HX"],
        "K_FLEX": best["K_FLEX"],
        "K_BAT_E": best["K_BAT_E"],
        "K_BAT_P": best["K_BAT_P"],
        "eff_level_index": best_idx,
        "eff_reduction": cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx],
        "eff_capex": eff_capex_from_reduction(
            cfg.fixed_eff_reduction if cfg.fixed_eff_reduction is not None else cfg.eff_levels[best_idx], cfg
        ),
        "status": best["status"],
        "lmp_settle_avg_dc": pf_ops.get("lmp_settle_avg_dc"),
        "lmp_shadow_avg_dc": pf_ops.get("lmp_shadow_avg_dc"),
    }


def solve_minmax_regret(
    scenarios: List[Scenario],
    profiles: Dict[int, Dict[str, List[float]]],
    C_star: Dict[int, float],
    cfg: Config,
) -> Dict:
    if cfg.regret_use_profit_tradeoff:
        if not (0.0 <= cfg.regret_tradeoff_lambda <= 1.0):
            raise ValueError("regret_tradeoff_lambda must be in [0, 1].")
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

    min_hx_lb = 0.0
    min_flex_lb = 0.0
    min_bat_e_lb = 0.0
    min_bat_p_lb = 0.0
    it_mw_reference = None
    vals = [sc.base_it_mw for sc in scenarios]
    if vals:
        it_mw_reference = sum(vals) / max(1, len(vals))
        if cfg.min_hx_mwth_per_it_mw > 0:
            min_hx_lb = cfg.min_hx_mwth_per_it_mw * it_mw_reference
        if cfg.min_flex_mw_per_it_mw > 0:
            min_flex_lb = cfg.min_flex_mw_per_it_mw * it_mw_reference
        if cfg.min_bat_e_hours > 0:
            min_bat_e_lb = cfg.min_bat_e_hours * it_mw_reference
        if cfg.min_bat_p_mw_per_it_mw > 0:
            min_bat_p_lb = cfg.min_bat_p_mw_per_it_mw * it_mw_reference

    def solve_fixed_eff(eff_reduction: float) -> Dict:
        eff_reduction = max(0.0, min(0.90, eff_reduction))
        m = pulp.LpProblem(f"MinMaxRegret_T{cfg.T}_eff{eff_reduction:.2f}", pulp.LpMinimize)

        K_HX = pulp.LpVariable("K_HX", lowBound=0, upBound=cfg.K_HX_max)
        K_FLEX = pulp.LpVariable("K_FLEX", lowBound=0, upBound=cfg.K_FLEX_max)
        K_BAT_E = pulp.LpVariable("K_BAT_E", lowBound=0, upBound=cfg.K_BAT_E_max)
        K_BAT_P = pulp.LpVariable("K_BAT_P", lowBound=0, upBound=cfg.K_BAT_P_max)

        if min_hx_lb > 0:
            m += K_HX >= min_hx_lb, "MinHX"
        if min_flex_lb > 0:
            m += K_FLEX >= min_flex_lb, "MinFlex"
        if min_bat_e_lb > 0:
            m += K_BAT_E >= min_bat_e_lb, "MinBatE"
        if min_bat_p_lb > 0:
            m += K_BAT_P >= min_bat_p_lb, "MinBatP"

        Z = pulp.LpVariable("Z", lowBound=0)

        m += K_BAT_P <= K_BAT_E, "BatteryPowerLeqEnergy"

        land_m2_expr = compute_land_m2(K_HX, K_FLEX, K_BAT_E, K_BAT_P, eff_reduction, cfg)
        land_cost_expr = land_cost_horizon(land_m2_expr, cfg)

        land_limits = []
        for sc in scenarios:
            lim = sc.land_limit_m2 if sc.land_limit_m2 is not None else cfg.land_limit_m2
            if lim is not None:
                land_limits.append(lim)
        if land_limits:
            m += land_m2_expr <= min(land_limits), "LandLimit_shared"

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
            dh_cap = prof["dh_cap"]
            p_flex = prof["p_flex"]
            p_it = prof.get("p_it", [0.0] * T)
            p_co2 = prof["p_co2"]
            cop_t = prof.get("cop_t", [sc.hp_cop] * T)
            node_loads = prof.get("node_loads", {n: [0.0] * T for n in cfg.network_nodes})
            ren_avail = prof.get("ren_avail", {n: [0.0] * T for n in cfg.network_nodes})
            fuel_price = prof.get("fuel_price", [0.0] * T)
            dc_node = prof.get("dc_node", cfg.network_nodes[0])

            allow_heat = bool(prof.get("allow_heat", True))
            allow_flex = bool(prof.get("allow_flex", True))
            allow_battery = bool(prof.get("allow_battery", True))
            allow_it_shortfall_sc = (
                cfg.objective_mode == "profit_max"
                and cfg.allow_it_shortfall
                and cfg.sla_penalty_eur_per_mwh > 0
            )

            alpha_base = prof.get("alpha_base", None)
            if alpha_base is None:
                alpha_base = make_overhead_alpha_base(sc, T, cfg)
            alpha_eff = [a * (1.0 - eff_reduction) for a in alpha_base]

            p_heat_eff = p_heat
            dh_cap_eff = dh_cap

            IT_SERV = pulp.LpVariable.dicts(f"IT_SERV_s{sc.sid}", range(T), lowBound=0)
            COOL = pulp.LpVariable.dicts(f"COOL_s{sc.sid}", range(T), lowBound=0)

            H_REC = pulp.LpVariable.dicts(f"H_REC_s{sc.sid}", range(T), lowBound=0)
            H_FLOW = pulp.LpVariable.dicts(f"H_FLOW_s{sc.sid}", range(T), lowBound=0)
            H_IN = pulp.LpVariable.dicts(f"H_IN_s{sc.sid}", range(T), lowBound=0)
            H_SOLD = pulp.LpVariable.dicts(f"H_SOLD_s{sc.sid}", range(T), lowBound=0)
            Q_DH = {k: pulp.LpVariable.dicts(f"Q_DH_{k+1}_s{sc.sid}", range(T), lowBound=0) for k in range(K)}

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
            F12 = pulp.LpVariable.dicts(f"F12_s{sc.sid}", range(T), lowBound=-cfg.line_cap_12, upBound=cfg.line_cap_12)
            F23 = pulp.LpVariable.dicts(f"F23_s{sc.sid}", range(T), lowBound=-cfg.line_cap_23, upBound=cfg.line_cap_23)
            F13 = pulp.LpVariable.dicts(f"F13_s{sc.sid}", range(T), lowBound=-cfg.line_cap_13, upBound=cfg.line_cap_13)
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

            if cfg.objective_mode == "cost_min":
                m += pulp.lpSum(IT_SERV[t] * dt for t in range(T)) == sum(it_base) * dt

            m += SOC[0] == 0.5 * K_BAT_E
            m += SOC[T] == SOC[0]

            if cfg.enable_thermal_storage:
                m += S_TH[0] == 0.0
                m += S_TH[T] == 0.0

            n1, n2, n3 = cfg.network_nodes

            size_proxy_expr = (
                sc.base_it_mw
                + cfg.ramp_k_hx * K_HX
                + cfg.ramp_k_flex * K_FLEX
                + cfg.ramp_k_bat_e * K_BAT_E
                + cfg.ramp_k_bat_p * K_BAT_P
            )
            m += ramp_share <= cfg.ramp_base_share - cfg.ramp_tau_per_mw * size_proxy_expr

            force_it_flex = bool(prof.get("force_it_flex", False))
            flex_frac_eff = min(0.30, sc.flex_frac * (cfg.it_flex_scale_services_active if allow_flex else 1.0))
            for t in range(T):
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
                elif cfg.objective_mode == "cost_min":
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
                    m += H_REC[t] <= sc.heat_recovery_frac * (IT_SERV[t] + COOL[t])
                    m += H_REC[t] <= K_HX
                    m += H_FLOW[t] <= H_REC[t]
                    m += H_FLOW[t] <= pipe_cap
                    m += H_IN[t] == (1.0 - cfg.heat_pipe_loss_frac) * H_FLOW[t]

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
                    m += H_FLOW[t] == 0
                    m += H_IN[t] == 0
                    m += H_SOLD[t] == 0
                    for k in range(K):
                        m += Q_DH[k][t] == 0
                    if cfg.enable_thermal_storage:
                        m += CH_TH[t] == 0
                        m += DIS_TH[t] == 0
                        m += S_TH[t + 1] == S_TH[t]

                min_it = (1 - flex_frac_eff) * it_base[t] if (cfg.fix_it_profile or cfg.objective_mode == "cost_min") else 0.0
                if allow_flex:
                    m += RF[t] <= K_FLEX
                    m += RF[t] <= IT_SERV[t] - min_it
                else:
                    m += RF[t] == 0

                m += SOC[t] <= K_BAT_E
                m += SOC[t + 1] <= K_BAT_E
                if allow_battery:
                    m += CH[t] <= K_BAT_P
                    m += DIS[t] <= K_BAT_P
                    m += SOC[t + 1] == SOC[t] + cfg.eta_ch * CH[t] * dt - (1.0 / cfg.eta_dis) * DIS[t] * dt
                else:
                    m += CH[t] == 0
                    m += DIS[t] == 0
                    m += SOC[t + 1] == SOC[t]

                P_HP = H_FLOW[t] * (1.0 / cop_t[t])
                P_SERV = (
                    cfg.service_parasitic_heat_frac * H_SOLD[t]
                    + cfg.service_parasitic_flex_frac * RF[t]
                    + cfg.service_parasitic_hx_mw_per_mwth * K_HX
                    + cfg.service_parasitic_flex_mw_per_mw * K_FLEX
                )
                m += DC_LOAD[t] == IT_SERV[t] + COOL[t] + P_HP + P_SERV + CH[t] - DIS[t]
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

                m += U_GRID[t] >= F12[t] / cfg.line_cap_12
                m += U_GRID[t] >= -F12[t] / cfg.line_cap_12
                m += U_GRID[t] >= F23[t] / cfg.line_cap_23
                m += U_GRID[t] >= -F23[t] / cfg.line_cap_23
                m += U_GRID[t] >= F13[t] / cfg.line_cap_13
                m += U_GRID[t] >= -F13[t] / cfg.line_cap_13

            srv_scale = service_revenue_scale(cfg, sc)

            el_cost = srv_scale * pulp.lpSum(
                (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * G_TH[n][t] * dt * w[t]
                for n in cfg.network_nodes for t in range(T)
            )
            co2_cost = srv_scale * pulp.lpSum(
                p_co2[t] * th_ef[n] * G_TH[n][t] * dt * w[t]
                for n in cfg.network_nodes for t in range(T)
            )
            trans_cost = 0
            if cfg.transmission_cost_eur_per_mwh > 0:
                trans_cost = srv_scale * pulp.lpSum(
                    cfg.transmission_cost_eur_per_mwh * (AF12[t] + AF23[t] + AF13[t]) * dt * w[t]
                    for t in range(T)
                )
            heat_rev = srv_scale * pulp.lpSum(
                (cfg.dh_value_multipliers[k] * p_heat_eff[t]) * Q_DH[k][t] * dt * w[t]
                for k in range(K) for t in range(T)
            )
            flex_rev = srv_scale * pulp.lpSum(p_flex[t] * RF[t] * dt * w[t] for t in range(T))
            it_rev = srv_scale * pulp.lpSum(p_it[t] * IT_SERV[t] * dt * w[t] for t in range(T))
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

            capex = capex_scale(cfg) * (
                cfg.capex_HX * K_HX
                + cfg.capex_FLEX * K_FLEX
                + cfg.capex_BAT_E * K_BAT_E
                + cfg.capex_BAT_P * K_BAT_P
                + eff_capex_from_reduction(eff_reduction, cfg)
            )

            if cfg.objective_mode == "profit_max":
                profit = it_rev + heat_rev + flex_rev - el_cost - co2_cost - trans_cost - demand_charge - sla_penalty - capex - land_cost_expr + eff_benefit
                m += Cost == -profit + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
            else:
                m += Cost == el_cost + co2_cost + trans_cost + demand_charge - heat_rev - flex_rev + capex + land_cost_expr - eff_benefit + cfg.eps_u_grid * pulp.lpSum(U_GRID[t] for t in range(T))
            m += Z >= Cost - C_star[sc.sid]

        if cfg.regret_use_profit_tradeoff and cfg.objective_mode == "profit_max":
            n_s = max(1, len(Cost_s_vars))
            avg_profit = -pulp.lpSum(Cost_s_vars[sid] for sid in Cost_s_vars) / n_s
            m += cfg.regret_tradeoff_lambda * Z - (1.0 - cfg.regret_tradeoff_lambda) * avg_profit
        else:
            m += Z
        m.solve(make_solver_lp(cfg))

        return {
            "status": pulp.LpStatus[m.status],
            "K_HX": pulp.value(K_HX),
            "K_FLEX": pulp.value(K_FLEX),
            "K_BAT_E": pulp.value(K_BAT_E),
            "K_BAT_P": pulp.value(K_BAT_P),
            "eff_reduction": eff_reduction,
            "eff_capex": capex_scale(cfg) * eff_capex_from_reduction(eff_reduction, cfg),
            "Z": pulp.value(Z),
            "avg_profit": (-sum(pulp.value(v) for v in Cost_s_vars.values()) / max(1, len(Cost_s_vars)))
            if cfg.objective_mode == "profit_max" else None,
            "min_hx_it_mw_reference": it_mw_reference,
            "Cost_s": {sid: pulp.value(v) for sid, v in Cost_s_vars.items()},
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
            if best is None or res["Z"] < best["Z"]:
                best = res
                best_idx = i

    assert best is not None
    best["eff_level_index"] = best_idx
    if cfg.fixed_eff_reduction is not None:
        best["eff_reduction"] = cfg.fixed_eff_reduction
        best["eff_capex"] = eff_capex_from_reduction(cfg.fixed_eff_reduction, cfg)
    return best
