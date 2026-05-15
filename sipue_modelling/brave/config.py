from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LOCKED_CALIBRATION = {
    "capex_HX": 22000.0,
    "capex_FLEX": 20000.0,
    "capex_BAT_E": 25000.0,
    "capex_BAT_P": 8000.0,
    "eff_levels": (0.00, 0.15),
    "fixed_eff_reduction": 0.15,
    "it_price_multiplier": 3.0,
    "service_parasitic_heat_frac": 0.03,
    "service_parasitic_flex_frac": 0.015,
    "min_hx_mwth_per_it_mw": 0.11,
    "min_flex_mw_per_it_mw": 0.09,
    "min_bat_e_hours": 0.5,
    "min_bat_p_mw_per_it_mw": 0.10,
    "heat_price_multiplier": 0.60,
    "flex_price_multiplier": 0.25,
    "heat_price_multipliers_by_sid": (1.0, 1.3, 1.1, 1.4, 0.7, 1.3, 1.1, 1.2, 0.9, 0.7),
    "flex_price_multipliers_by_sid": (1.0, 1.1, 1.0, 1.1, 0.9, 1.1, 1.0, 1.1, 0.9, 0.9),
    "sipue_credit_normalization": "settle_lmp_dc",
    "sipue_credit_price_mode": "lmp",
}

def _validate_locked_calibration(cfg: Config) -> None:
    for key, val in LOCKED_CALIBRATION.items():
        if getattr(cfg, key) != val:
            raise ValueError(f"Locked calibration mismatch: {key}={getattr(cfg, key)} (expected {val})")
    if cfg.heat_price_multiplier != LOCKED_CALIBRATION["heat_price_multiplier"]:
        raise ValueError(
            f"Locked calibration mismatch: heat_price_multiplier={cfg.heat_price_multiplier} "
            f"(expected {LOCKED_CALIBRATION['heat_price_multiplier']})"
        )
    if cfg.flex_price_multiplier != LOCKED_CALIBRATION["flex_price_multiplier"]:
        raise ValueError(
            f"Locked calibration mismatch: flex_price_multiplier={cfg.flex_price_multiplier} "
            f"(expected {LOCKED_CALIBRATION['flex_price_multiplier']})"
        )
    if tuple(cfg.heat_price_multipliers_by_sid) != LOCKED_CALIBRATION["heat_price_multipliers_by_sid"]:
        raise ValueError(
            "Locked calibration mismatch: heat_price_multipliers_by_sid="
            f"{tuple(cfg.heat_price_multipliers_by_sid)} "
            f"(expected {LOCKED_CALIBRATION['heat_price_multipliers_by_sid']})"
        )
    if tuple(cfg.flex_price_multipliers_by_sid) != LOCKED_CALIBRATION["flex_price_multipliers_by_sid"]:
        raise ValueError(
            "Locked calibration mismatch: flex_price_multipliers_by_sid="
            f"{tuple(cfg.flex_price_multipliers_by_sid)} "
            f"(expected {LOCKED_CALIBRATION['flex_price_multipliers_by_sid']})"
        )
    if cfg.sipue_credit_normalization != LOCKED_CALIBRATION["sipue_credit_normalization"]:
        raise ValueError(
            f"Locked calibration mismatch: sipue_credit_normalization={cfg.sipue_credit_normalization} "
            f"(expected {LOCKED_CALIBRATION['sipue_credit_normalization']})"
        )
    if cfg.sipue_credit_price_mode != LOCKED_CALIBRATION["sipue_credit_price_mode"]:
        raise ValueError(
            f"Locked calibration mismatch: sipue_credit_price_mode={cfg.sipue_credit_price_mode} "
            f"(expected {LOCKED_CALIBRATION['sipue_credit_price_mode']})"
        )

@dataclass
class Config:
    T: int = 8760  # 365 days hourly
    seed: int = 7
    representative_day_count: int = 7

    solver_lp: str = "highs"
    solver_msg: bool = False
    solver_time_limit_sec: Optional[int] = 86400  # 24 hours

    enforce_locked_calibration: bool = True
    calibration_tag: str = "locked_2026-03-07"

    regret_use_profit_tradeoff: bool = True
    regret_tradeoff_lambda: float = 0.7

    capex_HX: float = 22000.0       # EUR per MW_th (annualised) — IEA: 190-250k full, 10% ann.
    capex_FLEX: float = 20000.0     # EUR per MW (annualised) — DR controls + grid interface
    capex_BAT_E: float = 25000.0    # EUR per MWh (annualised) — BNEF 2024: ~250k full, 10% ann.
    capex_BAT_P: float = 8000.0     # EUR per MW (annualised) — power electronics component
    capex_scale_with_horizon: bool = True


    eff_levels: Tuple[float, ...] = (0.00, 0.15)
    eff_capex_levels: Tuple[float, ...] = (0.0, 1680000.0, 9800000.0)  # EUR (annualised)
    eff_cost_saving_eur_per_mwh_fac_per_eff: float = 0.0
    eff_opex_credit_eur_per_mwh_it_per_eff: float = 0.0
    fixed_eff_reduction: Optional[float] = 0.15

    objective_mode: str = "profit_max"  # "cost_min" or "profit_max"

    fix_it_profile: bool = True
    it_capacity_multiplier: float = 1.0
    it_demand_scale: float = 1.15
    it_price_level: float = 0.0
    it_price_multiplier: float = 3.0
    it_price_vol: float = 10.0
    it_price_daily_amp: float = 14.0
    heat_price_multiplier: float = 0.60
    flex_price_multiplier: float = 0.25
    heat_price_multipliers_by_sid: Tuple[float, ...] = (1.0, 1.3, 1.1, 1.4, 0.7, 1.3, 1.1, 1.2, 0.9, 0.7)
    flex_price_multipliers_by_sid: Tuple[float, ...] = (1.0, 1.1, 1.0, 1.1, 0.9, 1.1, 1.0, 1.1, 0.9, 0.9)

    K_HX_max: float = 50.0
    K_FLEX_max: float = 25.0
    K_BAT_E_max: float = 80.0
    K_BAT_P_max: float = 50.0

    eta_ch: float = 0.95
    eta_dis: float = 0.95

    sipue_clip_nonnegative: bool = True
    sipue_credit_normalization: str = "settle_lmp_dc"
    sipue_credit_price_mode: str = "lmp"
    sipue_ref_price_eur_per_mwh: float = 60.0

    p_dh_off_week: float = 0.02
    p_low_flex_week: float = 0.08
    p_low_heat_week: float = 0.05

    enable_time_varying_overhead: bool = True
    nordic_night_overhead_drop: float = 0.20
    day_overhead_bump: float = 0.07

    enable_congestion: bool = True
    cong_base: float = 0.15
    cong_amp: float = 0.30
    cong_spike_prob: float = 0.08
    cong_spike_mag: float = 0.50
    k_cong_el: float = 0.55
    k_cong_dh_cap: float = 0.35
    k_cong_heat_price: float = 0.25
    k_cong_gridcap: float = 0.30
    enable_nodal: bool = True
    grid_cap_share: Optional[float] = 1.5  # cap as multiple of base IT (if grid_cap_mw is None)
    grid_cap_mw: Optional[float] = None

    enable_temp_dependent_cop: bool = True
    cop_ref: float = 3.2
    t_amb_ref_c: float = 10.0
    t_dh_supply_ref_c: float = 70.0
    cop_slope_amb: float = 0.05
    cop_slope_dh: float = 0.03
    cop_min: float = 1.5
    cop_max: float = 5.0
    service_parasitic_heat_frac: float = 0.03
    service_parasitic_flex_frac: float = 0.015
    service_parasitic_hx_mw_per_mwth: float = 0.0
    service_parasitic_flex_mw_per_mw: float = 0.0
    it_flex_scale_services_active: float = 1.2

    enable_network_model: bool = True
    network_topology: str = "triangle3"
    network_nodes: Tuple[str, ...] = ("N1", "N2", "N3")
    line_cap_12: float = 2500.0
    line_cap_23: float = 2000.0
    line_cap_13: float = 1800.0
    node_assignment: Optional[Dict[str, str]] = None
    node_cost_base: Optional[Dict[str, float]] = None
    node_cost_amp: Optional[Dict[str, float]] = None
    node_load_base: Optional[Dict[str, float]] = None
    node_load_amp: Optional[Dict[str, float]] = None
    node_load_noise: float = 0.05

    network_timeseries_csv: str = "data/network_3node_timeseries.csv"
    network_load_scale: float = 0.04
    network_ren_scale: float = 0.04

    th_cap_N1_mw: float = 5500.0
    th_cap_N2_mw: float = 5500.0
    th_cap_N3_mw: float = 5500.0
    th_eff_N1: float = 0.52
    th_eff_N2: float = 0.47
    th_eff_N3: float = 0.42
    th_vom_N1: float = 8.0
    th_vom_N2: float = 14.0
    th_vom_N3: float = 22.0
    th_ef_N1: float = 0.20
    th_ef_N2: float = 0.20
    th_ef_N3: float = 0.20

    eps_u_grid: float = 1e-2

    transmission_cost_eur_per_mwh: float = 8.0

    enable_demand_charge: bool = True
    demand_charge_eur_per_mw_year: float = 50000.0

    heat_pipe_cap_share: float = 1.45
    heat_pipe_loss_frac: float = 0.05

    dh_segments: int = 3
    dh_seg_fracs: Tuple[float, ...] = (0.40, 0.35, 0.25)
    dh_value_multipliers: Tuple[float, ...] = (1.20, 1.05, 0.90)
    dh_base_value_source: str = "p_heat"

    lift_bins_c: Tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 120.0)
    cop_by_bin: Tuple[float, ...] = (3.4, 2.9, 2.5, 2.2, 1.9)

    enable_thermal_storage: bool = True
    k_th_store_share_of_daily_peak: float = 6.0
    th_store_p_ch_share: float = 1.0
    th_store_p_dis_share: float = 1.0
    eta_th_ch: float = 0.98
    eta_th_dis: float = 0.98
    th_store_loss_per_hour: float = 0.001

    validate_solutions: bool = True
    validate_tol: float = 1e-6
    require_duals: bool = True

    base_land_m2: float = 200.0
    land_a_hx: float = 30.0
    land_a_flex: float = 15.0
    land_a_bat_e: float = 5.0
    land_a_bat_p: float = 5.0
    land_cost_eur_per_m2: float = 35.0
    land_cost_is_annual: bool = True
    days_per_year: int = 365
    land_maintenance_frac: float = 0.0
    land_limit_m2: Optional[float] = None
    construction_delay_frac: float = 0.0
    override_heat_recovery_frac: Optional[float] = None
    override_hp_cop: Optional[float] = None

    ramp_base_share: float = 1.0
    ramp_tau_per_mw: float = 0.0
    ramp_k_hx: float = 0.0
    ramp_k_flex: float = 0.0
    ramp_k_bat_e: float = 0.0
    ramp_k_bat_p: float = 0.0

    sla_penalty_eur_per_mwh: float = 200.0
    sla_penalty_only_when_congested: bool = True
    allow_it_shortfall: bool = True
    max_it_curtail_frac: float = 0.02
    dc_grid_cap_share: float = 1.10

    export_csv: bool = True
    out_results_csv: str = "output/results_365day.csv"
    out_scenarios_csv: str = "output/scenarios.csv"
    export_plots: bool = True
    plot_output_dir: str = "output/paper-figures"
    plot_format: str = "pdf"
    export_regret_landscape: bool = False
    landscape_grid_nx: int = 8
    landscape_grid_ny: int = 8
    regret_landscape_tags: Optional[Tuple[str, ...]] = None
    export_diagnostic_plots: bool = False

    baseline_hx_mwth_per_it_mw: float = 0.25
    baseline_flex_mw_per_it_mw: float = 0.08
    baseline_bat_e_hours: float = 0.8
    baseline_bat_p_mw_per_it_mw: float = 0.20
    baseline_eff_reduction: float = 0.15

    min_hx_mwth_per_it_mw: float = 0.11
    min_flex_mw_per_it_mw: float = 0.09
    min_bat_e_hours: float = 0.5
    min_bat_p_mw_per_it_mw: float = 0.10

    sipue_price_floor_eur_per_mwh: float = 5.0
    sipue_include_co2_penalty: bool = True
    sipue_co2_scale: float = 1.0
    sipue_co2_ref_price_eur_per_mwh: float = 120.0
    co2_timeseries_scale: float = 1.0

    use_representative_days_annual: bool = True
