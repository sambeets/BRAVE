from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

LOCKED_CALIBRATION = {
    "capex_HX": 220000.0,
    "capex_HP": 600000.0,
    "capex_FLEX": 200000.0,
    "capex_BAT_E": 250000.0,
    "capex_BAT_P": 80000.0,
    "eff_levels": (0.00, 0.15),
    "fixed_eff_reduction": 0.15,
    "service_parasitic_heat_frac": 0.03,
    "service_parasitic_flex_frac": 0.015,
    "min_hx_mwth_per_it_mw": 0.11,
    "min_flex_mw_per_it_mw": 0.09,
    "min_bat_e_hours": 0.5,
    "min_bat_p_mw_per_it_mw": 0.10,
    "battery_capacity_headroom_multiplier": 2.0,
    "heat_price_multiplier": 0.60,
    "flex_price_multiplier": 0.25,
    "heat_price_multipliers_by_sid": (1.0, 1.3, 1.1, 1.4, 0.7, 1.3, 1.1, 1.2, 0.9, 0.7),
    "flex_price_multipliers_by_sid": (1.0, 1.1, 1.0, 1.1, 0.9, 1.1, 1.0, 1.1, 0.9, 0.9),
    "sipue_system_ref_price_eur_per_mwh": 120.0,
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
@dataclass
class Config:
    T: int = 8760  # 365 days hourly
    seed: int = 7
    representative_day_count: int = 7

    solver_lp: str = "highs"
    solver_msg: bool = False
    solver_time_limit_sec: Optional[int] = 86400  # 24 hours

    enforce_locked_calibration: bool = True

    robust_objective: str = "normalized_lexicographic"
    lexicographic_regret_tolerance: float = 1e-7

    capex_HX: float = 220000.0       # Upfront EUR per MW_th.
    capex_HP: float = 600000.0       # Upfront EUR per MW_e heat-pump input capacity.
    capex_FLEX: float = 200000.0     # Upfront EUR per MW of controllable capacity.
    capex_BAT_E: float = 250000.0    # Upfront EUR per MWh.
    capex_BAT_P: float = 80000.0     # Upfront EUR per MW.
    capex_scale_with_horizon: bool = False


    eff_levels: Tuple[float, ...] = (0.00, 0.15)
    eff_capex_levels: Tuple[float, ...] = (0.0, 16800000.0)  # Upfront EUR.
    eff_cost_saving_eur_per_mwh_fac_per_eff: float = 0.0
    eff_opex_credit_eur_per_mwh_it_per_eff: float = 0.0
    fixed_eff_reduction: Optional[float] = 0.15

    fix_it_profile: bool = True
    it_demand_scale: float = 1.15
    heat_price_multiplier: float = 0.60
    flex_price_multiplier: float = 0.25
    heat_price_multipliers_by_sid: Tuple[float, ...] = (1.0, 1.3, 1.1, 1.4, 0.7, 1.3, 1.1, 1.2, 0.9, 0.7)
    flex_price_multipliers_by_sid: Tuple[float, ...] = (1.0, 1.1, 1.0, 1.1, 0.9, 1.1, 1.0, 1.1, 0.9, 0.9)

    K_HX_max: float = 50.0
    K_HP_max: float = 25.0
    K_FLEX_max: float = 25.0
    # Absolute fallback ceilings, only used when the corresponding min_bat_*
    # floor below is disabled (set to 0). Otherwise battery capacity is
    # bounded by battery_capacity_headroom_multiplier x the min_bat_* floor,
    # scaled by each scenario's own base_it_mw -- see solve_perfect_foresight
    # and solve_minmax_regret. A fixed absolute cap divided by facility size
    # gives smaller facilities *more* per-MW-IT headroom, not less, which let
    # small archetypes install batteries many times oversized relative to
    # their own facility and earn an independent-benefit credit disconnected
    # from their actual energy footprint.
    K_BAT_E_max: float = 80.0
    K_BAT_P_max: float = 50.0
    # Ceiling on installable battery capacity, expressed as a multiple of the
    # min_bat_* floors (both already per-MW-IT ratios):
    #   power ceiling  = battery_capacity_headroom_multiplier * min_bat_p_mw_per_it_mw   (MW per MW IT)
    #   energy ceiling = battery_capacity_headroom_multiplier * min_bat_e_hours          (hours-equivalent per MW IT)
    # At the current default (2.0) this allows up to 0.20 MW of battery power
    # per MW of IT load (20% of IT) and 1.0 h of energy per MW of IT.
    #
    # Literature grounding for this multiplier:
    #  - NREL's Annual Technology Baseline (Cost Projections for Utility-Scale
    #    Battery Storage, 2024/2025 updates) treats 4-hour duration Li-ion as
    #    the standard grid-scale configuration; >90% of newly installed
    #    grid-scale capacity is 4 h or shorter.
    #  - Real data-center-paired batteries deployed/announced in 2025-2026
    #    (e.g. Aligned Data Centers' 31 MW / 62 MWh system) use ~2-hour
    #    duration, sized as a large-but-partial fraction of campus IT load.
    #  - This project's own "baseline" (non-optimized incumbent) design case
    #    uses baseline_bat_p_mw_per_it_mw=0.20 (20% of IT MW) at
    #    baseline_bat_e_hours=0.8 h -- i.e. roughly a 4-hour-duration battery
    #    sized at 20% of IT load.
    # A multiplier of 2.0 (20% of IT MW power ceiling) matches the project's
    # own baseline power assumption and the low end of real deployments
    # above. Higher multipliers remain available for diagnostic sweeps, but
    # any manuscript claim should use outputs regenerated from the current
    # source tree. Battery-ceiling sensitivity is intentionally excluded from
    # the current manuscript sensitivity set.
    battery_capacity_headroom_multiplier: float = 2.0

    eta_ch: float = 0.95
    eta_dis: float = 0.95
    battery_flex_reserve_duration_h: float = 1.0
    battery_ups_reserve_fraction: float = 0.10

    sipue_clip_nonnegative: bool = True
    sipue_system_ref_price_eur_per_mwh: float = 120.0
    auto_calibrate_sipue_reference_price: bool = True
    sipue_reference_price_perturbation_mw: float = 1.0
    sipue_reference_price_multiplier: float = 1.0

    p_dh_off_week: float = 0.02
    p_low_flex_week: float = 0.08
    p_low_heat_week: float = 0.05

    enable_time_varying_overhead: bool = True
    nordic_night_overhead_drop: float = 0.20
    day_overhead_bump: float = 0.07

    cop_ref: float = 3.2
    t_dh_supply_ref_c: float = 70.0
    service_parasitic_heat_frac: float = 0.03
    service_parasitic_flex_frac: float = 0.015
    service_parasitic_hx_mw_per_mwth: float = 0.0
    service_parasitic_flex_mw_per_mw: float = 0.0
    it_flex_scale_services_active: float = 1.2

    network_nodes: Tuple[str, ...] = ("N1", "N2", "N3")
    line_cap_12: float = 2500.0
    line_cap_23: float = 2000.0
    line_cap_13: float = 1800.0
    line_susceptance_12_mw_per_rad: float = 100.0
    line_susceptance_23_mw_per_rad: float = 100.0
    line_susceptance_13_mw_per_rad: float = 100.0
    auto_calibrate_line_limits: bool = True
    congestion_target_fraction: float = 0.10
    congestion_target_min_fraction: float = 0.05
    congestion_target_max_fraction: float = 0.15
    line_binding_tolerance: float = 1e-5
    line_capacity_multiplier: float = 1.0
    calibrated_no_dc_congestion_fraction: Optional[float] = None
    calibrated_no_dc_load_shedding_mwh: Optional[float] = None
    congestion_relief_perturbation_frac: float = 0.0001
    node_assignment: Optional[Dict[str, str]] = None
    node_cost_base: Optional[Dict[str, float]] = None
    node_cost_amp: Optional[Dict[str, float]] = None
    node_load_base: Optional[Dict[str, float]] = None
    node_load_amp: Optional[Dict[str, float]] = None

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
    independent_avoided_heat_cost_eur_per_mwhth: float = 55.0
    independent_heat_boiler_efficiency: float = 0.90
    independent_heat_boiler_emissions_tco2_per_mwh_fuel: float = 0.20
    independent_heat_boiler_vom_eur_per_mwhth: float = 3.0
    independent_avoided_flex_cost_eur_per_mw_h: float = 25.3
    independent_social_carbon_cost_eur_per_tco2: float = 120.0
    independent_peak_capacity_value_eur_per_mw_year: float = 50000.0

    heat_pipe_cap_share: float = 1.45
    heat_pipe_loss_frac: float = 0.05

    dh_segments: int = 3
    dh_seg_fracs: Tuple[float, ...] = (0.40, 0.35, 0.25)
    dh_value_multipliers: Tuple[float, ...] = (1.20, 1.05, 0.90)

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

    base_land_m2: float = 200.0  # Compatibility fallback when IT capacity is unavailable.
    existing_land_m2_per_it_mw: float = 8.0
    land_a_cooling: float = 5.0
    land_a_hx: float = 30.0
    land_a_hp: float = 12.0
    land_a_flex: float = 15.0
    land_a_bat_e: float = 5.0
    land_a_bat_p: float = 5.0
    land_cost_eur_per_m2: float = 35.0
    land_cost_is_annual: bool = True
    days_per_year: int = 365
    land_maintenance_frac: float = 0.0
    land_limit_m2: Optional[float] = None
    land_limit_multiplier: float = 1.0
    land_binding_tolerance_m2: float = 1.0
    construction_delay_frac: float = 0.0  # Deprecated compatibility input.
    construction_delay_months: float = 0.0
    override_construction_delay_months: Optional[float] = None
    lifecycle_years: int = 15
    real_discount_rate: float = 0.03
    enforce_carbon_cap: bool = True
    carbon_cap_reduction_frac: float = 0.05
    carbon_emissions_cap_tco2_by_sid: Optional[Dict[int, float]] = None
    override_heat_recovery_frac: Optional[float] = None
    override_hp_cop: Optional[float] = None

    ramp_k_hx: float = 0.0
    ramp_k_hp: float = 0.0
    ramp_k_flex: float = 0.0
    ramp_k_bat_e: float = 0.0
    ramp_k_bat_p: float = 0.0

    export_csv: bool = True
    canonical_output_dir: str = "output/congested3"
    export_plots: bool = True
    plot_output_dir: str = "output/congested3/figures"
    plot_format: str = "pdf"
    plot_extra_formats: Tuple[str, ...] = ()
    validation_output_dir: str = "output/validation"
    # run_train_test_scenario_validation trains on uncertainty_id <= this
    # value and evaluates on the rest, per archetype.
    train_test_max_train_id: int = 7
    out_of_sample_case_count: int = 200

    baseline_hx_mwth_per_it_mw: float = 0.25
    baseline_flex_mw_per_it_mw: float = 0.08
    baseline_bat_e_hours: float = 0.8
    baseline_bat_p_mw_per_it_mw: float = 0.20
    baseline_eff_reduction: float = 0.15

    min_hx_mwth_per_it_mw: float = 0.11
    min_flex_mw_per_it_mw: float = 0.09
    min_bat_e_hours: float = 0.5
    min_bat_p_mw_per_it_mw: float = 0.10

    co2_timeseries_scale: float = 1.0
    carbon_damage_multiplier: float = 1.0
    heat_system_value_multiplier: float = 1.0
    heat_demand_multiplier: float = 1.0
    flex_system_value_multiplier: float = 1.0
    flex_availability_multiplier: float = 1.0
    cooling_overhead_multiplier: float = 1.0

    use_representative_days_annual: bool = True
