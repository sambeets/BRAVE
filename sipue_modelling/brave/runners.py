from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from brave.config import Config, _validate_locked_calibration
from brave.metrics import compute_sipue, service_revenue_scale
from brave.network import calibrate_no_dc_network, load_network_timeseries, solve_no_dc_network
from brave.optimization import (
    baseline_design_for_scenario,
    solve_minmax_regret,
    solve_ops_given_design,
    solve_perfect_foresight,
)
from brave.provenance import export_parameter_provenance
from brave.literature import export_literature_matrix
from brave.scenario import (
    DATA_CENTER_ARCHETYPES,
    Scenario,
    apply_global_heat_assumptions,
    make_out_of_sample_scenarios,
    make_study_scenarios,
)
from brave.timeseries import (
    apply_services_mode,
    build_representative_days,
    make_hourly_profiles,
    make_hourly_profiles_from_indices,
)
from brave.validation import build_validation_summary


DESIGN_CASES = (
    "no_services",
    "fair_services_disabled",
    "baseline",
    "robust",
)

REFERENCE_CASE = "fair_services_disabled"


def _no_services_profile(prof: Dict[str, List[float]]) -> Dict[str, List[float]]:
    result = dict(prof)
    result["allow_heat"] = False
    result["allow_flex"] = False
    result["allow_battery"] = False
    result["force_it_flex"] = False
    return result


def _train_test_split(scenarios: Sequence[Scenario], max_train_uncertainty_id: int) -> List[List[Scenario]]:
    """One fold: train on uncertainty_id <= max_train_uncertainty_id, test on the rest."""
    test = [sc for sc in scenarios if sc.uncertainty_id > max_train_uncertainty_id]
    train_count = sum(1 for sc in scenarios if sc.uncertainty_id <= max_train_uncertainty_id)
    if not test or not train_count:
        raise ValueError(
            f"Train/test split at uncertainty_id={max_train_uncertainty_id} "
            f"produced an empty train or test set."
        )
    return [test]


def _prepare_profiles(
    cfg: Config,
    scenarios: Sequence[Scenario],
) -> Tuple[Config, Dict[int, Dict[str, List[float]]]]:
    def build_profile(sc: Scenario, builder) -> Dict[str, List[float]]:
        shared_state = sc.uncertainty_id if sc.uncertainty_id > 0 else sc.sid
        random.seed(cfg.seed + 10_009 * shared_state)
        return builder(sc)

    if cfg.T == 8760:
        if not cfg.use_representative_days_annual:
            raise ValueError("Annual runs require representative-day compression.")
        ts_full = load_network_timeseries(cfg, T=8760)
        indices, weights, starts, _ = build_representative_days(ts_full, cfg)
        cfg_local = replace(cfg, T=len(indices))
        base = {
            sc.sid: build_profile(
                sc,
                lambda item: make_hourly_profiles_from_indices(item, cfg_local, indices, weights, starts),
            )
            for sc in scenarios
        }
    else:
        cfg_local = cfg
        base = {
            sc.sid: build_profile(sc, lambda item: make_hourly_profiles(item, cfg.T, cfg))
            for sc in scenarios
        }
    if cfg.auto_calibrate_line_limits or cfg.auto_calibrate_sipue_reference_price:
        calibration = calibrate_no_dc_network(base[scenarios[0].sid], cfg_local)
        cfg_local = replace(
            cfg_local,
            line_cap_12=(calibration["line_cap_12_mw"] if cfg.auto_calibrate_line_limits else cfg_local.line_cap_12) * cfg.line_capacity_multiplier,
            line_cap_23=(calibration["line_cap_23_mw"] if cfg.auto_calibrate_line_limits else cfg_local.line_cap_23) * cfg.line_capacity_multiplier,
            line_cap_13=(calibration["line_cap_13_mw"] if cfg.auto_calibrate_line_limits else cfg_local.line_cap_13) * cfg.line_capacity_multiplier,
            sipue_system_ref_price_eur_per_mwh=(
                calibration["reference_price_eur_per_mwh"] * cfg.sipue_reference_price_multiplier
                if cfg.auto_calibrate_sipue_reference_price
                else cfg_local.sipue_system_ref_price_eur_per_mwh
            ),
            calibrated_no_dc_congestion_fraction=calibration["congestion_fraction"],
            calibrated_no_dc_load_shedding_mwh=calibration["load_shedding_mwh"],
        )
    return cfg_local, {
        sc.sid: apply_services_mode(sc, base[sc.sid], cfg_local) for sc in scenarios
    }


def _solve_design_ops(
    sc: Scenario,
    prof: Dict[str, List[float]],
    cfg: Config,
    design: Dict,
    objective_mode: str = "cost",
    no_services_independent_cost_eur: Optional[float] = None,
    availability: float = 1.0,
) -> Dict:
    realized = _realize_design(design, sc)
    return solve_ops_given_design(
        sc,
        prof,
        cfg,
        realized["K_HX"],
        realized["K_FLEX"],
        realized["K_BAT_E"],
        realized["K_BAT_P"],
        eff_reduction=design["eff_reduction"],
        K_HP=realized["K_HP"],
        objective_mode=objective_mode,
        no_services_independent_cost_eur=no_services_independent_cost_eur,
        availability=availability,
    )


def _realize_design(design: Dict, sc: Scenario) -> Dict[str, float]:
    if "K_HX_per_MW_IT" not in design:
        return {name: float(design[name]) for name in ("K_HX", "K_HP", "K_FLEX", "K_BAT_E", "K_BAT_P")}
    return {
        "K_HX": float(design["K_HX_per_MW_IT"]) * sc.base_it_mw,
        "K_HP": float(design["K_HP_per_MW_IT"]) * sc.base_it_mw,
        "K_FLEX": float(design["K_FLEX_per_MW_IT"]) * sc.base_it_mw,
        "K_BAT_E": float(design["K_BAT_E_per_MW_IT"]) * sc.base_it_mw,
        "K_BAT_P": float(design["K_BAT_P_per_MW_IT"]) * sc.base_it_mw,
    }


def _design_technical_feasibility(design: Dict, sc: Scenario, cfg: Config) -> Tuple[bool, str]:
    realized = _realize_design(design, sc)
    checks = {
        "heat exchanger": realized["K_HX"] + 1e-6 >= cfg.min_hx_mwth_per_it_mw * sc.base_it_mw,
        "heat pump": realized["K_HP"] + 1e-6 >= (
            cfg.min_hx_mwth_per_it_mw
            * sc.base_it_mw
            * (1.0 / max(1e-9, max(1.000001, sc.hp_cop) - 1.0))
        ),
        "flexibility": realized["K_FLEX"] + 1e-6 >= cfg.min_flex_mw_per_it_mw * sc.base_it_mw,
        "battery energy": realized["K_BAT_E"] + 1e-6 >= cfg.min_bat_e_hours * sc.base_it_mw,
        "battery power": realized["K_BAT_P"] + 1e-6 >= cfg.min_bat_p_mw_per_it_mw * sc.base_it_mw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, ", ".join(failures)


def _baseline_ops(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Dict:
    k_hx, k_hp, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(sc, prof, cfg)
    return solve_ops_given_design(
        sc, prof, cfg,
        k_hx, k_flex, k_bat_e, k_bat_p, eff_reduction=eff, K_HP=k_hp,
    )


def _no_services_ops(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Dict:
    # No-services is the zero-capacity, zero-efficiency-investment counterfactual.
    return solve_ops_given_design(
        sc,
        _no_services_profile(prof),
        replace(cfg, enforce_carbon_cap=False, carbon_emissions_cap_tco2_by_sid=None),
        0.0, 0.0, 0.0, 0.0, eff_reduction=0.0, K_HP=0.0,
    )


def _fair_services_disabled_ops(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Dict:
    eff = 0.0 if cfg.fixed_eff_reduction is None else float(cfg.fixed_eff_reduction)
    return solve_ops_given_design(
        sc,
        _no_services_profile(prof),
        cfg,
        0.0, 0.0, 0.0, 0.0, eff_reduction=eff, K_HP=0.0,
    )


def _carbon_reference(
    scenarios: Sequence[Scenario],
    profiles: Dict[int, Dict[str, List[float]]],
    cfg: Config,
) -> Tuple[Dict[int, Dict], Dict[int, float]]:
    no_services = {
        sc.sid: _no_services_ops(sc, profiles[sc.sid], cfg) for sc in scenarios
    }
    unconstrained_cfg = replace(cfg, enforce_carbon_cap=False, carbon_emissions_cap_tco2_by_sid=None)
    reduction = max(0.0, float(cfg.carbon_cap_reduction_frac))
    caps = {}
    for sc in scenarios:
        line_caps = (
            cfg.line_cap_12 * sc.congestion_multiplier,
            cfg.line_cap_23 * sc.congestion_multiplier,
            cfg.line_cap_13 * sc.congestion_multiplier,
        )
        no_dc = solve_no_dc_network(profiles[sc.sid], cfg, line_caps)
        no_dc_power_emissions = float(no_dc["emissions_tco2"])
        background = (
            no_dc_power_emissions
            + float(no_services[sc.sid]["baseline_heat_emissions_tco2"])
        )
        # Anchor the cap to the fair comparator; true_min_emissions is the feasibility floor.
        fair_unconstrained = float(
            _fair_services_disabled_ops(sc, profiles[sc.sid], unconstrained_cfg)["emissions_tco2"]
        )
        eff = 0.0 if cfg.fixed_eff_reduction is None else float(cfg.fixed_eff_reduction)
        true_min_emissions = float(
            solve_ops_given_design(
                sc, profiles[sc.sid], unconstrained_cfg,
                0.0, 0.0, 0.0, 0.0, eff_reduction=eff, K_HP=0.0,
                objective_mode="emissions",
            )["emissions_tco2"]
        )
        attributable = max(0.0, fair_unconstrained - background)
        policy_cap = background + attributable * (1.0 - reduction)
        caps[sc.sid] = max(policy_cap, true_min_emissions)
        no_services[sc.sid]["background_emissions_tco2"] = background
        no_services[sc.sid]["no_dc_power_emissions_tco2"] = no_dc_power_emissions
        no_services[sc.sid]["dc_attributable_emissions_tco2"] = attributable
        no_services[sc.sid]["unrelaxed_carbon_cap_tco2"] = policy_cap
        no_services[sc.sid]["fair_services_disabled_emissions_floor_tco2"] = fair_unconstrained
        no_services[sc.sid]["true_min_emissions_floor_tco2"] = true_min_emissions
        no_services[sc.sid]["carbon_cap_feasibility_floor_applied"] = caps[sc.sid] > policy_cap + 1e-6
        no_services[sc.sid]["carbon_cap_slack_tco2"] = caps[sc.sid] - true_min_emissions
    return no_services, caps


def _perfect_designs(
    scenarios: Sequence[Scenario],
    profiles: Dict[int, Dict[str, List[float]]],
    cfg: Config,
) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    objectives: Dict[int, float] = {}
    designs: Dict[int, Dict] = {}
    for sc in scenarios:
        objective, design = solve_perfect_foresight(sc, profiles[sc.sid], cfg)
        if cfg.validate_solutions and design.get("status") != "Optimal":
            raise RuntimeError(f"Perfect-foresight solve failed for scenario {sc.sid}.")
        objectives[sc.sid] = float(objective)
        designs[sc.sid] = design
    return objectives, designs


def _apply_sipue(ops: Dict, reference: Dict, cfg: Config, availability: float) -> None:
    raw_benefit = float(reference["independent_system_cost_eur"]) - float(ops["independent_system_cost_eur"])
    carbon_benefit = (
        float(reference["independent_carbon_damage_eur"])
        - float(ops["independent_carbon_damage_eur"])
    )
    no_carbon_benefit = raw_benefit - carbon_benefit
    delay_sipue, credit = compute_sipue(
        E_FAC=float(ops["E_FAC"]),
        E_IT=float(ops["E_IT"]),
        net_system_benefit_eur=raw_benefit,
        reference_price_eur_per_mwh=cfg.sipue_system_ref_price_eur_per_mwh,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
        benefit_availability_factor=availability,
    )
    steady_sipue, steady_credit = compute_sipue(
        E_FAC=float(ops["E_FAC"]),
        E_IT=float(ops["E_IT"]),
        net_system_benefit_eur=raw_benefit,
        reference_price_eur_per_mwh=cfg.sipue_system_ref_price_eur_per_mwh,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
        benefit_availability_factor=1.0,
    )
    no_carbon_sipue, no_carbon_credit = compute_sipue(
        E_FAC=float(ops["E_FAC"]),
        E_IT=float(ops["E_IT"]),
        net_system_benefit_eur=no_carbon_benefit,
        reference_price_eur_per_mwh=cfg.sipue_system_ref_price_eur_per_mwh,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
        benefit_availability_factor=availability,
    )
    carbon_only_sipue, carbon_only_credit = compute_sipue(
        E_FAC=float(ops["E_FAC"]),
        E_IT=float(ops["E_IT"]),
        net_system_benefit_eur=carbon_benefit,
        reference_price_eur_per_mwh=cfg.sipue_system_ref_price_eur_per_mwh,
        clip_nonnegative=cfg.sipue_clip_nonnegative,
        benefit_availability_factor=availability,
    )
    ops["siPUE"] = delay_sipue
    ops["steady_state_sipue"] = steady_sipue
    ops["siPUE_no_carbon_damage"] = no_carbon_sipue
    ops["siPUE_carbon_only"] = carbon_only_sipue
    ops["system_credit_mwh"] = credit
    ops["steady_state_system_credit_mwh"] = steady_credit
    ops["system_credit_no_carbon_damage_mwh"] = no_carbon_credit
    ops["system_credit_carbon_only_mwh"] = carbon_only_credit
    ops["raw_system_benefit_eur"] = raw_benefit
    ops["raw_system_benefit_no_carbon_damage_eur"] = no_carbon_benefit
    ops["raw_system_benefit_carbon_only_eur"] = carbon_benefit
    ops["delay_adjusted_system_benefit_eur"] = availability * raw_benefit
    ops["delay_adjusted_system_benefit_no_carbon_damage_eur"] = availability * no_carbon_benefit
    ops["delay_adjusted_system_benefit_carbon_only_eur"] = availability * carbon_benefit
    ops["sipue_credit_loss_mwh"] = steady_credit - credit


def _apply_emissions_attribution(ops: Dict, reference: Dict) -> None:
    background = float(reference["background_emissions_tco2"])
    ops["background_emissions_tco2"] = background
    ops["dc_attributable_emissions_tco2"] = float(ops["emissions_tco2"]) - background
    ops["CUE_tCO2_per_MWh_IT"] = ops["dc_attributable_emissions_tco2"] / max(1e-9, float(ops["E_IT"]))
    ops["dc_power_emissions_tco2"] = float(ops["power_emissions_tco2"]) - float(reference["no_dc_power_emissions_tco2"])


def _rebase_lifecycle(ops: Dict, reference: Dict, cfg: Config, is_reference: bool = False) -> None:
    """Keep the pre-commissioning system on the no-services trajectory."""
    full_factor = float(reference["lifecycle_operating_factor"]) / max(
        1e-9, float(reference["lifecycle_availability_factor"])
    )
    operating_factor = full_factor if is_reference else float(ops["lifecycle_operating_factor"])
    reference_annual = float(reference["annual_market_operating_cost_eur"])
    case_annual = float(ops["annual_market_operating_cost_eur"])
    operating_npv = (
        full_factor * reference_annual
        + operating_factor * (case_annual - reference_annual)
    )
    total_npv = operating_npv + float(ops["capex_npv_eur"]) + float(ops["land_cost_npv_eur"])
    zero_delay_npv = (
        full_factor * case_annual
        + float(ops["capex_npv_eur"])
        + float(ops["land_cost_npv_eur"])
    )
    if is_reference:
        ops["lifecycle_operating_factor"] = full_factor
        ops["lifecycle_availability_factor"] = 1.0
    ops["operating_cost_npv_eur"] = operating_npv
    ops["market_planning_cost_npv_eur"] = total_npv
    ops["market_planning_value_npv_eur"] = -total_npv
    ops["post_construction_market_cost_npv_eur"] = zero_delay_npv
    ops["lifecycle_value_loss_eur"] = total_npv - zero_delay_npv


def _case_fields(ops: Dict, suffix: str) -> Dict:
    fields = {
        "pue": "PUE",
        "sipue": "siPUE",
        "steady_sipue": "steady_state_sipue",
        "sipue_no_carbon_damage": "siPUE_no_carbon_damage",
        "sipue_carbon_only": "siPUE_carbon_only",
        "cue": "CUE_tCO2_per_MWh_IT",
        "e_it_mwh": "E_IT",
        "e_fac_mwh": "E_FAC",
        "cooling_mwh": "cooling_mwh",
        "recovered_heat_it_mwhth": "recovered_heat_it_mwhth",
        "recovered_heat_cooling_mwhth": "recovered_heat_cooling_mwhth",
        "heat_mwhth": "heat_MWhth",
        "heat_delivered_mwhth": "heat_MWhth",
        "flex_load_mwh": "flex_load_mwh",
        "flex_battery_mwh": "flex_bat_mwh",
        "battery_charge_mwh": "bat_charge_mwh",
        "battery_discharge_mwh": "bat_discharge_mwh",
        "storage_loss_mwh": "storage_loss_mwh",
        "market_cost_eur": "market_planning_cost_eur",
        "market_cost_npv_eur": "market_planning_cost_npv_eur",
        "market_value_npv_eur": "market_planning_value_npv_eur",
        "post_construction_market_cost_npv_eur": "post_construction_market_cost_npv_eur",
        "lifecycle_value_loss_eur": "lifecycle_value_loss_eur",
        "independent_cost_eur": "independent_system_cost_eur",
        "annual_independent_cost_eur": "independent_system_cost_eur",
        "resource_cost_eur": "el_cost_eur",
        "carbon_market_cost_eur": "co2_cost_eur",
        "carbon_damage_eur": "independent_carbon_damage_eur",
        "transmission_cost_eur": "trans_cost_eur",
        "grid_peak_cost_eur": "demand_charge_eur",
        "independent_peak_cost_eur": "independent_peak_capacity_cost_eur",
        "sla_cost_eur": "sla_penalty_eur",
        "avoided_heat_cost_eur": "independent_avoided_heat_cost_eur",
        "avoided_flex_cost_eur": "independent_avoided_flex_cost_eur",
        "heat_revenue_eur": "heat_revenue_eur",
        "flex_revenue_eur": "flex_revenue_eur",
        "capex_eur": "capex_eur",
        "land_cost_eur": "land_cost_eur",
        "land_cost_npv_eur": "land_cost_npv_eur",
        "land_m2": "land_m2",
        "existing_core_land_m2": "existing_core_land_m2",
        "cooling_land_m2": "existing_cooling_land_m2",
        "integration_land_m2": "integration_land_m2",
        "land_limit_m2": "land_limit_m2",
        "land_utilization_pct": "land_utilization_pct",
        "land_constraint_binding": "land_constraint_binding",
        "land_feasible": "land_feasible",
        "emissions_tco2": "emissions_tco2",
        "background_emissions_tco2": "background_emissions_tco2",
        "dc_attributed_emissions_tco2": "dc_attributable_emissions_tco2",
        "power_emissions_tco2": "power_emissions_tco2",
        "dc_power_emissions_tco2": "dc_power_emissions_tco2",
        "baseline_heat_emissions_tco2": "baseline_heat_emissions_tco2",
        "avoided_heat_emissions_tco2": "avoided_heat_emissions_tco2",
        "power_co2_cost_eur": "power_co2_cost_eur",
        "baseline_heat_co2_cost_eur": "baseline_heat_co2_cost_eur",
        "avoided_heat_co2_cost_eur": "avoided_heat_co2_cost_eur",
        "peak_import_mw": "peak_import_mw",
        "congestion_hours": "congestion_hours",
        "congestion_marginal_value_eur_per_1pct": "congestion_marginal_value_eur_per_1pct",
        "congestion_native_dual_max_eur_per_mw": "congestion_native_dual_max_eur_per_mw",
        "renewable_curtailment_mwh": "renewable_curtailment_mwh",
        "thermal_generation_mwh": "thermal_generation_mwh",
        "line_cap_12_mw": "line_cap_12_mw",
        "line_cap_23_mw": "line_cap_23_mw",
        "line_cap_13_mw": "line_cap_13_mw",
        "system_benefit_eur": "delay_adjusted_system_benefit_eur",
        "undelayed_system_benefit_eur": "raw_system_benefit_eur",
        "system_benefit_no_carbon_damage_eur": "delay_adjusted_system_benefit_no_carbon_damage_eur",
        "system_benefit_carbon_only_eur": "delay_adjusted_system_benefit_carbon_only_eur",
        "undelayed_system_benefit_no_carbon_damage_eur": "raw_system_benefit_no_carbon_damage_eur",
        "undelayed_system_benefit_carbon_only_eur": "raw_system_benefit_carbon_only_eur",
        "system_credit_mwh": "system_credit_mwh",
        "system_credit_no_carbon_damage_mwh": "system_credit_no_carbon_damage_mwh",
        "system_credit_carbon_only_mwh": "system_credit_carbon_only_mwh",
        "sipue_credit_loss_mwh": "sipue_credit_loss_mwh",
        "k_hx_mwth": "K_HX",
        "k_hp_mw": "K_HP",
        "k_flex_mw": "K_FLEX",
        "k_bat_e_mwh": "K_BAT_E",
        "k_bat_p_mw": "K_BAT_P",
        "eff_reduction": "eff_reduction",
    }
    result = {f"{name}_{suffix}": ops.get(key) for name, key in fields.items()}
    e_it = max(1e-9, float(ops["E_IT"]))
    result[f"system_benefit_per_mwh_it_{suffix}"] = float(ops.get("delay_adjusted_system_benefit_eur", 0.0)) / e_it
    result[f"system_benefit_no_carbon_damage_per_mwh_it_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_no_carbon_damage_eur", 0.0)) / e_it
    )
    result[f"system_benefit_carbon_only_per_mwh_it_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_carbon_only_eur", 0.0)) / e_it
    )
    result[f"annual_system_benefit_eur_{suffix}"] = float(ops.get("delay_adjusted_system_benefit_eur", 0.0))
    result[f"annual_system_benefit_eur_per_mwh_it_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_eur", 0.0)) / e_it
    )
    result[f"annual_system_benefit_no_carbon_damage_eur_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_no_carbon_damage_eur", 0.0))
    )
    result[f"annual_system_benefit_no_carbon_damage_eur_per_mwh_it_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_no_carbon_damage_eur", 0.0)) / e_it
    )
    result[f"annual_system_benefit_carbon_only_eur_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_carbon_only_eur", 0.0))
    )
    result[f"annual_system_benefit_carbon_only_eur_per_mwh_it_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_carbon_only_eur", 0.0)) / e_it
    )
    base_it_mw = max(1e-9, float(ops["base_it_mw"]))
    for name in ("K_HX", "K_HP", "K_FLEX", "K_BAT_E", "K_BAT_P"):
        result[f"{name}_per_MW_IT_{suffix}"] = float(ops[name]) / base_it_mw
    return result


def _reduction_pct(reference: float, case: float) -> float:
    return 100.0 * (reference - case) / max(1e-9, abs(reference))


def _change_pct(reference: float, case: float) -> float:
    return -_reduction_pct(reference, case)


def _change_direction(reference: float, case: float, tolerance: float = 1e-9) -> str:
    change = case - reference
    if abs(change) <= tolerance:
        return "unchanged"
    return "increase" if change > 0.0 else "decrease"


def _add_percentage_fields(row: Dict) -> None:
    ns_cost = float(row["market_cost_npv_eur_no_services"])
    fair_cost = float(row["market_cost_npv_eur_fair_services_disabled"])
    baseline_cost = float(row["market_cost_npv_eur_baseline"])
    ns_independent = float(row["independent_cost_eur_no_services"])
    fair_independent = float(row["independent_cost_eur_fair_services_disabled"])
    for design in DESIGN_CASES:
        cost = float(row[f"market_cost_npv_eur_{design}"])
        independent = float(row[f"independent_cost_eur_{design}"])
        emissions = float(row[f"emissions_tco2_{design}"])
        co2_cost = float(row[f"carbon_market_cost_eur_{design}"])
        damage = float(row[f"carbon_damage_eur_{design}"])
        pue = float(row[f"pue_{design}"])
        sipue = float(row[f"sipue_{design}"])
        row[f"lifecycle_cost_reduction_eur_vs_no_services_{design}"] = ns_cost - cost
        row[f"lifecycle_cost_reduction_pct_vs_no_services_{design}"] = _reduction_pct(ns_cost, cost)
        row[f"lifecycle_cost_reduction_eur_vs_fair_services_disabled_{design}"] = fair_cost - cost
        row[f"lifecycle_cost_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(fair_cost, cost)
        row[f"lifecycle_cost_reduction_eur_vs_baseline_{design}"] = baseline_cost - cost
        row[f"lifecycle_cost_reduction_pct_vs_baseline_{design}"] = _reduction_pct(baseline_cost, cost)
        row[f"independent_cost_reduction_pct_vs_no_services_{design}"] = _reduction_pct(ns_independent, independent)
        row[f"independent_cost_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            fair_independent, independent
        )
        row[f"emissions_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["emissions_tco2_no_services"]), emissions
        )
        row[f"emissions_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["emissions_tco2_fair_services_disabled"]), emissions
        )
        row[f"dc_attributed_emissions_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["dc_attributed_emissions_tco2_no_services"]),
            float(row[f"dc_attributed_emissions_tco2_{design}"]),
        )
        row[f"dc_attributed_emissions_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["dc_attributed_emissions_tco2_fair_services_disabled"]),
            float(row[f"dc_attributed_emissions_tco2_{design}"]),
        )
        row[f"co2_cost_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["carbon_market_cost_eur_no_services"]), co2_cost
        )
        row[f"co2_cost_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["carbon_market_cost_eur_fair_services_disabled"]), co2_cost
        )
        row[f"carbon_damage_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["carbon_damage_eur_no_services"]), damage
        )
        row[f"carbon_damage_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["carbon_damage_eur_fair_services_disabled"]), damage
        )
        row[f"pue_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["pue_no_services"]), pue
        )
        row[f"pue_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["pue_fair_services_disabled"]), pue
        )
        row[f"sipue_reduction_pct_vs_no_services_{design}"] = _reduction_pct(
            float(row["sipue_no_services"]), sipue
        )
        row[f"sipue_reduction_pct_vs_fair_services_disabled_{design}"] = _reduction_pct(
            float(row["sipue_fair_services_disabled"]), sipue
        )
        row[f"land_area_change_pct_vs_no_services_{design}"] = -_reduction_pct(
            float(row["land_m2_no_services"]), float(row[f"land_m2_{design}"])
        )
        row[f"land_area_change_pct_vs_fair_services_disabled_{design}"] = -_reduction_pct(
            float(row["land_m2_fair_services_disabled"]), float(row[f"land_m2_{design}"])
        )
        row[f"land_lease_cost_change_pct_vs_no_services_{design}"] = -_reduction_pct(
            float(row["land_cost_npv_eur_no_services"]), float(row[f"land_cost_npv_eur_{design}"])
        )
        row[f"land_lease_cost_change_pct_vs_fair_services_disabled_{design}"] = -_reduction_pct(
            float(row["land_cost_npv_eur_fair_services_disabled"]), float(row[f"land_cost_npv_eur_{design}"])
        )
        facility_burden = pue - float(row["pue_fair_services_disabled"])
        credit_per_it = float(row[f"system_credit_mwh_{design}"]) / max(1e-9, float(row[f"e_it_mwh_{design}"]))
        row[f"facility_burden_pue_vs_fair_services_disabled_{design}"] = facility_burden
        row[f"system_credit_mwh_per_mwh_it_{design}"] = credit_per_it
        row[f"benefit_to_burden_ratio_vs_fair_services_disabled_{design}"] = (
            credit_per_it / facility_burden if abs(facility_burden) > 1e-9 else None
        )
        steady_credit = float(row[f"system_credit_mwh_{design}"]) + float(row[f"sipue_credit_loss_mwh_{design}"])
        row[f"sipue_credit_loss_pct_{design}"] = (
            100.0 * float(row[f"sipue_credit_loss_mwh_{design}"]) / max(1e-9, abs(steady_credit))
        )
        zero_delay_gain = ns_cost - float(row[f"post_construction_market_cost_npv_eur_{design}"])
        row[f"delay_loss_pct_of_zero_delay_gain_{design}"] = (
            100.0 * float(row[f"lifecycle_value_loss_eur_{design}"]) / max(1e-9, abs(zero_delay_gain))
        )


def _add_land_status_fields(row: Dict, cfg: Config) -> None:
    for design in DESIGN_CASES:
        land = float(row[f"land_m2_{design}"])
        raw_limit = row.get(f"land_limit_m2_{design}")
        if raw_limit in (None, ""):
            row[f"land_feasible_{design}"] = True
            row[f"land_constraint_binding_{design}"] = False
            continue
        limit = float(raw_limit)
        row[f"land_feasible_{design}"] = land <= limit + cfg.validate_tol
        row[f"land_constraint_binding_{design}"] = (
            abs(limit - land) <= cfg.land_binding_tolerance_m2
        )


def _decomposition_rows(
    scenario: str,
    reference: Dict,
    case: Dict,
    suffix: str,
) -> List[Dict]:
    case_op = float(case["lifecycle_operating_factor"])
    availability = max(1e-9, float(case["lifecycle_availability_factor"]))
    full_op = case_op / availability
    ref_land = float(reference["land_cost_npv_eur"])
    case_land = float(case["land_cost_npv_eur"])
    components = {
        "resource electricity": full_op * (reference["resource_cost_other_eur"] - case["resource_cost_other_eur"]),
        "cooling electricity": full_op * (reference["cooling_resource_cost_eur"] - case["cooling_resource_cost_eur"]),
        "storage losses": full_op * (reference["storage_loss_resource_cost_eur"] - case["storage_loss_resource_cost_eur"]),
        "CO2 market": full_op * (reference["co2_cost_eur"] - case["co2_cost_eur"]),
        "transmission": full_op * (reference["trans_cost_eur"] - case["trans_cost_eur"]),
        "congestion": full_op * (reference["congestion_direct_cost_eur"] - case["congestion_direct_cost_eur"]),
        "peak capacity": full_op * (reference["demand_charge_eur"] - case["demand_charge_eur"]),
        "SLA": full_op * (reference["sla_penalty_eur"] - case["sla_penalty_eur"]),
        "heating revenue": full_op * (case["heat_revenue_eur"] - reference["heat_revenue_eur"]),
        "flexibility revenue": full_op * (case["flex_revenue_eur"] - reference["flex_revenue_eur"]),
        "efficiency": full_op * (case["eff_benefit_eur"] - reference["eff_benefit_eur"]),
        "capex": reference["capex_npv_eur"] - case["capex_npv_eur"],
        "land": ref_land - case_land,
    }
    components["construction delay"] = (
        case_op - full_op
    ) * (
        float(reference["annual_market_operating_cost_eur"])
        - float(case["annual_market_operating_cost_eur"])
    )
    total = sum(float(v) for v in components.values())
    expected = float(case["market_planning_value_npv_eur"]) - float(reference["market_planning_value_npv_eur"])
    e_it = max(1e-9, float(case["E_IT"]))

    def row(ledger: str, name: str, value: float, ledger_total: float) -> Dict:
        return {
            "scenario": scenario,
            "design": suffix,
            "ledger": ledger,
            "component": name,
            "value_eur": value,
            "value_per_mwh_it": value / e_it,
            "component_share_pct": 100.0 * value / ledger_total if abs(ledger_total) > 1e-9 else 0.0,
            "ledger_total_eur": ledger_total,
            "e_it_mwh": e_it,
        }

    rows = [row("market", name, value, expected) for name, value in components.items()]
    rows.append(row("market", "reconciliation residual", expected - total, expected))

    independent = {
        "resource electricity": reference["el_cost_eur"] - case["el_cost_eur"],
        "transmission": reference["trans_cost_eur"] - case["trans_cost_eur"],
        "peak capacity": reference["independent_peak_capacity_cost_eur"] - case["independent_peak_capacity_cost_eur"],
        "carbon damage": reference["independent_carbon_damage_eur"] - case["independent_carbon_damage_eur"],
        "heat system value": case["independent_avoided_heat_cost_eur"] - reference["independent_avoided_heat_cost_eur"],
        "flexibility system value": case["independent_avoided_flex_cost_eur"] - reference["independent_avoided_flex_cost_eur"],
    }
    independent_total = float(reference["independent_system_cost_eur"]) - float(case["independent_system_cost_eur"])
    rows.extend(row("independent", name, value, independent_total) for name, value in independent.items())
    rows.append(row(
        "independent",
        "reconciliation residual",
        independent_total - sum(float(value) for value in independent.values()),
        independent_total,
    ))
    return rows


def _summarize_decomposition(rows: Sequence[Dict]) -> List[Dict]:
    """Summarize decomposition rows per (archetype group, design, ledger, component).

    ``average_eur_per_mwh_it`` is the equal-weighted mean across rows in the
    group, kept for backward compatibility; for the ``archetype == "all"``
    group this gives every archetype the same one-quarter weight regardless
    of IT energy scale. ``average_eur_per_mwh_it_it_weighted`` instead
    weights each row by its own IT energy (a no-op within a single
    archetype's group, since IT energy is constant across that archetype's
    ten uncertainty states). The paper's credit-ablation table reports the
    IT-weighted values for the ``all`` group.
    """
    groups: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    for row in rows:
        for archetype in {"all", str(row.get("archetype", "all"))}:
            key = (archetype, row["design"], row["ledger"], row["component"])
            groups.setdefault(key, []).append(row)
    summary = []
    for (archetype, design, ledger, component), values in sorted(groups.items()):
        eur = [float(value["value_eur"]) for value in values]
        per_it = [float(value["value_per_mwh_it"]) for value in values]
        shares = [float(value["component_share_pct"]) for value in values]
        it_weights = [float(value["e_it_mwh"]) for value in values]
        summary.append({
            "archetype": archetype,
            "design": design,
            "ledger": ledger,
            "component": component,
            "scenario_count": len(values),
            "average_eur": sum(eur) / len(eur),
            "minimum_eur": min(eur),
            "maximum_eur": max(eur),
            "average_eur_per_mwh_it": sum(per_it) / len(per_it),
            "average_component_share_pct": sum(shares) / len(shares),
            "average_eur_per_mwh_it_it_weighted": _weighted_avg(per_it, it_weights),
        })
    return summary


def _summarize_design_comparison(rows: Sequence[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        for archetype in {"all", str(row.get("archetype", "all"))}:
            groups.setdefault((archetype, str(row["design"])), []).append(row)
    summary = []
    for (archetype, design), values in sorted(groups.items()):
        comparable = [value for value in values if value.get("regret_eur") not in (None, "")]
        regret = [float(value["regret_eur"]) for value in comparable]
        regret_per_it = [float(value["regret_per_mwh_it"]) for value in comparable]
        share = [
            float(value["regret_share_of_attainable_pct"])
            for value in comparable
            if value.get("regret_share_of_attainable_pct") not in (None, "")
        ]
        cost_share = [
            float(value["regret_share_of_lifecycle"])
            for value in comparable
            if value.get("regret_share_of_lifecycle") not in (None, "")
        ]
        lifecycle_value = [float(value["lifecycle_value_npv_eur"]) for value in values]
        summary.append({
            "archetype": archetype,
            "design": design,
            "scenario_count": len(values),
            "average_lifecycle_value_npv_eur": sum(lifecycle_value) / len(lifecycle_value),
            "comparable_scenario_count": len(comparable),
            "average_regret_eur": sum(regret) / len(regret) if regret else None,
            "worst_regret_eur": max(regret) if regret else None,
            "average_regret_per_mwh_it": sum(regret_per_it) / len(regret_per_it) if regret_per_it else None,
            "average_regret_share_of_attainable_pct": sum(share) / len(share) if share else None,
            "average_regret_share_of_lifecycle": sum(cost_share) / len(cost_share) if cost_share else None,
            "worst_regret_share_of_lifecycle": max(cost_share) if cost_share else None,
        })
    return summary


def _paper_metric_rows(rows: Sequence[Dict]) -> List[Dict]:
    metrics = {
        "PUE": ("pue", "dimensionless"),
        "siPUE": ("sipue", "dimensionless"),
        "siPUE excluding carbon damage": ("sipue_no_carbon_damage", "dimensionless"),
        "siPUE carbon damage only": ("sipue_carbon_only", "dimensionless"),
        "post-construction siPUE": ("steady_sipue", "dimensionless"),
        "CUE": ("cue", "tCO2/MWh IT"),
        "CO2 cost": ("carbon_market_cost_eur", "EUR"),
        "carbon damage": ("carbon_damage_eur", "EUR"),
        "heat revenue": ("heat_revenue_eur", "EUR"),
        "flexibility revenue": ("flex_revenue_eur", "EUR"),
        "emissions": ("emissions_tco2", "tCO2"),
        "annual system benefit": ("annual_system_benefit_eur", "annual EUR"),
        "annual benefit per MWh IT": ("annual_system_benefit_eur_per_mwh_it", "annual EUR/MWh IT"),
        "annual system benefit excluding carbon damage": ("annual_system_benefit_no_carbon_damage_eur", "annual EUR"),
        "annual benefit excluding carbon damage per MWh IT": (
            "annual_system_benefit_no_carbon_damage_eur_per_mwh_it",
            "annual EUR/MWh IT",
        ),
        "annual carbon-damage benefit": ("annual_system_benefit_carbon_only_eur", "annual EUR"),
        "annual carbon-damage benefit per MWh IT": (
            "annual_system_benefit_carbon_only_eur_per_mwh_it",
            "annual EUR/MWh IT",
        ),
        "lifecycle value": ("market_value_npv_eur", "EUR"),
        "lifecycle delay loss": ("lifecycle_value_loss_eur", "EUR"),
        "cooling electricity": ("cooling_mwh", "MWh"),
        "congestion hours": ("congestion_hours", "h"),
        "renewable curtailment": ("renewable_curtailment_mwh", "MWh"),
        "thermal generation": ("thermal_generation_mwh", "MWh"),
        "IT-source recovered heat": ("recovered_heat_it_mwhth", "MWhth"),
        "cooling-source recovered heat": ("recovered_heat_cooling_mwhth", "MWhth"),
        "total land": ("land_m2", "m2"),
        "integration land": ("integration_land_m2", "m2"),
        "land utilization": ("land_utilization_pct", "%"),
        "land lease NPV": ("land_cost_npv_eur", "EUR"),
        "lifecycle cost reduction vs no-services": ("lifecycle_cost_reduction_pct_vs_no_services", "%"),
        "lifecycle cost reduction vs fair services-disabled": ("lifecycle_cost_reduction_pct_vs_fair_services_disabled", "%"),
        "lifecycle cost reduction vs baseline": ("lifecycle_cost_reduction_pct_vs_baseline", "%"),
        "emissions reduction vs no-services": ("emissions_reduction_pct_vs_no_services", "%"),
        "emissions reduction vs fair services-disabled": ("emissions_reduction_pct_vs_fair_services_disabled", "%"),
        "CO2 cost reduction vs no-services": ("co2_cost_reduction_pct_vs_no_services", "%"),
        "CO2 cost reduction vs fair services-disabled": ("co2_cost_reduction_pct_vs_fair_services_disabled", "%"),
        "PUE reduction vs no-services": ("pue_reduction_pct_vs_no_services", "%"),
        "PUE reduction vs fair services-disabled": ("pue_reduction_pct_vs_fair_services_disabled", "%"),
        "siPUE reduction vs no-services": ("sipue_reduction_pct_vs_no_services", "%"),
        "siPUE reduction vs fair services-disabled": ("sipue_reduction_pct_vs_fair_services_disabled", "%"),
        "facility burden vs fair services-disabled": ("facility_burden_pue_vs_fair_services_disabled", "PUE units"),
        "system credit per MWh IT": ("system_credit_mwh_per_mwh_it", "MWh/MWh IT"),
        "benefit-to-burden ratio vs fair services-disabled": (
            "benefit_to_burden_ratio_vs_fair_services_disabled",
            "dimensionless",
        ),
        "delay loss share": ("delay_loss_pct_of_zero_delay_gain", "% of zero-delay gain"),
    }
    output = []
    for row in rows:
        for design in DESIGN_CASES:
            for label, (stem, unit) in metrics.items():
                output.append({
                    "run_id": row.get("run_id"),
                    "archetype": row.get("archetype"),
                    "uncertainty_state": row.get("scenario_id"),
                    "scenario": row["scenario"],
                    "design": design,
                    "metric": label,
                    "value": row[f"{stem}_{design}"],
                    "unit": unit,
                    "delay_factor": row["lifecycle_availability"],
                })
    return output


def _metric_change_pct(reference: float, case: float) -> float | None:
    if abs(reference) <= 1e-9:
        return None
    return 100.0 * (case - reference) / abs(reference)


def _kpi_value(row: Dict, design: str, stem: str) -> float:
    if stem == "modeled_erf":
        heat = float(row[f"heat_mwhth_{design}"])
        facility = max(1e-9, float(row[f"e_fac_mwh_{design}"]))
        return max(0.0, heat) / facility
    return float(row[f"{stem}_{design}"])


def _weighted_avg(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _kpi_benchmark_summary(rows: Sequence[Dict]) -> List[Dict]:
    """Compact KPI benchmark, one row per (archetype group, design case, metric).

    ``average_value`` and its siblings are the simple equal-weighted mean
    across rows in the group (each canonical row -- i.e. each uncertainty
    state -- counts once), preserved for backward compatibility with any
    existing consumer of this table. For the ``archetype == "all"`` group,
    which pools rows across all four archetypes, this equal-weighting gives
    every archetype the same one-quarter weight regardless of its IT energy
    scale (8-60 MW installed IT capacity across archetypes in this study).
    The ``*_it_weighted`` fields instead weight each row by that row's own
    IT energy for the given design case (``e_it_mwh_{design}``), so a larger
    archetype contributes proportionally more; within a single archetype's
    group this is a no-op (IT energy is constant across that archetype's ten
    uncertainty states), so it only changes the ``all`` group. The paper's
    main tables (KPI benchmark, sensitivity summary, credit ablation,
    reference-price sensitivity) report the IT-weighted values for the
    ``all`` group.
    """
    metrics = {
        "pue": ("pue", "dimensionless"),
        "sipue": ("sipue", "dimensionless"),
        "sipue_no_carbon_damage": ("sipue_no_carbon_damage", "dimensionless"),
        "sipue_carbon_only": ("sipue_carbon_only", "dimensionless"),
        "steady_sipue": ("steady_sipue", "dimensionless"),
        "cue": ("cue", "tCO2/MWh IT"),
        "modeled_erf": ("modeled_erf", "MWhth delivered useful heat/MWh facility electricity"),
        "system_benefit_per_mwh_it": ("system_benefit_per_mwh_it", "EUR/MWh IT"),
        "system_benefit_no_carbon_damage_per_mwh_it": (
            "system_benefit_no_carbon_damage_per_mwh_it",
            "EUR/MWh IT",
        ),
        "system_benefit_carbon_only_per_mwh_it": (
            "system_benefit_carbon_only_per_mwh_it",
            "EUR/MWh IT",
        ),
        "emissions_tco2": ("emissions_tco2", "tCO2"),
        "carbon_damage_eur": ("carbon_damage_eur", "EUR"),
    }

    groups: List[Tuple[str, str, Sequence[Dict]]] = [("all", "all", rows)]
    archetypes = sorted({
        (str(row["archetype"]), str(row["archetype_key"]))
        for row in rows
        if row.get("archetype") and row.get("archetype_key")
    })
    groups.extend(
        (name, key, [row for row in rows if row.get("archetype_key") == key])
        for name, key in archetypes
    )

    output: List[Dict] = []
    for archetype, archetype_key, values in groups:
        for design in DESIGN_CASES:
            it_weights = [float(row[f"e_it_mwh_{design}"]) for row in values]
            for metric, (stem, unit) in metrics.items():
                case_values = [_kpi_value(row, design, stem) for row in values]
                ref_values = [_kpi_value(row, "no_services", stem) for row in values]
                average = sum(case_values) / len(case_values)
                reference = sum(ref_values) / len(ref_values)
                average_wt = _weighted_avg(case_values, it_weights)
                reference_wt = _weighted_avg(ref_values, it_weights)
                output.append({
                    "run_id": values[0].get("run_id") if values else None,
                    "archetype": archetype,
                    "archetype_key": archetype_key,
                    "design_case": design,
                    "metric": metric,
                    "unit": unit,
                    "scenario_count": len(values),
                    "average_value": average,
                    "average_no_services_value": reference,
                    "average_absolute_change_vs_no_services": average - reference,
                    "average_pct_change_vs_no_services": _metric_change_pct(reference, average),
                    "average_value_it_weighted": average_wt,
                    "average_no_services_value_it_weighted": reference_wt,
                    "average_absolute_change_vs_no_services_it_weighted": average_wt - reference_wt,
                    "average_pct_change_vs_no_services_it_weighted": _metric_change_pct(reference_wt, average_wt),
                })
    return output


def _sipue_regret_summary(rows: Sequence[Dict]) -> List[Dict]:
    """Diagnostic-only comparison of the robust design against the
    scenario-perfect siPUE achievable with foresight of each scenario.

    Stage 2 minimizes worst-case *cost* regret (Eq. normalized-regret in the
    paper), never siPUE regret directly, so this quantifies how much siPUE
    headroom the cost-optimal shared design leaves on the table relative to
    a design chosen with perfect information about that one scenario. It is
    not a canonical design case.
    """
    groups: List[Tuple[str, str, Sequence[Dict]]] = [("all", "all", rows)]
    archetypes = sorted({
        (str(row["archetype"]), str(row["archetype_key"]))
        for row in rows
        if row.get("archetype") and row.get("archetype_key")
    })
    groups.extend(
        (name, key, [row for row in rows if row.get("archetype_key") == key])
        for name, key in archetypes
    )
    output: List[Dict] = []
    for archetype, archetype_key, values in groups:
        abs_regrets = [float(row["sipue_regret_robust_vs_scenario_perfect"]) for row in values]
        pct_regrets = [float(row["sipue_regret_pct_vs_scenario_perfect"]) for row in values]
        output.append({
            "run_id": values[0].get("run_id") if values else None,
            "archetype": archetype,
            "archetype_key": archetype_key,
            "scenario_count": len(values),
            "mean_sipue_regret": sum(abs_regrets) / len(abs_regrets),
            "min_sipue_regret": min(abs_regrets),
            "max_sipue_regret": max(abs_regrets),
            "mean_sipue_regret_pct": sum(pct_regrets) / len(pct_regrets),
            "max_sipue_regret_pct": max(pct_regrets),
        })
    return output


def _conclusion_summary_for_rows(
    rows: Sequence[Dict],
    comparison: Sequence[Dict],
    archetype_label: str,
) -> List[Dict]:
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    comparison_by_design: Dict[str, List[Dict]] = {}
    for item in comparison:
        comparison_by_design.setdefault(str(item["design"]), []).append(item)
    ns_cost = mean("market_cost_npv_eur_no_services")
    fair_cost = mean("market_cost_npv_eur_fair_services_disabled")
    baseline_cost = mean("market_cost_npv_eur_baseline")
    output = []
    for design in DESIGN_CASES:
        case_cost = mean(f"market_cost_npv_eur_{design}")
        items = comparison_by_design.get(design, [])
        comparable_regrets = [
            float(item["regret_eur"])
            for item in items if item.get("regret_eur") not in (None, "")
        ]
        attainable_shares = [
            float(item["regret_share_of_attainable_pct"])
            for item in items if item.get("regret_share_of_attainable_pct") not in (None, "")
        ]
        lifecycle_cost_shares = [
            float(item["regret_share_of_lifecycle"])
            for item in items if item.get("regret_share_of_lifecycle") not in (None, "")
        ]
        summary = {
            "archetype": archetype_label,
            "design": design,
            "scenario_count": len(rows),
            "reference_case": REFERENCE_CASE,
            "average_lifecycle_cost_npv_eur": case_cost,
            "average_lifecycle_cost_reduction_eur_vs_no_services": ns_cost - case_cost,
            "lifecycle_cost_reduction_pct_vs_no_services": _reduction_pct(ns_cost, case_cost),
            "average_scenario_lifecycle_cost_reduction_pct_vs_no_services": mean(f"lifecycle_cost_reduction_pct_vs_no_services_{design}"),
            "scenarios_with_lower_lifecycle_cost_vs_no_services": sum(
                float(row[f"lifecycle_cost_reduction_eur_vs_no_services_{design}"]) > 0.0 for row in rows
            ),
            "average_lifecycle_cost_reduction_eur_vs_fair_services_disabled": fair_cost - case_cost,
            "lifecycle_cost_reduction_pct_vs_fair_services_disabled": _reduction_pct(fair_cost, case_cost),
            "average_scenario_lifecycle_cost_reduction_pct_vs_fair_services_disabled": mean(
                f"lifecycle_cost_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "scenarios_with_lower_lifecycle_cost_vs_fair_services_disabled": sum(
                float(row[f"lifecycle_cost_reduction_eur_vs_fair_services_disabled_{design}"]) > 0.0
                for row in rows
            ),
            "average_lifecycle_cost_reduction_eur_vs_baseline": baseline_cost - case_cost,
            "lifecycle_cost_reduction_pct_vs_baseline": _reduction_pct(baseline_cost, case_cost),
            "average_scenario_lifecycle_cost_reduction_pct_vs_baseline": mean(f"lifecycle_cost_reduction_pct_vs_baseline_{design}"),
            "average_independent_cost_reduction_pct_vs_no_services": mean(f"independent_cost_reduction_pct_vs_no_services_{design}"),
            "average_independent_cost_reduction_pct_vs_fair_services_disabled": mean(
                f"independent_cost_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_emissions_reduction_pct_vs_no_services": mean(f"emissions_reduction_pct_vs_no_services_{design}"),
            "average_emissions_reduction_pct_vs_fair_services_disabled": mean(
                f"emissions_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_dc_attributed_emissions_reduction_pct_vs_no_services": mean(f"dc_attributed_emissions_reduction_pct_vs_no_services_{design}"),
            "average_dc_attributed_emissions_reduction_pct_vs_fair_services_disabled": mean(
                f"dc_attributed_emissions_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_co2_cost_reduction_pct_vs_no_services": mean(f"co2_cost_reduction_pct_vs_no_services_{design}"),
            "average_co2_cost_reduction_pct_vs_fair_services_disabled": mean(
                f"co2_cost_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_carbon_damage_reduction_pct_vs_no_services": mean(f"carbon_damage_reduction_pct_vs_no_services_{design}"),
            "average_carbon_damage_reduction_pct_vs_fair_services_disabled": mean(
                f"carbon_damage_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_pue_reduction_pct_vs_no_services": mean(f"pue_reduction_pct_vs_no_services_{design}"),
            "average_pue_reduction_pct_vs_fair_services_disabled": mean(
                f"pue_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "average_sipue_reduction_pct_vs_no_services": mean(f"sipue_reduction_pct_vs_no_services_{design}"),
            "average_sipue_reduction_pct_vs_fair_services_disabled": mean(
                f"sipue_reduction_pct_vs_fair_services_disabled_{design}"
            ),
            "scenarios_with_lower_sipue_vs_fair_services_disabled": sum(
                float(row[f"sipue_{design}"]) < float(row["sipue_fair_services_disabled"])
                for row in rows
            ),
            "average_facility_burden_pue_vs_fair_services_disabled": mean(
                f"facility_burden_pue_vs_fair_services_disabled_{design}"
            ),
            "average_system_credit_mwh_per_mwh_it": mean(f"system_credit_mwh_per_mwh_it_{design}"),
            "average_benefit_to_burden_ratio_vs_fair_services_disabled": (
                sum(
                    float(row[f"benefit_to_burden_ratio_vs_fair_services_disabled_{design}"])
                    for row in rows
                    if row[f"benefit_to_burden_ratio_vs_fair_services_disabled_{design}"] not in (None, "")
                )
                / max(
                    1,
                    sum(
                        row[f"benefit_to_burden_ratio_vs_fair_services_disabled_{design}"] not in (None, "")
                        for row in rows
                    ),
                )
            ),
            "average_delay_loss_eur": mean(f"lifecycle_value_loss_eur_{design}"),
            "average_delay_loss_pct_of_zero_delay_gain": mean(f"delay_loss_pct_of_zero_delay_gain_{design}"),
            "average_sipue_credit_loss_pct": mean(f"sipue_credit_loss_pct_{design}"),
            "average_land_m2": mean(f"land_m2_{design}"),
            "average_integration_land_m2": mean(f"integration_land_m2_{design}"),
            "average_land_utilization_pct": mean(f"land_utilization_pct_{design}"),
            "binding_land_scenario_count": sum(
                str(row[f"land_constraint_binding_{design}"]).lower() == "true" for row in rows
            ),
            "average_land_lease_npv_eur": mean(f"land_cost_npv_eur_{design}"),
            "average_incremental_land_lease_npv_eur_vs_no_services": (
                mean(f"land_cost_npv_eur_{design}") - mean("land_cost_npv_eur_no_services")
            ),
            "incremental_land_lease_share_of_lifecycle_gain_pct": (
                100.0
                * (mean(f"land_cost_npv_eur_{design}") - mean("land_cost_npv_eur_no_services"))
                / max(1e-9, abs(ns_cost - case_cost))
            ),
            "average_regret_eur": (
                sum(comparable_regrets) / len(comparable_regrets) if comparable_regrets else None
            ),
            "worst_regret_eur": max(comparable_regrets) if comparable_regrets else None,
            "average_regret_share_of_attainable_pct": (
                sum(attainable_shares) / len(attainable_shares) if attainable_shares else None
            ),
            "average_regret_share_of_lifecycle": (
                sum(lifecycle_cost_shares) / len(lifecycle_cost_shares) if lifecycle_cost_shares else None
            ),
            "worst_regret_share_of_lifecycle": (
                max(lifecycle_cost_shares) if lifecycle_cost_shares else None
            ),
        }
        directional_metrics = {
            "lifecycle_cost": (ns_cost, case_cost),
            "independent_cost": (
                mean("independent_cost_eur_no_services"),
                mean(f"independent_cost_eur_{design}"),
            ),
            "emissions": (mean("emissions_tco2_no_services"), mean(f"emissions_tco2_{design}")),
            "co2_cost": (
                mean("carbon_market_cost_eur_no_services"),
                mean(f"carbon_market_cost_eur_{design}"),
            ),
            "carbon_damage": (
                mean("carbon_damage_eur_no_services"),
                mean(f"carbon_damage_eur_{design}"),
            ),
            "pue": (mean("pue_no_services"), mean(f"pue_{design}")),
            "sipue": (mean("sipue_no_services"), mean(f"sipue_{design}")),
            "land_area": (mean("land_m2_no_services"), mean(f"land_m2_{design}")),
            "land_lease_cost": (
                mean("land_cost_npv_eur_no_services"),
                mean(f"land_cost_npv_eur_{design}"),
            ),
        }
        for label, (reference, case) in directional_metrics.items():
            summary[f"{label}_change_pct_vs_no_services"] = -_reduction_pct(reference, case)
            summary[f"{label}_direction_vs_no_services"] = _change_direction(reference, case)
        fair_directional_metrics = {
            "lifecycle_cost": (fair_cost, case_cost),
            "independent_cost": (
                mean("independent_cost_eur_fair_services_disabled"),
                mean(f"independent_cost_eur_{design}"),
            ),
            "emissions": (
                mean("emissions_tco2_fair_services_disabled"),
                mean(f"emissions_tco2_{design}"),
            ),
            "co2_cost": (
                mean("carbon_market_cost_eur_fair_services_disabled"),
                mean(f"carbon_market_cost_eur_{design}"),
            ),
            "carbon_damage": (
                mean("carbon_damage_eur_fair_services_disabled"),
                mean(f"carbon_damage_eur_{design}"),
            ),
            "pue": (mean("pue_fair_services_disabled"), mean(f"pue_{design}")),
            "sipue": (mean("sipue_fair_services_disabled"), mean(f"sipue_{design}")),
            "land_area": (
                mean("land_m2_fair_services_disabled"),
                mean(f"land_m2_{design}"),
            ),
            "land_lease_cost": (
                mean("land_cost_npv_eur_fair_services_disabled"),
                mean(f"land_cost_npv_eur_{design}"),
            ),
        }
        for label, (reference, case) in fair_directional_metrics.items():
            summary[f"{label}_change_pct_vs_fair_services_disabled"] = -_reduction_pct(reference, case)
            summary[f"{label}_direction_vs_fair_services_disabled"] = _change_direction(reference, case)
        baseline_change = -_reduction_pct(baseline_cost, case_cost)
        summary["lifecycle_cost_change_pct_vs_baseline"] = baseline_change
        summary["lifecycle_cost_direction_vs_baseline"] = _change_direction(baseline_cost, case_cost)
        output.append(summary)
    return output


def _conclusion_summary(rows: Sequence[Dict], comparison: Sequence[Dict]) -> List[Dict]:
    output = _conclusion_summary_for_rows(rows, comparison, "all")
    archetypes = sorted({str(row["archetype"]) for row in rows if row.get("archetype")})
    for archetype in archetypes:
        subset = [row for row in rows if row.get("archetype") == archetype]
        subset_comparison = [
            row for row in comparison if row.get("archetype") == archetype
        ]
        output.extend(_conclusion_summary_for_rows(subset, subset_comparison, archetype))
    return output


def _write_csv(rows: Iterable[Dict], path: str) -> None:
    data = list(rows)
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(data[0].keys())
    extra = sorted({key for row in data for key in row} - set(fields))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(data)


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(cfg: Config, scenarios: Sequence[Scenario]) -> str:
    config_json = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=str)
    scenario_json = json.dumps([asdict(sc) for sc in scenarios], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{config_json}|{scenario_json}".encode("utf-8")).hexdigest()
    return f"congested3-{digest[:16]}"


def _write_manifest(cfg: Config, scenarios: Sequence[Scenario], output_dir: str) -> None:
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_revision = "unavailable"
    inputs = {}
    for path in (cfg.network_timeseries_csv,):
        if os.path.isfile(path):
            inputs[path] = _hash_file(path)
    sources = {}
    source_roots = [
        "brave",
        "main.py",
        "run_train_test_scenario_validation.py",
        "run_out_of_sample_validation.py",
        "run_manuscript_sensitivity_repdays_annual.py",
        "run_creative_figures.py",
    ]
    for root in source_roots:
        if os.path.isfile(root):
            sources[root] = _hash_file(root)
        elif os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if os.path.isfile(path) and name.endswith(".py"):
                    sources[path.replace("\\", "/")] = _hash_file(path)
    outputs = {}
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path) and name != "manifest.json":
                outputs[name] = _hash_file(path)
    config_payload = asdict(cfg)
    config_json = json.dumps(config_payload, sort_keys=True, separators=(",", ":"), default=str)
    scenario_json = json.dumps([asdict(sc) for sc in scenarios], sort_keys=True, separators=(",", ":"))
    run_id = _run_id(cfg, scenarios)
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision,
        "scenario_count": len(scenarios),
        "scenario_ids": [sc.sid for sc in scenarios],
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "scenario_sha256": hashlib.sha256(scenario_json.encode("utf-8")).hexdigest(),
        "config": config_payload,
        "input_sha256": inputs,
        "source_sha256": sources,
        "output_sha256": outputs,
        "solver": {
            "name": cfg.solver_lp,
            "time_limit_seconds": cfg.solver_time_limit_sec,
            "messages": cfg.solver_msg,
            "robust_objective": cfg.robust_objective,
            "lexicographic_regret_tolerance": cfg.lexicographic_regret_tolerance,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    manifest_dir = os.path.join(output_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, f"{run_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_single(cfg: Config) -> List[Dict]:
    """Run the paper-facing core workflow.

    The default export is intentionally limited to the no-services
    counterfactual, incumbent baseline, and robust shared design. Heavier
    comparator, sensitivity, and validation runs live in separate scripts.
    """
    random.seed(cfg.seed)
    if cfg.enforce_locked_calibration:
        _validate_locked_calibration(cfg)
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    if len(scenarios) != 40 or any(len(group) != 10 for group in study.values()):
        raise ValueError("Canonical study requires four archetypes and ten states per archetype.")
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    decomposition: List[Dict] = []
    comparison: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, perfect_designs = _perfect_designs(group, profiles, constrained_cfg)
        robust = solve_minmax_regret(group, profiles, c_star, constrained_cfg)

        for sc in group:
            prof = profiles[sc.sid]
            cases = {
                "no_services": no_services[sc.sid],
                "fair_services_disabled": _fair_services_disabled_ops(sc, prof, constrained_cfg),
                "baseline": _baseline_ops(sc, prof, constrained_cfg),
                "robust": _solve_design_ops(sc, prof, constrained_cfg, robust),
            }
            for ops in cases.values():
                _apply_emissions_attribution(ops, cases["no_services"])
            _rebase_lifecycle(cases["no_services"], cases["no_services"], cfg_local, is_reference=True)
            _rebase_lifecycle(
                cases["fair_services_disabled"],
                cases["fair_services_disabled"],
                cfg_local,
                is_reference=True,
            )
            for name in DESIGN_CASES:
                if name not in {"no_services", "fair_services_disabled"}:
                    _rebase_lifecycle(cases[name], cases["fair_services_disabled"], cfg_local)
            availability = service_revenue_scale(cfg_local, sc)
            for name, ops in cases.items():
                _apply_sipue(
                    ops,
                    ops if name in {"no_services", "fair_services_disabled"} else cases["fair_services_disabled"],
                    cfg_local,
                    1.0 if name in {"no_services", "fair_services_disabled"} else availability,
                )

            # Compare the cost-regret robust design with the scenario-perfect siPUE benchmark.
            perfect_ops = _solve_design_ops(sc, prof, constrained_cfg, perfect_designs[sc.sid])
            _apply_emissions_attribution(perfect_ops, cases["no_services"])
            _rebase_lifecycle(perfect_ops, cases["fair_services_disabled"], cfg_local)
            _apply_sipue(perfect_ops, cases["fair_services_disabled"], cfg_local, availability)

            row = {
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "case_id": sc.sid,
                "type": sc.dc_type,
                "location": sc.location,
                "construction_delay_months": sc.construction_delay_months + cfg_local.construction_delay_months,
                "lifecycle_availability": availability,
                "carbon_cap_tco2": caps[sc.sid],
                "unrelaxed_carbon_cap_tco2": cases["no_services"].get("unrelaxed_carbon_cap_tco2"),
                "fair_services_disabled_emissions_floor_tco2": cases["no_services"].get(
                    "fair_services_disabled_emissions_floor_tco2"
                ),
                "true_min_emissions_floor_tco2": cases["no_services"].get("true_min_emissions_floor_tco2"),
                "carbon_cap_slack_tco2": cases["no_services"].get("carbon_cap_slack_tco2"),
                "carbon_cap_feasibility_floor_applied": cases["no_services"].get(
                    "carbon_cap_feasibility_floor_applied"
                ),
                "carbon_cap_reduction_target_pct": 100.0 * cfg_local.carbon_cap_reduction_frac,
                "reference_price_eur_per_mwh": cfg_local.sipue_system_ref_price_eur_per_mwh,
                "reference_case": REFERENCE_CASE,
                "service_active_efficiency_reduction": cfg_local.fixed_eff_reduction,
                "no_dc_congestion_fraction": cfg_local.calibrated_no_dc_congestion_fraction,
                "no_dc_load_shedding_mwh": cfg_local.calibrated_no_dc_load_shedding_mwh,
            }
            for name in DESIGN_CASES:
                row.update(_case_fields(cases[name], name))
            row.update(_case_fields(perfect_ops, "scenario_perfect"))
            row["sipue_regret_robust_vs_scenario_perfect"] = (
                float(cases["robust"]["siPUE"]) - float(perfect_ops["siPUE"])
            )
            row["sipue_regret_pct_vs_scenario_perfect"] = _change_pct(
                float(perfect_ops["siPUE"]), float(cases["robust"]["siPUE"])
            )
            _add_land_status_fields(row, cfg_local)
            _add_percentage_fields(row)
            rows.append(row)

            for name in DESIGN_CASES:
                if name not in {"no_services", "fair_services_disabled"}:
                    component_rows = _decomposition_rows(
                        sc.name, cases["fair_services_disabled"], cases[name], name
                    )
                    for item in component_rows:
                        item["archetype"] = archetype.name
                        item["uncertainty_state"] = sc.uncertainty_id
                    decomposition.extend(component_rows)
                case_cost = float(cases[name]["market_planning_cost_npv_eur"])
                discounted_it = max(1e-9, float(cases[name]["E_IT"]) * float(cases[name]["lifecycle_operating_factor"]))
                comparable = name == "robust"
                regret = (
                    max(0.0, float(robust["Regret_s"][sc.sid]))
                    if name == "robust" else None
                )
                comparison.append({
                    "archetype": archetype.name,
                    "uncertainty_state": sc.uncertainty_id,
                    "scenario": sc.name,
                    "design": name,
                    "market_cost_npv_eur": case_cost,
                    "lifecycle_value_npv_eur": -case_cost,
                    "regret_comparable": comparable,
                    "regret_eur": regret,
                    "regret_per_mwh_it": regret / discounted_it if regret is not None else None,
                    "regret_share_of_lifecycle": (
                        100.0 * regret / abs(case_cost) if regret is not None and case_cost != 0 else None
                    ),
                    "regret_share_of_attainable_pct": None,
                    "scenario_perfect_attainable_improvement_eur": None,
                    "emissions_tco2": cases[name]["emissions_tco2"],
                    "carbon_cap_satisfied": cases[name]["emissions_tco2"] <= caps[sc.sid] + 1e-6,
                })

    if cfg.export_csv:
        output_dir = cfg.canonical_output_dir
        run_id = _run_id(cfg_local, scenarios)
        for collection in (rows, decomposition, comparison):
            for item in collection:
                item["run_id"] = run_id
        decomposition_summary = _summarize_decomposition(decomposition)
        comparison_summary = _summarize_design_comparison(comparison)
        conclusion_summary = _conclusion_summary(rows, comparison)
        metric_tests = build_validation_summary(rows)
        for collection in (decomposition_summary, comparison_summary, conclusion_summary, metric_tests):
            for item in collection:
                item["run_id"] = run_id
        _write_csv(rows, os.path.join(output_dir, "results_4_archetypes_10_states.csv"))
        _write_csv(_paper_metric_rows(rows), os.path.join(output_dir, "paper_metrics_long.csv"))
        _write_csv(_kpi_benchmark_summary(rows), os.path.join(output_dir, "kpi_benchmark_summary.csv"))
        _write_csv(_sipue_regret_summary(rows), os.path.join(output_dir, "sipue_regret_summary.csv"))
        _write_csv(decomposition, os.path.join(output_dir, "decomposition.csv"))
        _write_csv(decomposition_summary, os.path.join(output_dir, "decomposition_summary.csv"))
        _write_csv(comparison, os.path.join(output_dir, "design_comparison.csv"))
        _write_csv(comparison_summary, os.path.join(output_dir, "design_comparison_summary.csv"))
        _write_csv(conclusion_summary, os.path.join(output_dir, "conclusion_summary.csv"))
        export_parameter_provenance(
            cfg_local,
            os.path.join(output_dir, "parameter_provenance.csv"),
            run_id=run_id,
        )
        export_literature_matrix(
            os.path.join(output_dir, "literature_matrix.csv"),
            run_id=run_id,
        )
        _write_csv(metric_tests, os.path.join(output_dir, "metric_property_tests.csv"))
        _write_manifest(cfg_local, scenarios, output_dir)
    if cfg.export_plots:
        from brave.plotting import export_study_plots

        export_study_plots(rows, comparison, decomposition, cfg_local)
    return rows


def run_primary_design(cfg: Config) -> List[Dict]:
    """Run only the carbon-constrained normalized-regret design and benchmark.

    This focused path is used for redesign sensitivity cases. It retains the
    same formulation and scenario-perfect benchmark as ``run_single`` without
    solving unrelated comparator and delay-redesign models.
    """
    random.seed(cfg.seed)
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, perfect_designs = _perfect_designs(group, profiles, constrained)
        robust = solve_minmax_regret(group, profiles, c_star, constrained)
        for sc in group:
            identity_reference = no_services[sc.sid]
            reference = _fair_services_disabled_ops(sc, profiles[sc.sid], constrained)
            baseline_ops = _baseline_ops(sc, profiles[sc.sid], constrained)
            robust_ops = _solve_design_ops(sc, profiles[sc.sid], constrained, robust)
            perfect_ops = _solve_design_ops(
                sc, profiles[sc.sid], constrained, perfect_designs[sc.sid]
            )
            for ops in (identity_reference, reference, baseline_ops, robust_ops, perfect_ops):
                _apply_emissions_attribution(ops, identity_reference)
            _rebase_lifecycle(identity_reference, identity_reference, cfg_local, is_reference=True)
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            _rebase_lifecycle(baseline_ops, reference, cfg_local)
            _rebase_lifecycle(robust_ops, reference, cfg_local)
            _rebase_lifecycle(perfect_ops, reference, cfg_local)
            _apply_sipue(identity_reference, identity_reference, cfg_local, 1.0)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            availability = service_revenue_scale(cfg_local, sc)
            _apply_sipue(baseline_ops, reference, cfg_local, availability)
            _apply_sipue(robust_ops, reference, cfg_local, availability)
            _apply_sipue(perfect_ops, reference, cfg_local, availability)
            row = {
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "carbon_cap_tco2": caps[sc.sid],
                "unrelaxed_carbon_cap_tco2": identity_reference.get("unrelaxed_carbon_cap_tco2"),
                "fair_services_disabled_emissions_floor_tco2": identity_reference.get(
                    "fair_services_disabled_emissions_floor_tco2"
                ),
                "true_min_emissions_floor_tco2": identity_reference.get("true_min_emissions_floor_tco2"),
                "carbon_cap_slack_tco2": identity_reference.get("carbon_cap_slack_tco2"),
                "carbon_cap_feasibility_floor_applied": identity_reference.get(
                    "carbon_cap_feasibility_floor_applied"
                ),
                "reference_price_eur_per_mwh": cfg_local.sipue_system_ref_price_eur_per_mwh,
                "reference_case": REFERENCE_CASE,
            }
            row.update(_case_fields(identity_reference, "no_services"))
            row.update(_case_fields(reference, "fair_services_disabled"))
            row.update(_case_fields(baseline_ops, "baseline"))
            row.update(_case_fields(robust_ops, "robust"))
            row.update(_case_fields(perfect_ops, "scenario_perfect"))
            _add_land_status_fields(row, cfg_local)
            _add_percentage_fields(row)
            rows.append(row)
    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["run_id"] = run_id
    return rows


def run_sipue_objective_comparison(cfg: Config) -> List[Dict]:
    """Compare a Stage-2 design chosen to minimize worst-case *cost* regret
    (the paper's canonical "robust" design case, unchanged) against a design
    chosen to minimize worst-case *siPUE* regret directly, for the same 40
    canonical rows.

    Both use the same Stage-1 scenario-perfect scenario set; they differ only in what Stage 2 treats as
    the worst-case criterion (cost regret vs. siPUE regret against a
    per-scenario siPUE-perfect benchmark). This is a diagnostic-only
    comparison run, not part of the canonical result bundle; it exists to
    test whether cost-regret minimization leaves siPUE value on the table
    relative to a design that targets siPUE directly.
    """
    random.seed(cfg.seed)
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    if len(scenarios) != 40 or any(len(group) != 10 for group in study.values()):
        raise ValueError("Canonical study requires four archetypes and ten states per archetype.")
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, _ = _perfect_designs(group, profiles, constrained_cfg)
        fair_refs = {
            sc.sid: _fair_services_disabled_ops(sc, profiles[sc.sid], constrained_cfg)
            for sc in group
        }

        reference_independent_cost = {
            sc.sid: float(fair_refs[sc.sid]["independent_system_cost_eur"]) for sc in group
        }
        availability_by_sid = {sc.sid: service_revenue_scale(constrained_cfg, sc) for sc in group}

        sipue_star: Dict[int, float] = {}
        for sc in group:
            _, sipue_perfect_design = solve_perfect_foresight(
                sc, profiles[sc.sid], constrained_cfg,
                objective_mode="sipue",
                no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                availability=availability_by_sid[sc.sid],
            )
            sipue_perfect_ops = _solve_design_ops(
                sc, profiles[sc.sid], constrained_cfg, sipue_perfect_design,
                objective_mode="sipue",
                no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                availability=availability_by_sid[sc.sid],
            )
            _apply_emissions_attribution(sipue_perfect_ops, no_services[sc.sid])
            _rebase_lifecycle(sipue_perfect_ops, fair_refs[sc.sid], cfg_local)
            _apply_sipue(sipue_perfect_ops, fair_refs[sc.sid], cfg_local, availability_by_sid[sc.sid])
            sipue_star[sc.sid] = float(sipue_perfect_ops["siPUE"])

        robust_cost = solve_minmax_regret(group, profiles, c_star, constrained_cfg)
        robust_sipue = solve_minmax_regret(
            group, profiles, c_star, constrained_cfg,
            objective_mode="sipue",
            no_services_independent_cost=reference_independent_cost,
            availability_by_sid=availability_by_sid,
            sipue_star=sipue_star,
        )

        for sc in group:
            identity_reference = no_services[sc.sid]
            reference = fair_refs[sc.sid]
            cases = {
                "cost_regret_robust": _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, robust_cost),
                "sipue_regret_robust": _solve_design_ops(
                    sc, profiles[sc.sid], constrained_cfg, robust_sipue,
                    objective_mode="sipue",
                    no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                    availability=availability_by_sid[sc.sid],
                ),
            }
            for ops in (identity_reference, reference, *cases.values()):
                _apply_emissions_attribution(ops, identity_reference)
            _rebase_lifecycle(identity_reference, identity_reference, cfg_local, is_reference=True)
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            for ops in cases.values():
                _rebase_lifecycle(ops, reference, cfg_local)
            availability = availability_by_sid[sc.sid]
            _apply_sipue(identity_reference, identity_reference, cfg_local, 1.0)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            for ops in cases.values():
                _apply_sipue(ops, reference, cfg_local, availability)

            row = {
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "carbon_cap_tco2": caps[sc.sid],
                "reference_price_eur_per_mwh": cfg_local.sipue_system_ref_price_eur_per_mwh,
                "reference_case": REFERENCE_CASE,
                "sipue_scenario_perfect_star": sipue_star[sc.sid],
            }
            row.update(_case_fields(identity_reference, "no_services"))
            row.update(_case_fields(reference, "fair_services_disabled"))
            for name, ops in cases.items():
                row.update(_case_fields(ops, name))
            row["sipue_delta_sipue_regret_minus_cost_regret"] = (
                float(cases["sipue_regret_robust"]["siPUE"]) - float(cases["cost_regret_robust"]["siPUE"])
            )
            row["market_cost_delta_sipue_regret_minus_cost_regret_eur"] = (
                float(cases["sipue_regret_robust"]["market_planning_cost_npv_eur"])
                - float(cases["cost_regret_robust"]["market_planning_cost_npv_eur"])
            )
            rows.append(row)

    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["run_id"] = run_id

    if cfg.export_csv:
        output_dir = cfg.canonical_output_dir
        _write_csv(rows, os.path.join(output_dir, "sipue_objective_comparison.csv"))
    return rows


def run_sipue_objective_comparison_out_of_sample(cfg: Config, count_per_archetype: int = 10) -> List[Dict]:
    """Out-of-sample counterpart to ``run_sipue_objective_comparison``.

    The in-sample comparison trains and evaluates both Stage-2 objectives
    (cost regret vs. siPUE regret) on the same ten canonical uncertainty
    states used throughout the paper, which leaves open whether the
    near-identical designs reported there are a structural property of the
    two objectives or a consequence of evaluating the same ten states. This
    function repeats the same
    comparison end to end -- both Stage-1 benchmarks and both Stage-2
    objectives, re-solved from scratch -- on freshly drawn uncertainty
    scenarios from ``make_out_of_sample_scenarios``, using a seed offset
    (900_001) disjoint from the one ``run_out_of_sample_validation`` uses
    (100_003), so these draws have not appeared in any other output in this
    repository. Nothing about the calibrated model itself (archetype
    parameters, prices, capex, land cost, reference price) is changed.

    Note the scope of this check: ``make_out_of_sample_scenarios`` draws each
    uncertain input independently and uniformly within the min/max range
    already spanned by the ten canonical states, so this tests robustness to
    which combination of conditions is drawn within the assumed uncertainty
    envelope, not robustness to the calibrated parameter values themselves.
    This is a diagnostic-only comparison run, not part of the canonical
    result bundle.
    """
    random.seed(cfg.seed)
    study = {
        archetype.key: make_out_of_sample_scenarios(
            cfg, archetype, count_per_archetype, cfg.seed + 900_001 * (index + 1),
        )
        for index, archetype in enumerate(DATA_CENTER_ARCHETYPES)
    }
    scenarios = [sc for group in study.values() for sc in group]
    if any(len(group) != count_per_archetype for group in study.values()):
        raise ValueError("Out-of-sample comparison requires count_per_archetype states per archetype.")
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, _ = _perfect_designs(group, profiles, constrained_cfg)
        fair_refs = {
            sc.sid: _fair_services_disabled_ops(sc, profiles[sc.sid], constrained_cfg)
            for sc in group
        }

        reference_independent_cost = {
            sc.sid: float(fair_refs[sc.sid]["independent_system_cost_eur"]) for sc in group
        }
        availability_by_sid = {sc.sid: service_revenue_scale(constrained_cfg, sc) for sc in group}

        sipue_star: Dict[int, float] = {}
        for sc in group:
            _, sipue_perfect_design = solve_perfect_foresight(
                sc, profiles[sc.sid], constrained_cfg,
                objective_mode="sipue",
                no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                availability=availability_by_sid[sc.sid],
            )
            sipue_perfect_ops = _solve_design_ops(
                sc, profiles[sc.sid], constrained_cfg, sipue_perfect_design,
                objective_mode="sipue",
                no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                availability=availability_by_sid[sc.sid],
            )
            _apply_emissions_attribution(sipue_perfect_ops, no_services[sc.sid])
            _rebase_lifecycle(sipue_perfect_ops, fair_refs[sc.sid], cfg_local)
            _apply_sipue(sipue_perfect_ops, fair_refs[sc.sid], cfg_local, availability_by_sid[sc.sid])
            sipue_star[sc.sid] = float(sipue_perfect_ops["siPUE"])

        robust_cost = solve_minmax_regret(group, profiles, c_star, constrained_cfg)
        robust_sipue = solve_minmax_regret(
            group, profiles, c_star, constrained_cfg,
            objective_mode="sipue",
            no_services_independent_cost=reference_independent_cost,
            availability_by_sid=availability_by_sid,
            sipue_star=sipue_star,
        )

        for sc in group:
            identity_reference = no_services[sc.sid]
            reference = fair_refs[sc.sid]
            cases = {
                "cost_regret_robust": _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, robust_cost),
                "sipue_regret_robust": _solve_design_ops(
                    sc, profiles[sc.sid], constrained_cfg, robust_sipue,
                    objective_mode="sipue",
                    no_services_independent_cost_eur=reference_independent_cost[sc.sid],
                    availability=availability_by_sid[sc.sid],
                ),
            }
            for ops in (identity_reference, reference, *cases.values()):
                _apply_emissions_attribution(ops, identity_reference)
            _rebase_lifecycle(identity_reference, identity_reference, cfg_local, is_reference=True)
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            for ops in cases.values():
                _rebase_lifecycle(ops, reference, cfg_local)
            availability = availability_by_sid[sc.sid]
            _apply_sipue(identity_reference, identity_reference, cfg_local, 1.0)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            for ops in cases.values():
                _apply_sipue(ops, reference, cfg_local, availability)

            row = {
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "carbon_cap_tco2": caps[sc.sid],
                "reference_price_eur_per_mwh": cfg_local.sipue_system_ref_price_eur_per_mwh,
                "reference_case": REFERENCE_CASE,
                "sipue_scenario_perfect_star": sipue_star[sc.sid],
            }
            row.update(_case_fields(identity_reference, "no_services"))
            row.update(_case_fields(reference, "fair_services_disabled"))
            for name, ops in cases.items():
                row.update(_case_fields(ops, name))
            row["sipue_delta_sipue_regret_minus_cost_regret"] = (
                float(cases["sipue_regret_robust"]["siPUE"]) - float(cases["cost_regret_robust"]["siPUE"])
            )
            row["market_cost_delta_sipue_regret_minus_cost_regret_eur"] = (
                float(cases["sipue_regret_robust"]["market_planning_cost_npv_eur"])
                - float(cases["cost_regret_robust"]["market_planning_cost_npv_eur"])
            )
            rows.append(row)

    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["run_id"] = run_id

    if cfg.export_csv:
        output_dir = os.path.join(cfg.validation_output_dir, "out-of-sample")
        _write_csv(rows, os.path.join(output_dir, "sipue_objective_comparison_out_of_sample.csv"))
    return rows


def run_train_test_scenario_validation(cfg: Config) -> List[Dict]:
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        folds = _train_test_split(group, cfg.train_test_max_train_id)
        for fold, test in enumerate(folds, start=1):
            test_ids = {sc.sid for sc in test}
            train = [sc for sc in group if sc.sid not in test_ids]
            c_star, _ = _perfect_designs(train, profiles, constrained_cfg)
            design = solve_minmax_regret(train, profiles, c_star, constrained_cfg)
            for sc in test:
                ops = _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, design)
                identity_reference = no_services[sc.sid]
                reference = _fair_services_disabled_ops(sc, profiles[sc.sid], constrained_cfg)
                _apply_emissions_attribution(identity_reference, identity_reference)
                _apply_emissions_attribution(reference, identity_reference)
                _apply_emissions_attribution(ops, identity_reference)
                design_land = float(ops["land_m2"])
                land_limit = ops["land_limit_m2"]
                land_feasible = land_limit is None or design_land <= float(land_limit) + 1e-6
                technical_feasible, technical_failure = _design_technical_feasibility(design, sc, cfg_local)
                heldout_feasible = land_feasible and technical_feasible
                _, heldout_designs = _perfect_designs([sc], profiles, constrained_cfg)
                perfect = _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, heldout_designs[sc.sid])
                _apply_emissions_attribution(perfect, identity_reference)
                _rebase_lifecycle(identity_reference, identity_reference, cfg_local, is_reference=True)
                _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
                _rebase_lifecycle(ops, reference, cfg_local)
                _rebase_lifecycle(perfect, reference, cfg_local)
                _apply_sipue(reference, reference, cfg_local, 1.0)
                _apply_sipue(ops, reference, cfg_local, service_revenue_scale(cfg_local, sc))
                regret = ops["market_planning_cost_npv_eur"] - perfect["market_planning_cost_npv_eur"]
                discounted_it = max(1e-9, ops["E_IT"] * ops["lifecycle_operating_factor"])
                if heldout_feasible and regret < -1e-3:
                    raise RuntimeError(
                        f"Held-out design appears better than scenario-perfect optimum for scenario {sc.sid}: {regret}."
                    )
                realized = _realize_design(design, sc)
                rows.append({
                    "archetype": archetype.name,
                    "archetype_key": archetype.key,
                    "fold": fold,
                    "train_count": len(train),
                    "test_count": len(test),
                    "scenario": sc.name,
                    "scenario_id": sc.uncertainty_id,
                    "train_scenario_ids": ",".join(str(x.uncertainty_id) for x in train),
                    "market_value_npv_eur": ops["market_planning_value_npv_eur"],
                    "market_cost_npv_eur": ops["market_planning_cost_npv_eur"],
                    "independent_cost_eur": ops["independent_system_cost_eur"],
                    "annual_independent_cost_eur": ops["independent_system_cost_eur"],
                    "regret_eur": max(0.0, regret) if heldout_feasible else None,
                    "regret_eur_per_mwh_it": max(0.0, regret) / discounted_it if heldout_feasible else None,
                    "regret_share_of_lifecycle": (
                        100.0 * max(0.0, regret) / abs(ops["market_planning_cost_npv_eur"])
                        if heldout_feasible else None
                    ),
                    "held_out_design_feasible": heldout_feasible,
                    "held_out_land_feasible": land_feasible,
                    "held_out_technical_feasible": technical_feasible,
                    "held_out_technical_failure": technical_failure,
                    "design_land_m2": design_land,
                    "held_out_land_limit_m2": land_limit,
                    "held_out_land_utilization_pct": ops["land_utilization_pct"],
                    "held_out_land_constraint_binding": ops["land_constraint_binding"],
                    "emissions_tco2": ops["emissions_tco2"],
                    "dc_attributed_emissions_tco2": ops["dc_attributable_emissions_tco2"],
                    "carbon_market_cost_eur": ops["co2_cost_eur"],
                    "carbon_damage_eur": ops["independent_carbon_damage_eur"],
                    "carbon_cap_tco2": caps[sc.sid],
                    "carbon_cap_satisfied": ops["emissions_tco2"] <= caps[sc.sid] + 1e-6,
                    "pue": ops["PUE"],
                    "sipue": ops["siPUE"],
                    "steady_sipue": ops["steady_state_sipue"],
                    "cue": ops["CUE_tCO2_per_MWh_IT"],
                    "system_benefit_per_mwh_it": ops["delay_adjusted_system_benefit_eur"] / max(1e-9, ops["E_IT"]),
                    "cooling_mwh": ops["cooling_mwh"],
                    "recovered_heat_it_mwhth": ops["recovered_heat_it_mwhth"],
                    "recovered_heat_cooling_mwhth": ops["recovered_heat_cooling_mwhth"],
                    "k_hx_mwth": realized["K_HX"],
                    "k_hp_mw": realized["K_HP"],
                    "k_flex_mw": realized["K_FLEX"],
                    "k_bat_e_mwh": realized["K_BAT_E"],
                    "k_bat_p_mw": realized["K_BAT_P"],
                })
    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["run_id"] = run_id
    if cfg.export_csv:
        out_name = f"train_test_1_{cfg.train_test_max_train_id}_vs_rest.csv"
        output_dir = os.path.join(cfg.validation_output_dir, "train-test")
        _write_csv(rows, os.path.join(output_dir, out_name))
        _write_manifest(cfg_local, scenarios, output_dir)
    return rows


def run_out_of_sample_validation(cfg: Config) -> List[Dict]:
    per_archetype = cfg.out_of_sample_case_count // len(DATA_CENTER_ARCHETYPES)
    remainder = cfg.out_of_sample_case_count % len(DATA_CENTER_ARCHETYPES)
    study = make_study_scenarios(cfg)
    oos = {
        archetype.key: make_out_of_sample_scenarios(
            cfg,
            archetype,
            per_archetype + (1 if index < remainder else 0),
            cfg.seed + 100_003 * (index + 1),
        )
        for index, archetype in enumerate(DATA_CENTER_ARCHETYPES)
    }
    all_scenarios = [
        sc for archetype in DATA_CENTER_ARCHETYPES
        for sc in study[archetype.key] + oos[archetype.key]
    ]
    for archetype in DATA_CENTER_ARCHETYPES:
        apply_global_heat_assumptions(study[archetype.key], cfg)
        apply_global_heat_assumptions(oos[archetype.key], cfg)
    cfg_local, profiles = _prepare_profiles(cfg, all_scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        train = study[archetype.key]
        train_no_services, train_caps = _carbon_reference(train, profiles, cfg_local)
        train_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=train_caps)
        c_star, _ = _perfect_designs(train, profiles, train_cfg)
        design = solve_minmax_regret(train, profiles, c_star, train_cfg)
        cases = oos[archetype.key]
        no_services, caps = _carbon_reference(cases, profiles, cfg_local)
        eval_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        for sc in cases:
            identity_reference = no_services[sc.sid]
            reference = _fair_services_disabled_ops(sc, profiles[sc.sid], eval_cfg)
            operational_feasible = True
            failure = ""
            technical_feasible, technical_failure = _design_technical_feasibility(design, sc, cfg_local)
            if not technical_feasible:
                operational_feasible = False
                failure = f"Technical design infeasibility: {technical_failure}"
            try:
                ops = _solve_design_ops(sc, profiles[sc.sid], eval_cfg, design)
            except RuntimeError as exc:
                operational_feasible = False
                failure = "; ".join(item for item in (failure, str(exc)) if item)
                ops = _solve_design_ops(
                    sc,
                    profiles[sc.sid],
                    replace(eval_cfg, enforce_carbon_cap=False),
                    design,
                )
            _apply_emissions_attribution(identity_reference, identity_reference)
            _apply_emissions_attribution(reference, identity_reference)
            _apply_emissions_attribution(ops, identity_reference)
            policy_feasible = True
            try:
                _, perfect_designs = _perfect_designs([sc], profiles, eval_cfg)
                perfect = _solve_design_ops(sc, profiles[sc.sid], eval_cfg, perfect_designs[sc.sid])
            except RuntimeError:
                policy_feasible = False
                unconstrained_eval = replace(eval_cfg, enforce_carbon_cap=False)
                _, perfect_designs = _perfect_designs([sc], profiles, unconstrained_eval)
                perfect = _solve_design_ops(
                    sc, profiles[sc.sid], unconstrained_eval, perfect_designs[sc.sid]
                )
            _apply_emissions_attribution(perfect, identity_reference)
            _rebase_lifecycle(identity_reference, identity_reference, cfg_local, is_reference=True)
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            _rebase_lifecycle(ops, reference, cfg_local)
            _rebase_lifecycle(perfect, reference, cfg_local)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            _apply_sipue(ops, reference, cfg_local, service_revenue_scale(cfg_local, sc))
            regret = ops["market_planning_cost_npv_eur"] - perfect["market_planning_cost_npv_eur"]
            discounted_it = max(1e-9, ops["E_IT"] * ops["lifecycle_operating_factor"])
            land_feasible = bool(ops["land_feasible"])
            carbon_satisfied = ops["emissions_tco2"] <= caps[sc.sid] + 1e-6
            comparable_regret = operational_feasible and policy_feasible and land_feasible
            rows.append({
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "oos_case": sc.uncertainty_id,
                "scenario": sc.name,
                "operational_feasible": operational_feasible,
                "carbon_constrained_perfect_feasible": policy_feasible,
                "land_feasible": land_feasible,
                "carbon_cap_satisfied": carbon_satisfied,
                "failure": failure,
                "market_cost_npv_eur": ops["market_planning_cost_npv_eur"],
                "annual_independent_cost_eur": ops["independent_system_cost_eur"],
                "regret_eur": max(0.0, regret) if comparable_regret else None,
                "regret_eur_per_mwh_it": max(0.0, regret) / discounted_it if comparable_regret else None,
                "regret_share_of_lifecycle": (
                    100.0 * max(0.0, regret) / abs(ops["market_planning_cost_npv_eur"])
                    if comparable_regret else None
                ),
                "emissions_tco2": ops["emissions_tco2"],
                "dc_attributed_emissions_tco2": ops["dc_attributable_emissions_tco2"],
                "carbon_cap_tco2": caps[sc.sid],
                "pue": ops["PUE"],
                "sipue": ops["siPUE"],
                "steady_sipue": ops["steady_state_sipue"],
                "system_benefit_per_mwh_it": ops["delay_adjusted_system_benefit_eur"] / max(1e-9, ops["E_IT"]),
                "land_m2": ops["land_m2"],
                "land_limit_m2": ops["land_limit_m2"],
                "congestion_hours": ops["congestion_hours"],
                "renewable_curtailment_mwh": ops["renewable_curtailment_mwh"],
            })
    run_id = _run_id(cfg_local, all_scenarios)
    for row in rows:
        row["run_id"] = run_id
    if cfg.export_csv:
        output_dir = os.path.join(cfg.validation_output_dir, "out-of-sample")
        _write_csv(rows, os.path.join(output_dir, "out_of_sample_validation.csv"))
        comparable = [row for row in rows if row["regret_eur_per_mwh_it"] is not None]
        summary = [{
            "run_id": run_id,
            "case_count": len(rows),
            "operational_feasibility_pct": 100.0 * sum(bool(row["operational_feasible"]) for row in rows) / max(1, len(rows)),
            "carbon_constrained_perfect_feasibility_pct": 100.0 * sum(bool(row["carbon_constrained_perfect_feasible"]) for row in rows) / max(1, len(rows)),
            "land_feasibility_pct": 100.0 * sum(bool(row["land_feasible"]) for row in rows) / max(1, len(rows)),
            "carbon_cap_satisfaction_pct": 100.0 * sum(bool(row["carbon_cap_satisfied"]) for row in rows) / max(1, len(rows)),
            "comparable_regret_case_count": len(comparable),
            "average_regret_eur_per_mwh_it": (
                sum(float(row["regret_eur_per_mwh_it"]) for row in comparable) / len(comparable)
                if comparable else None
            ),
            "worst_regret_eur_per_mwh_it": max(
                (float(row["regret_eur_per_mwh_it"]) for row in comparable), default=None
            ),
            "average_regret_share_of_lifecycle": (
                sum(float(row["regret_share_of_lifecycle"]) for row in comparable) / len(comparable)
                if comparable else None
            ),
            "worst_regret_share_of_lifecycle": max(
                (float(row["regret_share_of_lifecycle"]) for row in comparable), default=None
            ),
        }]
        _write_csv(summary, os.path.join(output_dir, "out_of_sample_summary.csv"))
        _write_manifest(cfg_local, all_scenarios, output_dir)
    return rows
