from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

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
    "baseline",
    "deterministic",
    "expected_value",
    "pure_minmax",
    "robust",
    "scenario_perfect",
)


def _no_services_profile(prof: Dict[str, List[float]]) -> Dict[str, List[float]]:
    result = dict(prof)
    result["allow_heat"] = False
    result["allow_flex"] = False
    result["allow_battery"] = False
    result["force_it_flex"] = False
    return result


def _make_cv_folds(scenarios: Sequence[Scenario], requested_folds: int) -> List[List[Scenario]]:
    ordered = sorted(scenarios, key=lambda sc: sc.sid)
    if len(ordered) < 2:
        raise ValueError("Cross-validation requires at least two scenarios.")
    k = max(2, min(int(requested_folds), len(ordered)))
    return [ordered[i::k] for i in range(k) if ordered[i::k]]


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
        sc, prof, replace(cfg, enforce_carbon_cap=False),
        k_hx, k_flex, k_bat_e, k_bat_p, eff_reduction=eff, K_HP=k_hp,
    )


def _no_services_ops(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Dict:
    eff = float(cfg.fixed_eff_reduction or 0.0)
    return solve_ops_given_design(
        sc,
        _no_services_profile(prof),
        replace(cfg, enforce_carbon_cap=False, carbon_emissions_cap_tco2_by_sid=None),
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
    reduction = max(0.0, float(cfg.carbon_cap_reduction_frac))
    caps = {}
    for sc in scenarios:
        line_caps = (
            cfg.line_cap_12 * sc.congestion_multiplier,
            cfg.line_cap_23 * sc.congestion_multiplier,
            cfg.line_cap_13 * sc.congestion_multiplier,
        )
        no_dc = solve_no_dc_network(profiles[sc.sid], cfg, line_caps)
        background = (
            float(no_dc["emissions_tco2"])
            + float(no_services[sc.sid]["baseline_heat_emissions_tco2"])
        )
        attributable = max(0.0, float(no_services[sc.sid]["emissions_tco2"]) - background)
        caps[sc.sid] = background + attributable * (1.0 - reduction)
        no_services[sc.sid]["background_emissions_tco2"] = background
        no_services[sc.sid]["dc_attributable_emissions_tco2"] = attributable
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
    ops["siPUE"] = delay_sipue
    ops["steady_state_sipue"] = steady_sipue
    ops["system_credit_mwh"] = credit
    ops["steady_state_system_credit_mwh"] = steady_credit
    ops["raw_system_benefit_eur"] = raw_benefit
    ops["delay_adjusted_system_benefit_eur"] = availability * raw_benefit
    ops["sipue_credit_loss_mwh"] = steady_credit - credit


def _apply_emissions_attribution(ops: Dict, reference: Dict) -> None:
    background = float(reference["background_emissions_tco2"])
    ops["background_emissions_tco2"] = background
    ops["dc_attributable_emissions_tco2"] = float(ops["emissions_tco2"]) - background


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
        "PUE": "PUE",
        "siPUE": "siPUE",
        "SteadySiPUE": "steady_state_sipue",
        "CUE": "CUE_tCO2_per_MWh_IT",
        "EITMWh": "E_IT",
        "EFACMWh": "E_FAC",
        "CoolingMWh": "cooling_mwh",
        "RecoveredHeatITMWhth": "recovered_heat_it_mwhth",
        "RecoveredHeatCoolingMWhth": "recovered_heat_cooling_mwhth",
        "HeatDeliveredMWhth": "heat_MWhth",
        "FlexLoadMWh": "flex_load_mwh",
        "FlexBatteryMWh": "flex_bat_mwh",
        "BatteryChargeMWh": "bat_charge_mwh",
        "BatteryDischargeMWh": "bat_discharge_mwh",
        "StorageLossMWh": "storage_loss_mwh",
        "MarketCostEUR": "market_planning_cost_eur",
        "MarketCostNPVEUR": "market_planning_cost_npv_eur",
        "MarketValueNPVEUR": "market_planning_value_npv_eur",
        "PostConstructionMarketCostNPVEUR": "post_construction_market_cost_npv_eur",
        "LifecycleValueLossEUR": "lifecycle_value_loss_eur",
        "IndependentCostEUR": "independent_system_cost_eur",
        "AnnualIndependentCostEUR": "independent_system_cost_eur",
        "ResourceCostEUR": "el_cost_eur",
        "CarbonMarketCostEUR": "co2_cost_eur",
        "CarbonDamageEUR": "independent_carbon_damage_eur",
        "TransmissionCostEUR": "trans_cost_eur",
        "GridPeakCostEUR": "demand_charge_eur",
        "IndependentPeakCostEUR": "independent_peak_capacity_cost_eur",
        "SLACostEUR": "sla_penalty_eur",
        "AvoidedHeatCostEUR": "independent_avoided_heat_cost_eur",
        "AvoidedFlexCostEUR": "independent_avoided_flex_cost_eur",
        "HeatRevenueEUR": "heat_revenue_eur",
        "FlexRevenueEUR": "flex_revenue_eur",
        "CapexEUR": "capex_eur",
        "LandCostEUR": "land_cost_eur",
        "LandCostNPVEUR": "land_cost_npv_eur",
        "LandM2": "land_m2",
        "ExistingCoreLandM2": "existing_core_land_m2",
        "CoolingLandM2": "existing_cooling_land_m2",
        "IntegrationLandM2": "integration_land_m2",
        "LandLimitM2": "land_limit_m2",
        "LandUtilizationPct": "land_utilization_pct",
        "LandConstraintBinding": "land_constraint_binding",
        "LandFeasible": "land_feasible",
        "EmissionsTCO2": "emissions_tco2",
        "BackgroundEmissionsTCO2": "background_emissions_tco2",
        "DCAttributedEmissionsTCO2": "dc_attributable_emissions_tco2",
        "PowerEmissionsTCO2": "power_emissions_tco2",
        "BaselineHeatEmissionsTCO2": "baseline_heat_emissions_tco2",
        "AvoidedHeatEmissionsTCO2": "avoided_heat_emissions_tco2",
        "PowerCO2CostEUR": "power_co2_cost_eur",
        "BaselineHeatCO2CostEUR": "baseline_heat_co2_cost_eur",
        "AvoidedHeatCO2CostEUR": "avoided_heat_co2_cost_eur",
        "PeakImportMW": "peak_import_mw",
        "CongestionHours": "congestion_hours",
        "CongestionMarginalValueEURPer1Pct": "congestion_marginal_value_eur_per_1pct",
        "CongestionNativeDualMaxEURPerMW": "congestion_native_dual_max_eur_per_mw",
        "RenewableCurtailmentMWh": "renewable_curtailment_mwh",
        "ThermalGenerationMWh": "thermal_generation_mwh",
        "LineCap12MW": "line_cap_12_mw",
        "LineCap23MW": "line_cap_23_mw",
        "LineCap13MW": "line_cap_13_mw",
        "SystemBenefitEUR": "delay_adjusted_system_benefit_eur",
        "UndelayedSystemBenefitEUR": "raw_system_benefit_eur",
        "SystemCreditMWh": "system_credit_mwh",
        "SiPUECreditLossMWh": "sipue_credit_loss_mwh",
        "K_HX_MWth": "K_HX",
        "K_HP_MW": "K_HP",
        "K_FLEX_MW": "K_FLEX",
        "K_BAT_E_MWh": "K_BAT_E",
        "K_BAT_P_MW": "K_BAT_P",
    }
    result = {f"{name}_{suffix}": ops.get(key) for name, key in fields.items()}
    e_it = max(1e-9, float(ops["E_IT"]))
    result[f"SystemBenefitPerMWhIT_{suffix}"] = float(ops.get("delay_adjusted_system_benefit_eur", 0.0)) / e_it
    result[f"AnnualSystemBenefitEUR_{suffix}"] = float(ops.get("delay_adjusted_system_benefit_eur", 0.0))
    result[f"AnnualSystemBenefitEURPerMWhIT_{suffix}"] = (
        float(ops.get("delay_adjusted_system_benefit_eur", 0.0)) / e_it
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
    ns_cost = float(row["MarketCostNPVEUR_no_services"])
    baseline_cost = float(row["MarketCostNPVEUR_baseline"])
    ns_independent = float(row["IndependentCostEUR_no_services"])
    for design in DESIGN_CASES:
        cost = float(row[f"MarketCostNPVEUR_{design}"])
        independent = float(row[f"IndependentCostEUR_{design}"])
        emissions = float(row[f"EmissionsTCO2_{design}"])
        co2_cost = float(row[f"CarbonMarketCostEUR_{design}"])
        damage = float(row[f"CarbonDamageEUR_{design}"])
        pue = float(row[f"PUE_{design}"])
        sipue = float(row[f"siPUE_{design}"])
        row[f"LifecycleCostReductionEURVsNoServices_{design}"] = ns_cost - cost
        row[f"LifecycleCostReductionPctVsNoServices_{design}"] = _reduction_pct(ns_cost, cost)
        row[f"LifecycleCostReductionEURVsBaseline_{design}"] = baseline_cost - cost
        row[f"LifecycleCostReductionPctVsBaseline_{design}"] = _reduction_pct(baseline_cost, cost)
        row[f"IndependentCostReductionPctVsNoServices_{design}"] = _reduction_pct(ns_independent, independent)
        row[f"EmissionsReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["EmissionsTCO2_no_services"]), emissions
        )
        row[f"DCAttributedEmissionsReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["DCAttributedEmissionsTCO2_no_services"]),
            float(row[f"DCAttributedEmissionsTCO2_{design}"]),
        )
        row[f"CO2CostReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["CarbonMarketCostEUR_no_services"]), co2_cost
        )
        row[f"CarbonDamageReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["CarbonDamageEUR_no_services"]), damage
        )
        row[f"PUEReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["PUE_no_services"]), pue
        )
        row[f"SiPUEReductionPctVsNoServices_{design}"] = _reduction_pct(
            float(row["siPUE_no_services"]), sipue
        )
        row[f"LandAreaChangePctVsNoServices_{design}"] = -_reduction_pct(
            float(row["LandM2_no_services"]), float(row[f"LandM2_{design}"])
        )
        row[f"LandLeaseCostChangePctVsNoServices_{design}"] = -_reduction_pct(
            float(row["LandCostNPVEUR_no_services"]), float(row[f"LandCostNPVEUR_{design}"])
        )
        steady_credit = float(row[f"SystemCreditMWh_{design}"]) + float(row[f"SiPUECreditLossMWh_{design}"])
        row[f"SiPUECreditLossPct_{design}"] = (
            100.0 * float(row[f"SiPUECreditLossMWh_{design}"]) / max(1e-9, abs(steady_credit))
        )
        zero_delay_gain = ns_cost - float(row[f"PostConstructionMarketCostNPVEUR_{design}"])
        row[f"DelayLossPctOfZeroDelayGain_{design}"] = (
            100.0 * float(row[f"LifecycleValueLossEUR_{design}"]) / max(1e-9, abs(zero_delay_gain))
        )


def _add_land_status_fields(row: Dict, cfg: Config) -> None:
    for design in DESIGN_CASES:
        land = float(row[f"LandM2_{design}"])
        raw_limit = row.get(f"LandLimitM2_{design}")
        if raw_limit in (None, ""):
            row[f"LandFeasible_{design}"] = True
            row[f"LandConstraintBinding_{design}"] = False
            continue
        limit = float(raw_limit)
        row[f"LandFeasible_{design}"] = land <= limit + cfg.validate_tol
        row[f"LandConstraintBinding_{design}"] = (
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
            "Scenario": scenario,
            "Design": suffix,
            "Ledger": ledger,
            "Component": name,
            "ValueEUR": value,
            "ValuePerMWhIT": value / e_it,
            "ComponentSharePct": 100.0 * value / ledger_total if abs(ledger_total) > 1e-9 else 0.0,
            "LedgerTotalEUR": ledger_total,
            "EITMWh": e_it,
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
    groups: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    for row in rows:
        for archetype in {"all", str(row.get("Archetype", "all"))}:
            key = (archetype, row["Design"], row["Ledger"], row["Component"])
            groups.setdefault(key, []).append(row)
    summary = []
    for (archetype, design, ledger, component), values in sorted(groups.items()):
        eur = [float(value["ValueEUR"]) for value in values]
        per_it = [float(value["ValuePerMWhIT"]) for value in values]
        shares = [float(value["ComponentSharePct"]) for value in values]
        summary.append({
            "Archetype": archetype,
            "Design": design,
            "Ledger": ledger,
            "Component": component,
            "ScenarioCount": len(values),
            "AverageEUR": sum(eur) / len(eur),
            "MinimumEUR": min(eur),
            "MaximumEUR": max(eur),
            "AverageEURPerMWhIT": sum(per_it) / len(per_it),
            "AverageComponentSharePct": sum(shares) / len(shares),
        })
    return summary


def _summarize_design_comparison(rows: Sequence[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        for archetype in {"all", str(row.get("Archetype", "all"))}:
            groups.setdefault((archetype, str(row["Design"])), []).append(row)
    summary = []
    for (archetype, design), values in sorted(groups.items()):
        comparable = [value for value in values if value.get("RegretEUR") not in (None, "")]
        regret = [float(value["RegretEUR"]) for value in comparable]
        regret_per_it = [float(value["RegretPerMWhIT"]) for value in comparable]
        share = [
            float(value["RegretShareOfAttainablePct"])
            for value in comparable
            if value.get("RegretShareOfAttainablePct") not in (None, "")
        ]
        lifecycle_value = [float(value["LifecycleValueNPVEUR"]) for value in values]
        summary.append({
            "Archetype": archetype,
            "Design": design,
            "ScenarioCount": len(values),
            "AverageLifecycleValueNPVEUR": sum(lifecycle_value) / len(lifecycle_value),
            "ComparableScenarioCount": len(comparable),
            "AverageRegretEUR": sum(regret) / len(regret) if regret else None,
            "WorstRegretEUR": max(regret) if regret else None,
            "AverageRegretPerMWhIT": sum(regret_per_it) / len(regret_per_it) if regret_per_it else None,
            "AverageRegretShareOfAttainablePct": sum(share) / len(share) if share else None,
        })
    return summary


def _paper_metric_rows(rows: Sequence[Dict]) -> List[Dict]:
    metrics = {
        "PUE": ("PUE", "dimensionless"),
        "siPUE": ("siPUE", "dimensionless"),
        "post-construction siPUE": ("SteadySiPUE", "dimensionless"),
        "CUE": ("CUE", "tCO2/MWh IT"),
        "CO2 cost": ("CarbonMarketCostEUR", "EUR"),
        "carbon damage": ("CarbonDamageEUR", "EUR"),
        "heat revenue": ("HeatRevenueEUR", "EUR"),
        "flexibility revenue": ("FlexRevenueEUR", "EUR"),
        "emissions": ("EmissionsTCO2", "tCO2"),
        "annual system benefit": ("AnnualSystemBenefitEUR", "annual EUR"),
        "annual benefit per MWh IT": ("AnnualSystemBenefitEURPerMWhIT", "annual EUR/MWh IT"),
        "lifecycle value": ("MarketValueNPVEUR", "EUR"),
        "lifecycle delay loss": ("LifecycleValueLossEUR", "EUR"),
        "cooling electricity": ("CoolingMWh", "MWh"),
        "congestion hours": ("CongestionHours", "h"),
        "renewable curtailment": ("RenewableCurtailmentMWh", "MWh"),
        "thermal generation": ("ThermalGenerationMWh", "MWh"),
        "IT-source recovered heat": ("RecoveredHeatITMWhth", "MWhth"),
        "cooling-source recovered heat": ("RecoveredHeatCoolingMWhth", "MWhth"),
        "total land": ("LandM2", "m2"),
        "integration land": ("IntegrationLandM2", "m2"),
        "land utilization": ("LandUtilizationPct", "%"),
        "land lease NPV": ("LandCostNPVEUR", "EUR"),
        "lifecycle cost reduction vs no-services": ("LifecycleCostReductionPctVsNoServices", "%"),
        "lifecycle cost reduction vs baseline": ("LifecycleCostReductionPctVsBaseline", "%"),
        "emissions reduction vs no-services": ("EmissionsReductionPctVsNoServices", "%"),
        "CO2 cost reduction vs no-services": ("CO2CostReductionPctVsNoServices", "%"),
        "PUE reduction vs no-services": ("PUEReductionPctVsNoServices", "%"),
        "siPUE reduction vs no-services": ("SiPUEReductionPctVsNoServices", "%"),
        "delay loss share": ("DelayLossPctOfZeroDelayGain", "% of zero-delay gain"),
    }
    output = []
    for row in rows:
        for design in DESIGN_CASES:
            for label, (stem, unit) in metrics.items():
                output.append({
                    "RunID": row.get("RunID"),
                    "Archetype": row.get("Archetype"),
                    "UncertaintyState": row.get("ScenarioID"),
                    "Scenario": row["Scenario"],
                    "Design": design,
                    "Metric": label,
                    "Value": row[f"{stem}_{design}"],
                    "Unit": unit,
                    "DelayFactor": row["LifecycleAvailability"],
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
        comparison_by_design.setdefault(str(item["Design"]), []).append(item)
    ns_cost = mean("MarketCostNPVEUR_no_services")
    baseline_cost = mean("MarketCostNPVEUR_baseline")
    output = []
    for design in DESIGN_CASES:
        case_cost = mean(f"MarketCostNPVEUR_{design}")
        items = comparison_by_design.get(design, [])
        comparable_regrets = [
            float(item["RegretEUR"])
            for item in items if item.get("RegretEUR") not in (None, "")
        ]
        attainable_shares = [
            float(item["RegretShareOfAttainablePct"])
            for item in items if item.get("RegretShareOfAttainablePct") not in (None, "")
        ]
        summary = {
            "Archetype": archetype_label,
            "Design": design,
            "ScenarioCount": len(rows),
            "AverageLifecycleCostNPVEUR": case_cost,
            "AverageLifecycleCostReductionEURVsNoServices": ns_cost - case_cost,
            "LifecycleCostReductionPctVsNoServices": _reduction_pct(ns_cost, case_cost),
            "AverageScenarioLifecycleCostReductionPctVsNoServices": mean(f"LifecycleCostReductionPctVsNoServices_{design}"),
            "ScenariosWithLowerLifecycleCostVsNoServices": sum(
                float(row[f"LifecycleCostReductionEURVsNoServices_{design}"]) > 0.0 for row in rows
            ),
            "AverageLifecycleCostReductionEURVsBaseline": baseline_cost - case_cost,
            "LifecycleCostReductionPctVsBaseline": _reduction_pct(baseline_cost, case_cost),
            "AverageScenarioLifecycleCostReductionPctVsBaseline": mean(f"LifecycleCostReductionPctVsBaseline_{design}"),
            "AverageIndependentCostReductionPctVsNoServices": mean(f"IndependentCostReductionPctVsNoServices_{design}"),
            "AverageEmissionsReductionPctVsNoServices": mean(f"EmissionsReductionPctVsNoServices_{design}"),
            "AverageDCAttributedEmissionsReductionPctVsNoServices": mean(f"DCAttributedEmissionsReductionPctVsNoServices_{design}"),
            "AverageCO2CostReductionPctVsNoServices": mean(f"CO2CostReductionPctVsNoServices_{design}"),
            "AverageCarbonDamageReductionPctVsNoServices": mean(f"CarbonDamageReductionPctVsNoServices_{design}"),
            "AveragePUEReductionPctVsNoServices": mean(f"PUEReductionPctVsNoServices_{design}"),
            "AverageSiPUEReductionPctVsNoServices": mean(f"SiPUEReductionPctVsNoServices_{design}"),
            "AverageDelayLossEUR": mean(f"LifecycleValueLossEUR_{design}"),
            "AverageDelayLossPctOfZeroDelayGain": mean(f"DelayLossPctOfZeroDelayGain_{design}"),
            "AverageSiPUECreditLossPct": mean(f"SiPUECreditLossPct_{design}"),
            "AverageLandM2": mean(f"LandM2_{design}"),
            "AverageIntegrationLandM2": mean(f"IntegrationLandM2_{design}"),
            "AverageLandUtilizationPct": mean(f"LandUtilizationPct_{design}"),
            "BindingLandScenarioCount": sum(
                str(row[f"LandConstraintBinding_{design}"]).lower() == "true" for row in rows
            ),
            "AverageLandLeaseNPVEUR": mean(f"LandCostNPVEUR_{design}"),
            "AverageIncrementalLandLeaseNPVEURVsNoServices": (
                mean(f"LandCostNPVEUR_{design}") - mean("LandCostNPVEUR_no_services")
            ),
            "IncrementalLandLeaseShareOfLifecycleGainPct": (
                100.0
                * (mean(f"LandCostNPVEUR_{design}") - mean("LandCostNPVEUR_no_services"))
                / max(1e-9, abs(ns_cost - case_cost))
            ),
            "AverageRegretEUR": (
                sum(comparable_regrets) / len(comparable_regrets) if comparable_regrets else None
            ),
            "WorstRegretEUR": max(comparable_regrets) if comparable_regrets else None,
            "AverageRegretShareOfAttainablePct": (
                sum(attainable_shares) / len(attainable_shares) if attainable_shares else None
            ),
        }
        directional_metrics = {
            "LifecycleCost": (ns_cost, case_cost),
            "IndependentCost": (
                mean("IndependentCostEUR_no_services"),
                mean(f"IndependentCostEUR_{design}"),
            ),
            "Emissions": (mean("EmissionsTCO2_no_services"), mean(f"EmissionsTCO2_{design}")),
            "CO2Cost": (
                mean("CarbonMarketCostEUR_no_services"),
                mean(f"CarbonMarketCostEUR_{design}"),
            ),
            "CarbonDamage": (
                mean("CarbonDamageEUR_no_services"),
                mean(f"CarbonDamageEUR_{design}"),
            ),
            "PUE": (mean("PUE_no_services"), mean(f"PUE_{design}")),
            "SiPUE": (mean("siPUE_no_services"), mean(f"siPUE_{design}")),
            "LandArea": (mean("LandM2_no_services"), mean(f"LandM2_{design}")),
            "LandLeaseCost": (
                mean("LandCostNPVEUR_no_services"),
                mean(f"LandCostNPVEUR_{design}"),
            ),
        }
        for label, (reference, case) in directional_metrics.items():
            summary[f"{label}ChangePctVsNoServices"] = -_reduction_pct(reference, case)
            summary[f"{label}DirectionVsNoServices"] = _change_direction(reference, case)
        baseline_change = -_reduction_pct(baseline_cost, case_cost)
        summary["LifecycleCostChangePctVsBaseline"] = baseline_change
        summary["LifecycleCostDirectionVsBaseline"] = _change_direction(baseline_cost, case_cost)
        output.append(summary)
    return output


def _conclusion_summary(rows: Sequence[Dict], comparison: Sequence[Dict]) -> List[Dict]:
    output = _conclusion_summary_for_rows(rows, comparison, "all")
    archetypes = sorted({str(row["Archetype"]) for row in rows if row.get("Archetype")})
    for archetype in archetypes:
        subset = [row for row in rows if row.get("Archetype") == archetype]
        subset_comparison = [
            row for row in comparison if row.get("Archetype") == archetype
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
        "run_cross_validated_validation.py",
        "run_out_of_sample_validation.py",
        "run_sensitivity_prices_congestion_repdays_annual.py",
        "run_temporal_validation.py",
        "run_service_ablation_validation.py",
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
    redesign: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, perfect_designs = _perfect_designs(group, profiles, constrained_cfg)
        robust = solve_minmax_regret(group, profiles, c_star, constrained_cfg)
        expected = solve_minmax_regret(
            group, profiles, c_star, replace(constrained_cfg, robust_objective="expected_value")
        )
        pure_cfg = replace(
            cfg_local,
            enforce_carbon_cap=False,
            carbon_emissions_cap_tco2_by_sid=None,
            robust_objective="pure_minmax",
        )
        pure_c_star, pure_perfect_designs = _perfect_designs(group, profiles, pure_cfg)
        pure = solve_minmax_regret(group, profiles, pure_c_star, pure_cfg)
        central = next(sc for sc in group if sc.uncertainty_id == 1)
        deterministic = perfect_designs[central.sid]
        steady_group = [
            replace(sc, construction_delay_frac=0.0, construction_delay_months=0.0)
            for sc in group
        ]
        steady_cfg = replace(constrained_cfg, construction_delay_frac=0.0, construction_delay_months=0.0)
        steady_c_star, _ = _perfect_designs(steady_group, profiles, steady_cfg)
        steady_design = solve_minmax_regret(steady_group, profiles, steady_c_star, steady_cfg)

        for sc in group:
            prof = profiles[sc.sid]
            cases = {
                "no_services": no_services[sc.sid],
                "baseline": _baseline_ops(sc, prof, cfg_local),
                "deterministic": _solve_design_ops(sc, prof, constrained_cfg, deterministic),
                "expected_value": _solve_design_ops(sc, prof, constrained_cfg, expected),
                "pure_minmax": _solve_design_ops(sc, prof, pure_cfg, pure),
                "robust": _solve_design_ops(sc, prof, constrained_cfg, robust),
                "scenario_perfect": _solve_design_ops(sc, prof, constrained_cfg, perfect_designs[sc.sid]),
            }
            pure_perfect = _solve_design_ops(sc, prof, pure_cfg, pure_perfect_designs[sc.sid])
            for ops in cases.values():
                _apply_emissions_attribution(ops, cases["no_services"])
            _apply_emissions_attribution(pure_perfect, cases["no_services"])
            _rebase_lifecycle(cases["no_services"], cases["no_services"], cfg_local, is_reference=True)
            for name in DESIGN_CASES:
                if name != "no_services":
                    _rebase_lifecycle(cases[name], cases["no_services"], cfg_local)
            _rebase_lifecycle(pure_perfect, cases["no_services"], cfg_local)
            availability = service_revenue_scale(cfg_local, sc)
            for name, ops in cases.items():
                _apply_sipue(
                    ops,
                    cases["no_services"],
                    cfg_local,
                    1.0 if name == "no_services" else availability,
                )
            steady_under_delay = _solve_design_ops(sc, prof, constrained_cfg, steady_design)
            _apply_emissions_attribution(steady_under_delay, cases["no_services"])
            _rebase_lifecycle(steady_under_delay, cases["no_services"], cfg_local)
            _apply_sipue(steady_under_delay, cases["no_services"], cfg_local, availability)
            robust_realized = _realize_design(robust, sc)
            steady_realized = _realize_design(steady_design, sc)
            redesign.append({
                "Archetype": archetype.name,
                "UncertaintyState": sc.uncertainty_id,
                "Scenario": sc.name,
                "ConstructionDelayMonths": sc.construction_delay_months + cfg_local.construction_delay_months,
                "DelayAwareCostNPVEUR": cases["robust"]["market_planning_cost_npv_eur"],
                "ZeroDelayDesignCostNPVEUR": steady_under_delay["market_planning_cost_npv_eur"],
                "RedesignValueEUR": steady_under_delay["market_planning_cost_npv_eur"] - cases["robust"]["market_planning_cost_npv_eur"],
                "DelayAwareSiPUE": cases["robust"]["siPUE"],
                "ZeroDelayDesignSiPUE": steady_under_delay["siPUE"],
                "DelayAwareWorstRegretEUR": robust["worst_regret_eur"],
                "DelayAwareWorstRegretEURPerMWhIT": robust["worst_regret_eur_per_mwh_it"],
                "ZeroDelayWorstRegretEUR": steady_design["worst_regret_eur"],
                "DelayAwareK_HX_MWth": robust_realized["K_HX"],
                "ZeroDelayK_HX_MWth": steady_realized["K_HX"],
                "DelayAwareK_HP_MW": robust_realized["K_HP"],
                "ZeroDelayK_HP_MW": steady_realized["K_HP"],
                "DelayAwareK_FLEX_MW": robust_realized["K_FLEX"],
                "ZeroDelayK_FLEX_MW": steady_realized["K_FLEX"],
                "DelayAwareK_BAT_E_MWh": robust_realized["K_BAT_E"],
                "ZeroDelayK_BAT_E_MWh": steady_realized["K_BAT_E"],
                "DelayAwareK_BAT_P_MW": robust_realized["K_BAT_P"],
                "ZeroDelayK_BAT_P_MW": steady_realized["K_BAT_P"],
            })

            row = {
                "Archetype": archetype.name,
                "ArchetypeKey": archetype.key,
                "Scenario": sc.name,
                "ScenarioID": sc.uncertainty_id,
                "CaseID": sc.sid,
                "Type": sc.dc_type,
                "Location": sc.location,
                "ConstructionDelayMonths": sc.construction_delay_months + cfg_local.construction_delay_months,
                "LifecycleAvailability": availability,
                "CarbonCapTCO2": caps[sc.sid],
                "CarbonCapReductionTargetPct": 100.0 * cfg_local.carbon_cap_reduction_frac,
                "ReferencePriceEURPerMWh": cfg_local.sipue_system_ref_price_eur_per_mwh,
                "NoDCCongestionFraction": cfg_local.calibrated_no_dc_congestion_fraction,
                "NoDCLoadSheddingMWh": cfg_local.calibrated_no_dc_load_shedding_mwh,
            }
            for name in DESIGN_CASES:
                row.update(_case_fields(cases[name], name))
            _add_land_status_fields(row, cfg_local)
            _add_percentage_fields(row)
            rows.append(row)

            for name in DESIGN_CASES:
                if name != "no_services":
                    component_rows = _decomposition_rows(
                        sc.name, cases["no_services"], cases[name], name
                    )
                    for item in component_rows:
                        item["Archetype"] = archetype.name
                        item["UncertaintyState"] = sc.uncertainty_id
                    decomposition.extend(component_rows)
                comparator = pure_perfect if name == "pure_minmax" else cases["scenario_perfect"]
                perfect_cost = float(comparator["market_planning_cost_npv_eur"])
                case_cost = float(cases[name]["market_planning_cost_npv_eur"])
                no_services_cost = float(cases["no_services"]["market_planning_cost_npv_eur"])
                attainable = no_services_cost - perfect_cost
                discounted_it = max(1e-9, float(cases[name]["E_IT"]) * float(cases[name]["lifecycle_operating_factor"]))
                comparable = name not in {"no_services", "baseline"}
                regret = max(0.0, case_cost - perfect_cost) if comparable else None
                comparison.append({
                    "Archetype": archetype.name,
                    "UncertaintyState": sc.uncertainty_id,
                    "Scenario": sc.name,
                    "Design": name,
                    "MarketCostNPVEUR": case_cost,
                    "LifecycleValueNPVEUR": -case_cost,
                    "RegretComparable": comparable,
                    "RegretEUR": regret,
                    "RegretPerMWhIT": regret / discounted_it if regret is not None else None,
                    "RegretShareOfAttainablePct": (
                        100.0 * regret / attainable
                        if regret is not None and attainable > 1e-6 else None
                    ),
                    "ScenarioPerfectAttainableImprovementEUR": max(0.0, attainable),
                    "EmissionsTCO2": cases[name]["emissions_tco2"],
                    "CarbonCapSatisfied": cases[name]["emissions_tco2"] <= caps[sc.sid] + 1e-6,
                })

    if cfg.export_csv:
        output_dir = cfg.canonical_output_dir
        run_id = _run_id(cfg_local, scenarios)
        for collection in (rows, decomposition, comparison, redesign):
            for item in collection:
                item["RunID"] = run_id
        decomposition_summary = _summarize_decomposition(decomposition)
        comparison_summary = _summarize_design_comparison(comparison)
        conclusion_summary = _conclusion_summary(rows, comparison)
        metric_tests = build_validation_summary(rows)
        for collection in (decomposition_summary, comparison_summary, conclusion_summary, metric_tests):
            for item in collection:
                item["RunID"] = run_id
        _write_csv(rows, os.path.join(output_dir, "results_4_archetypes_10_states.csv"))
        _write_csv(_paper_metric_rows(rows), os.path.join(output_dir, "paper_metrics_long.csv"))
        _write_csv(decomposition, os.path.join(output_dir, "decomposition.csv"))
        _write_csv(decomposition_summary, os.path.join(output_dir, "decomposition_summary.csv"))
        _write_csv(comparison, os.path.join(output_dir, "design_comparison.csv"))
        _write_csv(comparison_summary, os.path.join(output_dir, "design_comparison_summary.csv"))
        _write_csv(conclusion_summary, os.path.join(output_dir, "conclusion_summary.csv"))
        _write_csv(redesign, os.path.join(output_dir, "delay_redesign_comparison.csv"))
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
            reference = no_services[sc.sid]
            robust_ops = _solve_design_ops(sc, profiles[sc.sid], constrained, robust)
            perfect_ops = _solve_design_ops(
                sc, profiles[sc.sid], constrained, perfect_designs[sc.sid]
            )
            for ops in (reference, robust_ops, perfect_ops):
                _apply_emissions_attribution(ops, reference)
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            _rebase_lifecycle(robust_ops, reference, cfg_local)
            _rebase_lifecycle(perfect_ops, reference, cfg_local)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            availability = service_revenue_scale(cfg_local, sc)
            _apply_sipue(robust_ops, reference, cfg_local, availability)
            _apply_sipue(perfect_ops, reference, cfg_local, availability)
            row = {
                "Archetype": archetype.name,
                "ArchetypeKey": archetype.key,
                "Scenario": sc.name,
                "ScenarioID": sc.uncertainty_id,
                "CarbonCapTCO2": caps[sc.sid],
                "ReferencePriceEURPerMWh": cfg_local.sipue_system_ref_price_eur_per_mwh,
            }
            row.update(_case_fields(reference, "no_services"))
            row.update(_case_fields(robust_ops, "robust"))
            row.update(_case_fields(perfect_ops, "scenario_perfect"))
            rows.append(row)
    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["RunID"] = run_id
    return rows


def run_cross_validated_validation(cfg: Config) -> List[Dict]:
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
        for fold, test in enumerate(_make_cv_folds(group, cfg.validation_cv_folds), start=1):
            test_ids = {sc.sid for sc in test}
            train = [sc for sc in group if sc.sid not in test_ids]
            c_star, _ = _perfect_designs(train, profiles, constrained_cfg)
            design = solve_minmax_regret(train, profiles, c_star, constrained_cfg)
            for sc in test:
                ops = _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, design)
                _apply_emissions_attribution(ops, no_services[sc.sid])
                design_land = float(ops["land_m2"])
                land_limit = ops["land_limit_m2"]
                land_feasible = land_limit is None or design_land <= float(land_limit) + 1e-6
                technical_feasible, technical_failure = _design_technical_feasibility(design, sc, cfg_local)
                heldout_feasible = land_feasible and technical_feasible
                _, heldout_designs = _perfect_designs([sc], profiles, constrained_cfg)
                perfect = _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, heldout_designs[sc.sid])
                _rebase_lifecycle(no_services[sc.sid], no_services[sc.sid], cfg_local, is_reference=True)
                _rebase_lifecycle(ops, no_services[sc.sid], cfg_local)
                _rebase_lifecycle(perfect, no_services[sc.sid], cfg_local)
                _apply_sipue(ops, no_services[sc.sid], cfg_local, service_revenue_scale(cfg_local, sc))
                regret = ops["market_planning_cost_npv_eur"] - perfect["market_planning_cost_npv_eur"]
                discounted_it = max(1e-9, ops["E_IT"] * ops["lifecycle_operating_factor"])
                if heldout_feasible and regret < -1e-3:
                    raise RuntimeError(
                        f"Held-out design appears better than scenario-perfect optimum for scenario {sc.sid}: {regret}."
                    )
                realized = _realize_design(design, sc)
                rows.append({
                    "Archetype": archetype.name,
                    "ArchetypeKey": archetype.key,
                    "Fold": fold,
                    "TrainCount": len(train),
                    "TestCount": len(test),
                    "Scenario": sc.name,
                    "ScenarioID": sc.uncertainty_id,
                    "TrainScenarioIDs": ",".join(str(x.uncertainty_id) for x in train),
                    "MarketValueNPVEUR": ops["market_planning_value_npv_eur"],
                    "MarketCostNPVEUR": ops["market_planning_cost_npv_eur"],
                    "IndependentCostEUR": ops["independent_system_cost_eur"],
                    "AnnualIndependentCostEUR": ops["independent_system_cost_eur"],
                    "RegretEUR": max(0.0, regret) if heldout_feasible else None,
                    "RegretEURPerMWhIT": max(0.0, regret) / discounted_it if heldout_feasible else None,
                    "HeldOutDesignFeasible": heldout_feasible,
                    "HeldOutLandFeasible": land_feasible,
                    "HeldOutTechnicalFeasible": technical_feasible,
                    "HeldOutTechnicalFailure": technical_failure,
                    "DesignLandM2": design_land,
                    "HeldOutLandLimitM2": land_limit,
                    "HeldOutLandUtilizationPct": ops["land_utilization_pct"],
                    "HeldOutLandConstraintBinding": ops["land_constraint_binding"],
                    "EmissionsTCO2": ops["emissions_tco2"],
                    "DCAttributedEmissionsTCO2": ops["dc_attributable_emissions_tco2"],
                    "CarbonMarketCostEUR": ops["co2_cost_eur"],
                    "CarbonDamageEUR": ops["independent_carbon_damage_eur"],
                    "CarbonCapTCO2": caps[sc.sid],
                    "CarbonCapSatisfied": ops["emissions_tco2"] <= caps[sc.sid] + 1e-6,
                    "PUE": ops["PUE"],
                    "siPUE": ops["siPUE"],
                    "SteadySiPUE": ops["steady_state_sipue"],
                    "CUE": ops["CUE_tCO2_per_MWh_IT"],
                    "SystemBenefitPerMWhIT": ops["delay_adjusted_system_benefit_eur"] / max(1e-9, ops["E_IT"]),
                    "CoolingMWh": ops["cooling_mwh"],
                    "RecoveredHeatITMWhth": ops["recovered_heat_it_mwhth"],
                    "RecoveredHeatCoolingMWhth": ops["recovered_heat_cooling_mwhth"],
                    "K_HX_MWth": realized["K_HX"],
                    "K_HP_MW": realized["K_HP"],
                    "K_FLEX_MW": realized["K_FLEX"],
                    "K_BAT_E_MWh": realized["K_BAT_E"],
                    "K_BAT_P_MW": realized["K_BAT_P"],
                })
    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["RunID"] = run_id
    if cfg.export_csv:
        _write_csv(rows, os.path.join(cfg.canonical_output_dir, "cross_validated_design_performance.csv"))
        _write_manifest(cfg_local, scenarios, cfg.canonical_output_dir)
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
            _apply_emissions_attribution(ops, no_services[sc.sid])
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
            _rebase_lifecycle(no_services[sc.sid], no_services[sc.sid], cfg_local, is_reference=True)
            _rebase_lifecycle(ops, no_services[sc.sid], cfg_local)
            _rebase_lifecycle(perfect, no_services[sc.sid], cfg_local)
            _apply_sipue(ops, no_services[sc.sid], cfg_local, service_revenue_scale(cfg_local, sc))
            regret = ops["market_planning_cost_npv_eur"] - perfect["market_planning_cost_npv_eur"]
            discounted_it = max(1e-9, ops["E_IT"] * ops["lifecycle_operating_factor"])
            land_feasible = bool(ops["land_feasible"])
            carbon_satisfied = ops["emissions_tco2"] <= caps[sc.sid] + 1e-6
            comparable_regret = operational_feasible and policy_feasible and land_feasible
            rows.append({
                "Archetype": archetype.name,
                "ArchetypeKey": archetype.key,
                "OOSCase": sc.uncertainty_id,
                "Scenario": sc.name,
                "OperationalFeasible": operational_feasible,
                "CarbonConstrainedPerfectFeasible": policy_feasible,
                "LandFeasible": land_feasible,
                "CarbonCapSatisfied": carbon_satisfied,
                "Failure": failure,
                "MarketCostNPVEUR": ops["market_planning_cost_npv_eur"],
                "AnnualIndependentCostEUR": ops["independent_system_cost_eur"],
                "RegretEUR": max(0.0, regret) if comparable_regret else None,
                "RegretEURPerMWhIT": max(0.0, regret) / discounted_it if comparable_regret else None,
                "EmissionsTCO2": ops["emissions_tco2"],
                "DCAttributedEmissionsTCO2": ops["dc_attributable_emissions_tco2"],
                "CarbonCapTCO2": caps[sc.sid],
                "PUE": ops["PUE"],
                "siPUE": ops["siPUE"],
                "SteadySiPUE": ops["steady_state_sipue"],
                "SystemBenefitPerMWhIT": ops["delay_adjusted_system_benefit_eur"] / max(1e-9, ops["E_IT"]),
                "LandM2": ops["land_m2"],
                "LandLimitM2": ops["land_limit_m2"],
                "CongestionHours": ops["congestion_hours"],
                "RenewableCurtailmentMWh": ops["renewable_curtailment_mwh"],
            })
    run_id = _run_id(cfg_local, all_scenarios)
    for row in rows:
        row["RunID"] = run_id
    if cfg.export_csv:
        _write_csv(rows, os.path.join(cfg.canonical_output_dir, "out_of_sample_validation.csv"))
        comparable = [row for row in rows if row["RegretEURPerMWhIT"] is not None]
        summary = [{
            "RunID": run_id,
            "CaseCount": len(rows),
            "OperationalFeasibilityPct": 100.0 * sum(bool(row["OperationalFeasible"]) for row in rows) / max(1, len(rows)),
            "CarbonConstrainedPerfectFeasibilityPct": 100.0 * sum(bool(row["CarbonConstrainedPerfectFeasible"]) for row in rows) / max(1, len(rows)),
            "LandFeasibilityPct": 100.0 * sum(bool(row["LandFeasible"]) for row in rows) / max(1, len(rows)),
            "CarbonCapSatisfactionPct": 100.0 * sum(bool(row["CarbonCapSatisfied"]) for row in rows) / max(1, len(rows)),
            "ComparableRegretCaseCount": len(comparable),
            "AverageRegretEURPerMWhIT": (
                sum(float(row["RegretEURPerMWhIT"]) for row in comparable) / len(comparable)
                if comparable else None
            ),
            "WorstRegretEURPerMWhIT": max(
                (float(row["RegretEURPerMWhIT"]) for row in comparable), default=None
            ),
        }]
        _write_csv(summary, os.path.join(cfg.canonical_output_dir, "out_of_sample_summary.csv"))
        _write_manifest(cfg_local, all_scenarios, cfg.canonical_output_dir)
    return rows


SERVICE_ABLATIONS = {
    "cooling_only": (False, False, False),
    "heat_only": (True, False, False),
    "flexibility_only": (False, True, False),
    "storage_only": (False, False, True),
    "all_services": (True, True, True),
}


def _service_ablation_profile(profile: Dict, mode: str, storage_available: bool) -> Dict:
    allow_heat, allow_flex, allow_battery = SERVICE_ABLATIONS[mode]
    result = dict(profile)
    result["allow_heat"] = allow_heat
    result["allow_flex"] = allow_flex
    result["allow_battery"] = allow_battery and storage_available
    result["force_it_flex"] = False
    return result


def run_service_ablation_validation(cfg: Config) -> List[Dict]:
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, base_profiles = _prepare_profiles(cfg, scenarios)
    rows: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, base_profiles, cfg_local)
        constrained = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        for mode in SERVICE_ABLATIONS:
            mode_profiles = {
                sc.sid: _service_ablation_profile(
                    base_profiles[sc.sid], mode, sc.storage_availability > 0.0
                )
                for sc in group
            }
            carbon_constrained_design_feasible = True
            formulation = constrained
            try:
                c_star, _ = _perfect_designs(group, mode_profiles, formulation)
                design = solve_minmax_regret(group, mode_profiles, c_star, formulation)
            except RuntimeError:
                carbon_constrained_design_feasible = False
                formulation = replace(
                    cfg_local,
                    enforce_carbon_cap=False,
                    carbon_emissions_cap_tco2_by_sid=None,
                )
                c_star, _ = _perfect_designs(group, mode_profiles, formulation)
                design = solve_minmax_regret(group, mode_profiles, c_star, formulation)
            for sc in group:
                ops = _solve_design_ops(sc, mode_profiles[sc.sid], formulation, design)
                reference = no_services[sc.sid]
                _apply_emissions_attribution(ops, reference)
                _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
                _rebase_lifecycle(ops, reference, cfg_local)
                _apply_sipue(ops, reference, cfg_local, service_revenue_scale(cfg_local, sc))
                realized = _realize_design(design, sc)
                discounted_it = max(1e-9, ops["E_IT"] * ops["lifecycle_operating_factor"])
                rows.append({
                    "Archetype": archetype.name,
                    "ArchetypeKey": archetype.key,
                    "Mode": mode,
                    "ScenarioID": sc.uncertainty_id,
                    "Scenario": sc.name,
                    "CarbonConstrainedDesignFeasible": carbon_constrained_design_feasible,
                    "EvaluationFormulation": (
                        "carbon_constrained" if carbon_constrained_design_feasible else "unconstrained_diagnostic"
                    ),
                    "MarketCostNPVEUR": ops["market_planning_cost_npv_eur"],
                    "MarketCostChangePctVsNoServices": _change_pct(
                        reference["market_planning_cost_npv_eur"], ops["market_planning_cost_npv_eur"]
                    ),
                    "IndependentBenefitEUR": reference["independent_system_cost_eur"] - ops["independent_system_cost_eur"],
                    "IndependentBenefitEURPerMWhIT": (
                        reference["independent_system_cost_eur"] - ops["independent_system_cost_eur"]
                    ) / max(1e-9, ops["E_IT"]),
                    "PUE": ops["PUE"],
                    "siPUE": ops["siPUE"],
                    "PostConstructionSiPUE": ops["steady_state_sipue"],
                    "CUE": ops["CUE_tCO2_per_MWh_IT"],
                    "EmissionsTCO2": ops["emissions_tco2"],
                    "DCAttributedEmissionsTCO2": ops["dc_attributable_emissions_tco2"],
                    "CarbonCapSatisfied": ops["emissions_tco2"] <= caps[sc.sid] + 1e-6,
                    "HeatRevenueEUR": ops["heat_revenue_eur"],
                    "FlexRevenueEUR": ops["flex_revenue_eur"],
                    "AvoidedHeatCostEUR": ops["independent_avoided_heat_cost_eur"],
                    "AvoidedFlexCostEUR": ops["independent_avoided_flex_cost_eur"],
                    "CarbonMarketCostEUR": ops["co2_cost_eur"],
                    "CarbonDamageEUR": ops["independent_carbon_damage_eur"],
                    "ResourceCostEUR": ops["el_cost_eur"] + ops["baseline_heat_resource_cost_eur"],
                    "TransmissionCostEUR": ops["trans_cost_eur"],
                    "PeakCapacityCostEUR": ops["independent_peak_capacity_cost_eur"],
                    "CoolingMWh": ops["cooling_mwh"],
                    "RecoveredHeatITMWhth": ops["recovered_heat_it_mwhth"],
                    "RecoveredHeatCoolingMWhth": ops["recovered_heat_cooling_mwhth"],
                    "CongestionHours": ops["congestion_hours"],
                    "CongestionMarginalValueEURPer1Pct": ops["congestion_marginal_value_eur_per_1pct"],
                    "RenewableCurtailmentMWh": ops["renewable_curtailment_mwh"],
                    "ThermalGenerationMWh": ops["thermal_generation_mwh"],
                    "RegretEURPerMWhIT": design["NormalizedRegret_s"][sc.sid],
                    "DiscountedITMWh": discounted_it,
                    "K_HX_MWth": realized["K_HX"],
                    "K_HP_MW": realized["K_HP"],
                    "K_FLEX_MW": realized["K_FLEX"],
                    "K_BAT_E_MWh": realized["K_BAT_E"],
                    "K_BAT_P_MW": realized["K_BAT_P"],
                })
    run_id = _run_id(cfg_local, scenarios)
    for row in rows:
        row["RunID"] = run_id
    if cfg.export_csv:
        output_dir = cfg.canonical_output_dir
        _write_csv(rows, os.path.join(output_dir, "service_ablation_results.csv"))
        summary = []
        for archetype in ["all", *(item.name for item in DATA_CENTER_ARCHETYPES)]:
            subset = rows if archetype == "all" else [row for row in rows if row["Archetype"] == archetype]
            for mode in SERVICE_ABLATIONS:
                values = [row for row in subset if row["Mode"] == mode]
                summary.append({
                    "RunID": run_id,
                    "Archetype": archetype,
                    "Mode": mode,
                    "ScenarioCount": len(values),
                    "AverageMarketCostChangePctVsNoServices": sum(float(row["MarketCostChangePctVsNoServices"]) for row in values) / len(values),
                    "AverageIndependentBenefitEURPerMWhIT": sum(float(row["IndependentBenefitEURPerMWhIT"]) for row in values) / len(values),
                    "AveragePUE": sum(float(row["PUE"]) for row in values) / len(values),
                    "AverageSiPUE": sum(float(row["siPUE"]) for row in values) / len(values),
                    "AverageEmissionsTCO2": sum(float(row["EmissionsTCO2"]) for row in values) / len(values),
                    "CarbonCapSatisfactionPct": 100.0 * sum(bool(row["CarbonCapSatisfied"]) for row in values) / len(values),
                    "CarbonConstrainedDesignFeasible": all(bool(row["CarbonConstrainedDesignFeasible"]) for row in values),
                    "AverageCongestionHours": sum(float(row["CongestionHours"]) for row in values) / len(values),
                    "AverageCurtailmentMWh": sum(float(row["RenewableCurtailmentMWh"]) for row in values) / len(values),
                    "WorstRegretEURPerMWhIT": max(float(row["RegretEURPerMWhIT"]) for row in values),
                })
        _write_csv(summary, os.path.join(output_dir, "service_ablation_summary.csv"))
        _write_manifest(cfg_local, scenarios, output_dir)
    return rows
