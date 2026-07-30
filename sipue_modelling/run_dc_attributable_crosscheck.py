"""Cross-check the siPUE system-benefit credit against DC-attributable-only cost.

``IndependentSystemCostEUR`` (the counterfactual cost term siPUE is built on)
is computed from the full 3-node network's resource cost, carbon damage, and
transmission cost -- background load at all three nodes plus the data center
-- not from a DC-only accounting. The siPUE credit is the delta between the
no-services and services-active versions of that whole-network total. This
diagnostic script compares that accounting choice with a DC-attributable-only
alternative.

This script builds that second accounting the same way the existing
DCAttributedEmissionsTCO2 diagnostic already does for emissions (background
= no-DC-only dispatch; attributable = max(0, with-DC total - background)),
extended here to resource cost, transmission cost, and carbon damage.
"""
from __future__ import annotations

import csv
import os
from dataclasses import replace
from typing import Dict, List

from brave.config import Config
from brave.metrics import compute_sipue, service_revenue_scale
from brave.network import solve_no_dc_network
from brave.optimization import solve_minmax_regret
from brave.runners import (
    _apply_emissions_attribution,
    _carbon_reference,
    _perfect_designs,
    _prepare_profiles,
    _solve_design_ops,
)
from brave.scenario import DATA_CENTER_ARCHETYPES, apply_global_heat_assumptions, make_study_scenarios


def _line_caps(cfg: Config, sc) -> tuple:
    return (
        cfg.line_cap_12 * sc.congestion_multiplier,
        cfg.line_cap_23 * sc.congestion_multiplier,
        cfg.line_cap_13 * sc.congestion_multiplier,
    )


def _trans_cost_from_flows(cfg: Config, flows: Dict[str, List[float]], w: List[float]) -> float:
    if cfg.transmission_cost_eur_per_mwh <= 0:
        return 0.0
    T = len(w)
    return cfg.transmission_cost_eur_per_mwh * sum(
        (abs(flows["12"][t]) + abs(flows["23"][t]) + abs(flows["13"][t])) * w[t]
        for t in range(T)
    )


def _dc_attributable_independent_cost(
    ops: Dict,
    cfg: Config,
    sc,
    no_dc_resource_cost_eur: float,
    no_dc_trans_cost_eur: float,
) -> Dict[str, float]:
    resource_cost_attrib = max(
        0.0,
        float(ops["el_cost_eur"]) + float(ops["baseline_heat_resource_cost_eur"]) - no_dc_resource_cost_eur,
    )
    trans_cost_attrib = max(0.0, float(ops["trans_cost_eur"]) - no_dc_trans_cost_eur)
    emissions_attrib = max(0.0, float(ops["dc_attributable_emissions_tco2"]))
    damage_value = (
        float(sc.carbon_damage_eur_per_tco2)
        if sc.carbon_damage_eur_per_tco2 is not None
        else cfg.independent_social_carbon_cost_eur_per_tco2
    )
    carbon_damage_attrib = damage_value * emissions_attrib
    independent_cost_attrib = (
        resource_cost_attrib
        + trans_cost_attrib
        + carbon_damage_attrib
        + float(ops["independent_peak_capacity_cost_eur"])
        - float(ops["independent_avoided_heat_cost_eur"])
        - float(ops["independent_avoided_flex_cost_eur"])
    )
    return {
        "resource_cost_attrib_eur": resource_cost_attrib,
        "trans_cost_attrib_eur": trans_cost_attrib,
        "emissions_attrib_tco2": emissions_attrib,
        "carbon_damage_attrib_eur": carbon_damage_attrib,
        "independent_cost_attrib_eur": independent_cost_attrib,
    }


def main() -> None:
    cfg = Config(enforce_locked_calibration=False, export_csv=False, export_plots=False, export_regret_landscape=False)
    study = make_study_scenarios(cfg)
    scenarios = [sc for group in study.values() for sc in group]
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)

    detailed: List[Dict] = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        c_star, _perfect_designs_by_sid = _perfect_designs(group, profiles, constrained_cfg)
        robust = solve_minmax_regret(group, profiles, c_star, constrained_cfg)

        for sc in group:
            reference = no_services[sc.sid]
            robust_ops = _solve_design_ops(sc, profiles[sc.sid], constrained_cfg, robust)
            _apply_emissions_attribution(robust_ops, reference)

            prof = profiles[sc.sid]
            w = prof.get("w", [1.0] * len(prof["fuel_price"]))
            no_dc = solve_no_dc_network(prof, cfg_local, _line_caps(cfg_local, sc))
            no_dc_trans_cost = _trans_cost_from_flows(cfg_local, no_dc["flows"], w)

            attrib_ref = _dc_attributable_independent_cost(
                reference, cfg_local, sc, float(no_dc["resource_cost_eur"]), no_dc_trans_cost
            )
            attrib_robust = _dc_attributable_independent_cost(
                robust_ops, cfg_local, sc, float(no_dc["resource_cost_eur"]), no_dc_trans_cost
            )

            credit_whole_network = float(reference["independent_system_cost_eur"]) - float(
                robust_ops["independent_system_cost_eur"]
            )
            credit_dc_attrib = attrib_ref["independent_cost_attrib_eur"] - attrib_robust["independent_cost_attrib_eur"]

            availability = service_revenue_scale(cfg_local, sc)
            sipue_whole, _ = compute_sipue(
                E_FAC=float(robust_ops["E_FAC"]),
                E_IT=float(robust_ops["E_IT"]),
                net_system_benefit_eur=credit_whole_network,
                reference_price_eur_per_mwh=cfg_local.sipue_system_ref_price_eur_per_mwh,
                clip_nonnegative=cfg_local.sipue_clip_nonnegative,
                benefit_availability_factor=availability,
            )
            sipue_dc_attrib, _ = compute_sipue(
                E_FAC=float(robust_ops["E_FAC"]),
                E_IT=float(robust_ops["E_IT"]),
                net_system_benefit_eur=credit_dc_attrib,
                reference_price_eur_per_mwh=cfg_local.sipue_system_ref_price_eur_per_mwh,
                clip_nonnegative=cfg_local.sipue_clip_nonnegative,
                benefit_availability_factor=availability,
            )
            pue = float(robust_ops["PUE"])
            sipue_no_services = float(reference["siPUE"])

            detailed.append({
                "Archetype": archetype.name,
                "ScenarioID": sc.uncertainty_id,
                "Scenario": sc.name,
                "NoDCResourceCostEUR": float(no_dc["resource_cost_eur"]),
                "NoDCTransCostEUR": no_dc_trans_cost,
                "WholeNetworkResourceCostEUR_no_services": float(reference["el_cost_eur"]) + float(reference["baseline_heat_resource_cost_eur"]),
                "WholeNetworkResourceCostEUR_robust": float(robust_ops["el_cost_eur"]) + float(robust_ops["baseline_heat_resource_cost_eur"]),
                "DCAttribResourceCostEUR_no_services": attrib_ref["resource_cost_attrib_eur"],
                "DCAttribResourceCostEUR_robust": attrib_robust["resource_cost_attrib_eur"],
                "DCAttribIndependentCostEUR_no_services": attrib_ref["independent_cost_attrib_eur"],
                "DCAttribIndependentCostEUR_robust": attrib_robust["independent_cost_attrib_eur"],
                "WholeNetworkIndependentCostEUR_no_services": float(reference["independent_system_cost_eur"]),
                "WholeNetworkIndependentCostEUR_robust": float(robust_ops["independent_system_cost_eur"]),
                "CreditWholeNetworkEUR": credit_whole_network,
                "CreditDCAttributableEUR": credit_dc_attrib,
                "CreditSignAgree": (credit_whole_network >= 0) == (credit_dc_attrib >= 0),
                "PUE_robust": pue,
                "SiPUE_no_services": sipue_no_services,
                "SiPUE_robust_wholeNetworkCredit": sipue_whole,
                "SiPUE_robust_DCAttributableCredit": sipue_dc_attrib,
                "SiPUEChangePct_wholeNetwork": 100.0 * (sipue_whole - sipue_no_services) / max(1e-9, abs(sipue_no_services)),
                "SiPUEChangePct_DCAttributable": 100.0 * (sipue_dc_attrib - sipue_no_services) / max(1e-9, abs(sipue_no_services)),
            })

    out_dir = os.path.join("output", "validation")
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "dc_attributable_crosscheck.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0].keys()))
        writer.writeheader()
        writer.writerows(detailed)

    n = len(detailed)
    mean_whole = sum(r["SiPUEChangePct_wholeNetwork"] for r in detailed) / n
    mean_attrib = sum(r["SiPUEChangePct_DCAttributable"] for r in detailed) / n
    sign_agree = sum(1 for r in detailed if r["CreditSignAgree"]) / n
    n_attrib_improve = sum(1 for r in detailed if r["SiPUEChangePct_DCAttributable"] < 0.0)
    mean_credit_whole = sum(r["CreditWholeNetworkEUR"] for r in detailed) / n
    mean_credit_attrib = sum(r["CreditDCAttributableEUR"] for r in detailed) / n

    print(f"Scenarios evaluated: {n}")
    print(f"Mean siPUE change vs no-services -- whole-network credit:      {mean_whole:+.2f}%")
    print(f"Mean siPUE change vs no-services -- DC-attributable-only credit: {mean_attrib:+.2f}%")
    print(f"Scenarios where DC-attributable credit still improves siPUE: {n_attrib_improve}/{n}")
    print(f"Credit sign agreement (whole-network vs DC-attributable): {sign_agree*100:.1f}%")
    print(f"Mean credit EUR -- whole-network: {mean_credit_whole:,.0f}   DC-attributable: {mean_credit_attrib:,.0f}")
    print(f"\nWrote per-scenario detail to {detail_path}")


if __name__ == "__main__":
    main()
