from __future__ import annotations

import argparse
import os
import random
from dataclasses import replace
from typing import Dict, List

from brave.config import Config
from brave.runners import (
    _apply_emissions_attribution,
    _apply_sipue,
    _carbon_reference,
    _no_services_ops,
    _rebase_lifecycle,
    _solve_design_ops,
    _write_csv,
    run_single,
)
from brave.scenario import DATA_CENTER_ARCHETYPES, apply_global_heat_assumptions, make_study_scenarios
from brave.timeseries import apply_services_mode, make_hourly_profiles
from brave.metrics import service_revenue_scale


def _design_from_row(row: Dict) -> Dict[str, float]:
    return {
        "K_HX_per_MW_IT": float(row["K_HX_per_MW_IT_robust"]),
        "K_HP_per_MW_IT": float(row["K_HP_per_MW_IT_robust"]),
        "K_FLEX_per_MW_IT": float(row["K_FLEX_per_MW_IT_robust"]),
        "K_BAT_E_per_MW_IT": float(row["K_BAT_E_per_MW_IT_robust"]),
        "K_BAT_P_per_MW_IT": float(row["K_BAT_P_per_MW_IT_robust"]),
        "eff_reduction": 0.15,
    }


def _metric_rows(label: str, rows: List[Dict]) -> List[Dict]:
    output = []
    for row in rows:
        output.append({
            "TemporalRepresentation": label,
            "Archetype": row["Archetype"],
            "ScenarioID": row["ScenarioID"],
            "PUE": row["PUE_robust"],
            "siPUE": row["siPUE_robust"],
            "LifecycleCostNPVEUR": row["MarketCostNPVEUR_robust"],
            "RegretEUR": float(row["MarketCostNPVEUR_robust"]) - float(row["MarketCostNPVEUR_scenario_perfect"]),
            "EmissionsTCO2": row["EmissionsTCO2_robust"],
            "CongestionHours": row["CongestionHours_robust"],
            "RenewableCurtailmentMWh": row["RenewableCurtailmentMWh_robust"],
        })
    return output


def run_temporal_validation(smoke: bool = False) -> List[Dict]:
    counts = (7, 14) if smoke else (7, 14, 28, 56)
    all_rows: List[Dict] = []
    converged_rows: List[Dict] = []
    for count in counts:
        cfg = Config(
            T=24 if smoke else 8760,
            representative_day_count=count,
            use_representative_days_annual=not smoke,
            enforce_locked_calibration=False,
            export_csv=False,
            export_plots=False,
        )
        rows = run_single(cfg)
        all_rows.extend(_metric_rows(f"{count} representative days", rows))
        converged_rows = rows
    if smoke:
        return all_rows

    first = converged_rows[0]
    full_cfg = Config(
        T=8760,
        use_representative_days_annual=False,
        enforce_locked_calibration=False,
        export_csv=False,
        export_plots=False,
        auto_calibrate_line_limits=False,
        auto_calibrate_sipue_reference_price=False,
        line_cap_12=float(first["LineCap12MW_robust"]),
        line_cap_23=float(first["LineCap23MW_robust"]),
        line_cap_13=float(first["LineCap13MW_robust"]),
        sipue_system_ref_price_eur_per_mwh=float(first["ReferencePriceEURPerMWh"]),
    )
    study = make_study_scenarios(full_cfg)
    profiles = {}
    for group in study.values():
        apply_global_heat_assumptions(group, full_cfg)
        for sc in group:
            random.seed(full_cfg.seed + 10_009 * sc.uncertainty_id)
            profiles[sc.sid] = apply_services_mode(
                sc,
                make_hourly_profiles(sc, full_cfg.T, full_cfg),
                full_cfg,
            )
    full_rows = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, full_cfg)
        eval_cfg = replace(full_cfg, carbon_emissions_cap_tco2_by_sid=caps)
        source = next(row for row in converged_rows if row["ArchetypeKey"] == archetype.key)
        design = _design_from_row(source)
        for sc in group:
            ops = _solve_design_ops(sc, profiles[sc.sid], eval_cfg, design)
            _apply_emissions_attribution(ops, no_services[sc.sid])
            _rebase_lifecycle(no_services[sc.sid], no_services[sc.sid], full_cfg, is_reference=True)
            _rebase_lifecycle(ops, no_services[sc.sid], full_cfg)
            _apply_sipue(ops, no_services[sc.sid], full_cfg, service_revenue_scale(full_cfg, sc))
            full_rows.append({
                "TemporalRepresentation": "8760-hour fixed 56-day design",
                "Archetype": archetype.name,
                "ScenarioID": sc.uncertainty_id,
                "PUE": ops["PUE"],
                "siPUE": ops["siPUE"],
                "LifecycleCostNPVEUR": ops["market_planning_cost_npv_eur"],
                "RegretEUR": None,
                "EmissionsTCO2": ops["emissions_tco2"],
                "CongestionHours": ops["congestion_hours"],
                "RenewableCurtailmentMWh": ops["renewable_curtailment_mwh"],
            })
    all_rows.extend(full_rows)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = run_temporal_validation(smoke=args.smoke)
    suffix = "smoke" if args.smoke else "annual"
    _write_csv(rows, os.path.join("output", "congested3", f"temporal_validation_{suffix}.csv"))


if __name__ == "__main__":
    main()
