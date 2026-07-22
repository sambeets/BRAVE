from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from brave.config import Config
from brave.runners import run_primary_design


def _scale_capex(cfg: Config, factor: float) -> Config:
    return replace(
        cfg,
        capex_HX=cfg.capex_HX * factor,
        capex_HP=cfg.capex_HP * factor,
        capex_FLEX=cfg.capex_FLEX * factor,
        capex_BAT_E=cfg.capex_BAT_E * factor,
        capex_BAT_P=cfg.capex_BAT_P * factor,
        eff_capex_levels=tuple(value * factor for value in cfg.eff_capex_levels),
    )


def _scale_land_footprint(cfg: Config, factor: float) -> Config:
    return replace(
        cfg,
        existing_land_m2_per_it_mw=cfg.existing_land_m2_per_it_mw * factor,
        land_a_cooling=cfg.land_a_cooling * factor,
        land_a_hx=cfg.land_a_hx * factor,
        land_a_hp=cfg.land_a_hp * factor,
        land_a_flex=cfg.land_a_flex * factor,
        land_a_bat_e=cfg.land_a_bat_e * factor,
        land_a_bat_p=cfg.land_a_bat_p * factor,
    )


def sensitivity_cases(cfg: Config) -> List[Tuple[str, str, Config]]:
    cases = [("canonical", "base", cfg)]
    cases.extend([
        ("ETS price", "low", replace(cfg, co2_timeseries_scale=0.5)),
        ("ETS price", "high", replace(cfg, co2_timeseries_scale=1.5)),
        ("carbon damage", "low", replace(cfg, carbon_damage_multiplier=0.5)),
        ("carbon damage", "high", replace(cfg, carbon_damage_multiplier=1.5)),
        ("carbon cap", "0%", replace(cfg, carbon_cap_reduction_frac=0.0)),
        ("carbon cap", "2.5%", replace(cfg, carbon_cap_reduction_frac=0.025)),
        ("carbon cap", "10%", replace(cfg, carbon_cap_reduction_frac=0.10)),
        ("heat price", "low", replace(cfg, heat_price_multiplier=cfg.heat_price_multiplier * 0.5)),
        ("heat price", "high", replace(cfg, heat_price_multiplier=cfg.heat_price_multiplier * 1.5)),
        ("heat system value", "low", replace(cfg, heat_system_value_multiplier=0.5)),
        ("heat system value", "high", replace(cfg, heat_system_value_multiplier=1.5)),
        ("heat demand", "low", replace(cfg, heat_demand_multiplier=0.5)),
        ("heat demand", "high", replace(cfg, heat_demand_multiplier=1.5)),
        ("flexibility price", "low", replace(cfg, flex_price_multiplier=cfg.flex_price_multiplier * 0.5)),
        ("flexibility price", "high", replace(cfg, flex_price_multiplier=cfg.flex_price_multiplier * 1.5)),
        ("flexibility system value", "low", replace(cfg, flex_system_value_multiplier=0.5)),
        ("flexibility system value", "high", replace(cfg, flex_system_value_multiplier=1.5)),
        ("flexibility availability", "low", replace(cfg, flex_availability_multiplier=0.5)),
        ("flexibility availability", "high", replace(cfg, flex_availability_multiplier=1.5)),
        ("cooling overhead", "low", replace(cfg, cooling_overhead_multiplier=0.8)),
        ("cooling overhead", "high", replace(cfg, cooling_overhead_multiplier=1.2)),
        ("free cooling", "low", replace(cfg, nordic_night_overhead_drop=0.10)),
        ("free cooling", "high", replace(cfg, nordic_night_overhead_drop=0.40)),
        ("heat recovery", "low", replace(cfg, override_heat_recovery_frac=0.30)),
        ("heat recovery", "high", replace(cfg, override_heat_recovery_frac=0.50)),
        ("heat-pump COP", "low", replace(cfg, override_hp_cop=2.2)),
        ("heat-pump COP", "high", replace(cfg, override_hp_cop=5.0)),
        ("storage", "low", replace(
            cfg,
            enable_thermal_storage=False,
            K_BAT_E_max=0.0,
            K_BAT_P_max=0.0,
            min_bat_e_hours=0.0,
            min_bat_p_mw_per_it_mw=0.0,
            baseline_bat_e_hours=0.0,
            baseline_bat_p_mw_per_it_mw=0.0,
        )),
        ("storage", "high", replace(cfg, K_BAT_E_max=cfg.K_BAT_E_max * 1.5, K_BAT_P_max=cfg.K_BAT_P_max * 1.5)),
        ("congestion", "low", replace(cfg, line_capacity_multiplier=1.5)),
        ("congestion", "high", replace(cfg, line_capacity_multiplier=0.6)),
        ("peak and grid charges", "low", replace(cfg, demand_charge_eur_per_mw_year=cfg.demand_charge_eur_per_mw_year * 0.5, independent_peak_capacity_value_eur_per_mw_year=cfg.independent_peak_capacity_value_eur_per_mw_year * 0.5, transmission_cost_eur_per_mwh=cfg.transmission_cost_eur_per_mwh * 0.5)),
        ("peak and grid charges", "high", replace(cfg, demand_charge_eur_per_mw_year=cfg.demand_charge_eur_per_mw_year * 1.5, independent_peak_capacity_value_eur_per_mw_year=cfg.independent_peak_capacity_value_eur_per_mw_year * 1.5, transmission_cost_eur_per_mwh=cfg.transmission_cost_eur_per_mwh * 1.5)),
        ("capex", "low", _scale_capex(cfg, 0.5)),
        ("capex", "high", _scale_capex(cfg, 1.5)),
        ("land", "low", replace(cfg, land_cost_eur_per_m2=cfg.land_cost_eur_per_m2 * 0.5)),
        ("land", "high", replace(cfg, land_cost_eur_per_m2=cfg.land_cost_eur_per_m2 * 2.0)),
        ("land footprint", "low", _scale_land_footprint(cfg, 0.5)),
        ("land footprint", "high", _scale_land_footprint(cfg, 1.5)),
        ("land availability", "low", replace(cfg, land_limit_multiplier=0.65)),
        ("land availability", "high", replace(cfg, land_limit_multiplier=1.2)),
        ("land availability", "unconstrained", replace(cfg, land_limit_multiplier=100.0)),
        ("construction delay", "low", replace(cfg, override_construction_delay_months=0.0)),
        ("construction delay", "high", replace(cfg, override_construction_delay_months=12.0)),
        ("siPUE reference price", "low", replace(cfg, sipue_reference_price_multiplier=0.5)),
        ("siPUE reference price", "high", replace(cfg, sipue_reference_price_multiplier=1.5)),
    ])
    return cases


def _write(rows: Iterable[Dict], path: str) -> None:
    data = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(data[0].keys())
    extras = sorted({key for row in data for key in row} - set(fields))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extras)
        writer.writeheader()
        writer.writerows(data)


def _mean(rows: List[Dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(factor: str, level: str, rows: List[Dict], archetype: str) -> Dict:
    suffix = "robust"
    return {
        "Factor": factor,
        "Level": level,
        "Archetype": archetype,
        "ScenarioCount": len(rows),
        "AverageMarketCostNPVEUR": _mean(rows, f"MarketCostNPVEUR_{suffix}"),
        "AverageSystemBenefitEUR": _mean(rows, f"SystemBenefitEUR_{suffix}"),
        "AverageSystemBenefitPerMWhIT": _mean(rows, f"SystemBenefitPerMWhIT_{suffix}"),
        "AverageRegretEUR": sum(
            float(row[f"MarketCostNPVEUR_{suffix}"]) - float(row["MarketCostNPVEUR_scenario_perfect"])
            for row in rows
        ) / len(rows),
        "AveragePUE": _mean(rows, f"PUE_{suffix}"),
        "AverageSiPUE": _mean(rows, f"siPUE_{suffix}"),
        "AverageSteadySiPUE": _mean(rows, f"SteadySiPUE_{suffix}"),
        "AverageCUE": _mean(rows, f"CUE_{suffix}"),
        "AverageEmissionsTCO2": _mean(rows, f"EmissionsTCO2_{suffix}"),
        "AverageCarbonMarketCostEUR": _mean(rows, f"CarbonMarketCostEUR_{suffix}"),
        "AverageCarbonDamageEUR": _mean(rows, f"CarbonDamageEUR_{suffix}"),
        "AverageCoolingMWh": _mean(rows, f"CoolingMWh_{suffix}"),
        "AverageRecoveredHeatITMWhth": _mean(rows, f"RecoveredHeatITMWhth_{suffix}"),
        "AverageRecoveredHeatCoolingMWhth": _mean(rows, f"RecoveredHeatCoolingMWhth_{suffix}"),
        "AverageLifecycleValueLossEUR": _mean(rows, f"LifecycleValueLossEUR_{suffix}"),
        "AverageLandM2": _mean(rows, f"LandM2_{suffix}"),
        "AverageIntegrationLandM2": _mean(rows, f"IntegrationLandM2_{suffix}"),
        "AverageLandUtilizationPct": _mean(rows, f"LandUtilizationPct_{suffix}"),
        "AverageLandLeaseNPVEUR": _mean(rows, f"LandCostNPVEUR_{suffix}"),
        "BindingLandScenarioCount": sum(bool(row[f"LandConstraintBinding_{suffix}"]) for row in rows),
        "K_HX_MWth": _mean(rows, f"K_HX_MWth_{suffix}"),
        "K_HP_MW": _mean(rows, f"K_HP_MW_{suffix}"),
        "K_FLEX_MW": _mean(rows, f"K_FLEX_MW_{suffix}"),
        "K_BAT_E_MWh": _mean(rows, f"K_BAT_E_MWh_{suffix}"),
        "K_BAT_P_MW": _mean(rows, f"K_BAT_P_MW_{suffix}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use a 24-hour decision-support horizon.")
    args = parser.parse_args()
    cfg = Config(
        T=24 if args.smoke else 8760,
        use_representative_days_annual=not args.smoke,
        enforce_locked_calibration=False,
        export_csv=False,
        export_plots=False,
        export_regret_landscape=False,
    )
    detailed = []
    summary = []
    for factor, level, case_cfg in sensitivity_cases(cfg):
        rows = run_primary_design(case_cfg)
        if len(rows) != 40:
            raise RuntimeError(f"Sensitivity case {factor}/{level} returned {len(rows)} rows.")
        detailed.extend({"Factor": factor, "Level": level, **row} for row in rows)
        summary.append(summarize(factor, level, rows, "all"))
        for archetype in sorted({str(row["Archetype"]) for row in rows}):
            subset = [row for row in rows if row["Archetype"] == archetype]
            summary.append(summarize(factor, level, subset, archetype))
    tag = "smoke" if args.smoke else "annual"
    _write(detailed, os.path.join("output", "congested3", f"sensitivity_{tag}_4x10.csv"))
    _write(summary, os.path.join("output", "congested3", f"sensitivity_{tag}_summary.csv"))


if __name__ == "__main__":
    main()
