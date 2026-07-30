"""Run the manuscript sensitivity set on the primary design workflow.

This diagnostic varies only the approved factors used for the current
manuscript sensitivity appendix: heat price, flexibility price, cooling
overhead, land availability, and construction delay. It intentionally does not
include the old battery-ceiling, congestion, capex, storage, carbon-cap, or
reference-price sweeps.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from brave.config import Config
from brave.runners import run_primary_design


SensitivityCase = Tuple[str, str, str, float, str, Config]


def sensitivity_cases(cfg: Config) -> List[SensitivityCase]:
    heat_low = round(cfg.heat_price_multiplier * 0.5, 10)
    heat_high = round(cfg.heat_price_multiplier * 1.5, 10)
    flex_low = round(cfg.flex_price_multiplier * 0.5, 10)
    flex_high = round(cfg.flex_price_multiplier * 1.5, 10)
    cases: List[SensitivityCase] = [
        ("canonical", "base", "locked canonical configuration", 1.0, "baseline", cfg)
    ]
    cases.extend([
        (
            "heat price",
            "low",
            "heat_price_multiplier",
            heat_low,
            "global multiplier on scenario heat price",
            replace(cfg, heat_price_multiplier=heat_low),
        ),
        (
            "heat price",
            "high",
            "heat_price_multiplier",
            heat_high,
            "global multiplier on scenario heat price",
            replace(cfg, heat_price_multiplier=heat_high),
        ),
        (
            "flexibility price",
            "low",
            "flex_price_multiplier",
            flex_low,
            "global multiplier on scenario flexibility price",
            replace(cfg, flex_price_multiplier=flex_low),
        ),
        (
            "flexibility price",
            "high",
            "flex_price_multiplier",
            flex_high,
            "global multiplier on scenario flexibility price",
            replace(cfg, flex_price_multiplier=flex_high),
        ),
        (
            "cooling overhead",
            "low",
            "cooling_overhead_multiplier",
            0.8,
            "multiplier of scenario cooling overhead",
            replace(cfg, cooling_overhead_multiplier=0.8),
        ),
        (
            "cooling overhead",
            "high",
            "cooling_overhead_multiplier",
            1.2,
            "multiplier of scenario cooling overhead",
            replace(cfg, cooling_overhead_multiplier=1.2),
        ),
        (
            "land availability",
            "low",
            "land_limit_multiplier",
            0.65,
            "multiplier of archetype land limit",
            replace(cfg, land_limit_multiplier=0.65),
        ),
        (
            "land availability",
            "high",
            "land_limit_multiplier",
            1.2,
            "multiplier of archetype land limit",
            replace(cfg, land_limit_multiplier=1.2),
        ),
        (
            "land availability",
            "unconstrained",
            "land_limit_multiplier",
            100.0,
            "multiplier of archetype land limit",
            replace(cfg, land_limit_multiplier=100.0),
        ),
        (
            "construction delay",
            "low",
            "override_construction_delay_months",
            0.0,
            "months",
            replace(cfg, override_construction_delay_months=0.0),
        ),
        (
            "construction delay",
            "high",
            "override_construction_delay_months",
            12.0,
            "months",
            replace(cfg, override_construction_delay_months=12.0),
        ),
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


def summarize(
    factor: str,
    level: str,
    setting: str,
    value: float,
    unit: str,
    rows: List[Dict],
    archetype: str,
) -> Dict:
    suffix = "robust"
    return {
        "factor": factor,
        "level": level,
        "sensitivity_setting": setting,
        "sensitivity_value": value,
        "sensitivity_unit": unit,
        "archetype": archetype,
        "scenario_count": len(rows),
        "average_market_cost_npv_eur": _mean(rows, f"market_cost_npv_eur_{suffix}"),
        "average_system_benefit_eur": _mean(rows, f"system_benefit_eur_{suffix}"),
        "average_system_benefit_per_mwh_it": _mean(rows, f"system_benefit_per_mwh_it_{suffix}"),
        "average_regret_eur": sum(
            float(row[f"market_cost_npv_eur_{suffix}"]) - float(row["market_cost_npv_eur_scenario_perfect"])
            for row in rows
        ) / len(rows),
        "average_pue": _mean(rows, f"pue_{suffix}"),
        "average_sipue": _mean(rows, f"sipue_{suffix}"),
        "average_steady_sipue": _mean(rows, f"steady_sipue_{suffix}"),
        "average_cue": _mean(rows, f"cue_{suffix}"),
        "average_emissions_tco2": _mean(rows, f"emissions_tco2_{suffix}"),
        "average_carbon_market_cost_eur": _mean(rows, f"carbon_market_cost_eur_{suffix}"),
        "average_carbon_damage_eur": _mean(rows, f"carbon_damage_eur_{suffix}"),
        "average_cooling_mwh": _mean(rows, f"cooling_mwh_{suffix}"),
        "average_recovered_heat_it_mwhth": _mean(rows, f"recovered_heat_it_mwhth_{suffix}"),
        "average_recovered_heat_cooling_mwhth": _mean(rows, f"recovered_heat_cooling_mwhth_{suffix}"),
        "average_lifecycle_value_loss_eur": _mean(rows, f"lifecycle_value_loss_eur_{suffix}"),
        "average_land_m2": _mean(rows, f"land_m2_{suffix}"),
        "average_integration_land_m2": _mean(rows, f"integration_land_m2_{suffix}"),
        "average_land_utilization_pct": _mean(rows, f"land_utilization_pct_{suffix}"),
        "average_land_lease_npv_eur": _mean(rows, f"land_cost_npv_eur_{suffix}"),
        "binding_land_scenario_count": sum(bool(row[f"land_constraint_binding_{suffix}"]) for row in rows),
        "k_hx_mwth": _mean(rows, f"k_hx_mwth_{suffix}"),
        "k_hp_mw": _mean(rows, f"k_hp_mw_{suffix}"),
        "k_flex_mw": _mean(rows, f"k_flex_mw_{suffix}"),
        "k_bat_e_mwh": _mean(rows, f"k_bat_e_mwh_{suffix}"),
        "k_bat_p_mw": _mean(rows, f"k_bat_p_mw_{suffix}"),
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
    for factor, level, setting, value, unit, case_cfg in sensitivity_cases(cfg):
        rows = run_primary_design(case_cfg)
        if len(rows) != 40:
            raise RuntimeError(f"Sensitivity case {factor}/{level} returned {len(rows)} rows.")
        detailed.extend({
            "factor": factor,
            "level": level,
            "sensitivity_setting": setting,
            "sensitivity_value": value,
            "sensitivity_unit": unit,
            **row,
        } for row in rows)
        summary.append(summarize(factor, level, setting, value, unit, rows, "all"))
        for archetype in sorted({str(row["archetype"]) for row in rows}):
            subset = [row for row in rows if row["archetype"] == archetype]
            summary.append(summarize(factor, level, setting, value, unit, subset, archetype))
    tag = "smoke" if args.smoke else "annual"
    _write(detailed, os.path.join("output", "congested3", f"sensitivity_{tag}_4x10.csv"))
    _write(summary, os.path.join("output", "congested3", f"sensitivity_{tag}_summary.csv"))


if __name__ == "__main__":
    main()
