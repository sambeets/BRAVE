from __future__ import annotations

"""Carbon-cap-relaxed ablation for the manuscript diagnostic.

The canonical baseline and robust designs use a shared physical carbon cap
applied to the fair services-disabled comparator and every service-active
design. This script re-solves the full two-stage design and the baseline
design, per archetype, with that physical cap removed while keeping the
carbon-damage price term unchanged. Joining the cap-relaxed designs against
the canonical cap-enforced designs isolates the compliance-constraint effect
from the price-driven carbon-damage valuation.

Writes output/congested3/carbon_cap_relaxed_ablation.csv.
"""

import csv
import os
from dataclasses import replace
from typing import Dict, Optional, Tuple

from brave.config import Config
from brave.metrics import service_revenue_scale
from brave.optimization import solve_minmax_regret
from brave.runners import (
    _apply_emissions_attribution,
    _apply_sipue,
    _baseline_ops,
    _carbon_reference,
    _fair_services_disabled_ops,
    _perfect_designs,
    _prepare_profiles,
    _realize_design,
    _rebase_lifecycle,
    _solve_design_ops,
    _write_csv,
)
from brave.scenario import DATA_CENTER_ARCHETYPES, apply_global_heat_assumptions, make_study_scenarios

CANONICAL_RESULTS_CSV = "output/congested3/results_4_archetypes_10_states.csv"
OUT_PATH = "output/congested3/carbon_cap_relaxed_ablation.csv"


def _load_canonical(path: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {(row["archetype_key"], int(row["scenario_id"])): row for row in csv.DictReader(handle)}


def _canon_float(row: Dict[str, str], name: str) -> Optional[float]:
    value = row.get(name, "")
    return float(value) if value not in ("", None) else None


def main() -> None:
    cfg = Config(export_plots=False, canonical_output_dir="output/congested3/carbon-cap-relaxed")
    canonical = _load_canonical(CANONICAL_RESULTS_CSV)

    study = make_study_scenarios(cfg)
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    scenarios = [sc for group in study.values() for sc in group]
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)

    rows = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        relaxed_cfg = replace(cfg_local, enforce_carbon_cap=False, carbon_emissions_cap_tco2_by_sid=None)

        c_star_relaxed, _ = _perfect_designs(group, profiles, relaxed_cfg)
        robust_relaxed = solve_minmax_regret(group, profiles, c_star_relaxed, relaxed_cfg)

        for sc in group:
            prof = profiles[sc.sid]
            reference = _fair_services_disabled_ops(sc, prof, relaxed_cfg)
            baseline = _baseline_ops(sc, prof, relaxed_cfg)
            robust = _solve_design_ops(sc, prof, relaxed_cfg, robust_relaxed)

            for ops in (reference, baseline, robust):
                _apply_emissions_attribution(ops, no_services[sc.sid])
            _rebase_lifecycle(reference, reference, cfg_local, is_reference=True)
            _rebase_lifecycle(baseline, reference, cfg_local)
            _rebase_lifecycle(robust, reference, cfg_local)

            availability = service_revenue_scale(cfg_local, sc)
            _apply_sipue(reference, reference, cfg_local, 1.0)
            _apply_sipue(baseline, reference, cfg_local, availability)
            _apply_sipue(robust, reference, cfg_local, availability)

            canon_row = canonical.get((archetype.key, sc.uncertainty_id), {})
            realized_relaxed = _realize_design(robust_relaxed, sc)
            rows.append({
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "e_it_mwh": robust["E_IT"],
                "carbon_cap_tco2_enforced": caps[sc.sid],
                "pue_fair_services_disabled_cap_relaxed": reference["PUE"],
                "sipue_fair_services_disabled_cap_relaxed": reference["siPUE"],
                "pue_baseline_cap_relaxed": baseline["PUE"],
                "sipue_baseline_cap_relaxed": baseline["siPUE"],
                "pue_robust_cap_relaxed": robust["PUE"],
                "sipue_robust_cap_relaxed": robust["siPUE"],
                "emissions_tco2_baseline_cap_relaxed": baseline["emissions_tco2"],
                "emissions_tco2_robust_cap_relaxed": robust["emissions_tco2"],
                "carbon_damage_eur_baseline_cap_relaxed": baseline["independent_carbon_damage_eur"],
                "carbon_damage_eur_robust_cap_relaxed": robust["independent_carbon_damage_eur"],
                "k_hx_mwth_robust_cap_relaxed": realized_relaxed["K_HX"],
                "k_hp_mw_robust_cap_relaxed": realized_relaxed["K_HP"],
                "k_flex_mw_robust_cap_relaxed": realized_relaxed["K_FLEX"],
                "k_bat_e_mwh_robust_cap_relaxed": realized_relaxed["K_BAT_E"],
                "k_bat_p_mw_robust_cap_relaxed": realized_relaxed["K_BAT_P"],
                "pue_baseline_cap_enforced": _canon_float(canon_row, "pue_baseline"),
                "sipue_baseline_cap_enforced": _canon_float(canon_row, "sipue_baseline"),
                "pue_robust_cap_enforced": _canon_float(canon_row, "pue_robust"),
                "sipue_robust_cap_enforced": _canon_float(canon_row, "sipue_robust"),
                "emissions_tco2_baseline_cap_enforced": _canon_float(canon_row, "emissions_tco2_baseline"),
                "emissions_tco2_robust_cap_enforced": _canon_float(canon_row, "emissions_tco2_robust"),
                "carbon_damage_eur_baseline_cap_enforced": _canon_float(canon_row, "carbon_damage_eur_baseline"),
                "carbon_damage_eur_robust_cap_enforced": _canon_float(canon_row, "carbon_damage_eur_robust"),
                "k_hx_mwth_robust_cap_enforced": _canon_float(canon_row, "k_hx_mwth_robust"),
                "k_hp_mw_robust_cap_enforced": _canon_float(canon_row, "k_hp_mw_robust"),
                "k_flex_mw_robust_cap_enforced": _canon_float(canon_row, "k_flex_mw_robust"),
                "k_bat_e_mwh_robust_cap_enforced": _canon_float(canon_row, "k_bat_e_mwh_robust"),
                "k_bat_p_mw_robust_cap_enforced": _canon_float(canon_row, "k_bat_p_mw_robust"),
            })

    _write_csv(rows, OUT_PATH)

    def _it_weighted(field: str) -> float:
        total_it = sum(r["e_it_mwh"] for r in rows)
        return sum(r[field] * r["e_it_mwh"] for r in rows) / total_it

    print(f"Wrote {len(rows)} rows to {OUT_PATH}\n")
    print("IT-weighted PUE / siPUE, cap-enforced (canonical) vs cap-relaxed (this ablation):")
    for design in ("baseline", "robust"):
        print(
            f"  {design:8s} PUE   {_it_weighted(f'pue_{design}_cap_enforced'):.3f} -> "
            f"{_it_weighted(f'pue_{design}_cap_relaxed'):.3f}"
        )
        print(
            f"  {design:8s} siPUE {_it_weighted(f'sipue_{design}_cap_enforced'):.3f} -> "
            f"{_it_weighted(f'sipue_{design}_cap_relaxed'):.3f}"
        )
        below_enforced = sum(
            1 for r in rows if r[f"sipue_{design}_cap_enforced"] < r["sipue_fair_services_disabled_cap_relaxed"]
        )
        below_relaxed = sum(
            1 for r in rows if r[f"sipue_{design}_cap_relaxed"] < r["sipue_fair_services_disabled_cap_relaxed"]
        )
        print(f"  {design:8s} rows below comparator: {below_enforced}/{len(rows)} enforced, {below_relaxed}/{len(rows)} relaxed")

    print("\nRobust shared design capacity (per MW IT), cap-enforced vs cap-relaxed, by archetype:")
    seen = set()
    for r in rows:
        if r["archetype_key"] in seen:
            continue
        seen.add(r["archetype_key"])
        for cap_name, unit in (("k_hx_mwth", "MWth"), ("k_hp_mw", "MW"), ("k_flex_mw", "MW"),
                                ("k_bat_e_mwh", "MWh"), ("k_bat_p_mw", "MW")):
            enforced = r[f"{cap_name}_robust_cap_enforced"]
            relaxed = r[f"{cap_name}_robust_cap_relaxed"]
            print(f"  {r['archetype']:16s} {cap_name:12s} {enforced:.4f} -> {relaxed:.4f} {unit}/MW_IT")


if __name__ == "__main__":
    main()
