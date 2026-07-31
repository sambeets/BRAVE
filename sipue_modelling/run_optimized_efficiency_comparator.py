from __future__ import annotations

"""Optimized-efficiency services-disabled comparator (supplementary diagnostic).

The paper's reference services-disabled comparator locks the efficiency
retrofit at the same 0.15 used by the service-active cases, to isolate the
effect of adding services from the effect of the retrofit itself (a
"design-matched" comparator). This script instead lets a services-disabled
design choose its own efficiency level freely from cfg.eff_levels (0.00 or
0.15), representing the best conventional (non-integrated) alternative an
operator could independently choose (an "optimized-efficiency" comparator).

Writes output/congested3/optimized_efficiency_comparator.csv
"""

import os
from dataclasses import replace

from brave.config import Config
from brave.scenario import make_study_scenarios, apply_global_heat_assumptions, DATA_CENTER_ARCHETYPES
from brave.runners import _prepare_profiles, _carbon_reference, _no_services_profile, _write_csv
from brave.optimization import solve_perfect_foresight


def main() -> None:
    cfg = Config()
    study = make_study_scenarios(cfg)
    for group in study.values():
        apply_global_heat_assumptions(group, cfg)
    scenarios = [sc for group in study.values() for sc in group]
    cfg_local, profiles = _prepare_profiles(cfg, scenarios)

    rows = []
    for archetype in DATA_CENTER_ARCHETYPES:
        group = study[archetype.key]
        no_services, caps = _carbon_reference(group, profiles, cfg_local)
        constrained_cfg = replace(cfg_local, carbon_emissions_cap_tco2_by_sid=caps)
        for sc in group:
            prof = profiles[sc.sid]
            locked_cfg = replace(constrained_cfg, fixed_eff_reduction=0.15)
            free_cfg = replace(constrained_cfg, fixed_eff_reduction=None)

            _, locked_design = solve_perfect_foresight(sc, _no_services_profile(prof), locked_cfg)
            obj_free, free_design = solve_perfect_foresight(sc, _no_services_profile(prof), free_cfg)

            rows.append({
                "archetype": archetype.name,
                "archetype_key": archetype.key,
                "scenario": sc.name,
                "scenario_id": sc.uncertainty_id,
                "locked_eff_reduction": locked_design["eff_reduction"],
                "free_eff_reduction": free_design["eff_reduction"],
                "free_eff_matches_locked": free_design["eff_reduction"] == locked_design["eff_reduction"],
                "free_design_objective_cost_npv_eur": obj_free,
            })
            print(
                archetype.key, sc.name,
                "locked=", locked_design["eff_reduction"],
                "free=", free_design["eff_reduction"],
            )

    out_path = os.path.join(cfg.canonical_output_dir, "optimized_efficiency_comparator.csv")
    _write_csv(rows, out_path)
    n_match = sum(1 for r in rows if r["free_eff_matches_locked"])
    print(f"\n{n_match} of {len(rows)} rows: freely-optimized efficiency matches the locked 0.15 retrofit.")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
