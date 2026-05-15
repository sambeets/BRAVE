"""Capture golden values from the current implementation.

Run once:  python tests/capture_golden.py
"""
import json
import os
import sys
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import (
    Config, make_scenarios, apply_global_heat_assumptions,
    make_hourly_profiles, apply_services_mode,
    _ensure_network_defaults, _NETWORK_TS_CACHE,
    baseline_design_for_scenario, solve_ops_given_design,
    solve_perfect_foresight,
)

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
os.makedirs(GOLDEN_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "network_3node_timeseries.csv")


def _serializable(v):
    """Convert value to JSON-serializable form."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v


def _scalar_dict(d):
    """Extract only scalar (int/float/str/bool) entries from a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = _serializable(v)
    return out


def main():
    cfg = Config(
        T=24,
        seed=42,
        enforce_locked_calibration=False,
        solver_msg=False,
        export_csv=False,
        export_plots=False,
        use_representative_days_annual=False,
        network_timeseries_csv=CSV_PATH,
    )
    _NETWORK_TS_CACHE.clear()
    _ensure_network_defaults(cfg)

    all_sc = make_scenarios(cfg)
    apply_global_heat_assumptions(all_sc, cfg)
    sc1 = all_sc[0]  # Hyperscale-Netherlands

    # --- solve_ops with baseline design ---
    random.seed(42)
    prof1 = make_hourly_profiles(sc1, cfg.T, cfg)
    prof1 = apply_services_mode(sc1, prof1, cfg)

    k_hx, k_flex, k_bat_e, k_bat_p, eff = baseline_design_for_scenario(sc1, prof1, cfg)
    ops = solve_ops_given_design(sc1, prof1, cfg, k_hx, k_flex, k_bat_e, k_bat_p, eff)
    scalars = _scalar_dict(ops)
    path_ops = os.path.join(GOLDEN_DIR, "solve_ops_scenario1_24h.json")
    with open(path_ops, "w") as f:
        json.dump(scalars, f, indent=2)
    print(f"Wrote {path_ops} ({len(scalars)} keys)")

    # --- solve_perfect_foresight ---
    random.seed(42)
    prof1_pf = make_hourly_profiles(sc1, cfg.T, cfg)
    prof1_pf = apply_services_mode(sc1, prof1_pf, cfg)
    cstar, design = solve_perfect_foresight(sc1, prof1_pf, cfg)
    design_out = {"cstar": cstar, **_scalar_dict(design)}
    path_pf = os.path.join(GOLDEN_DIR, "solve_pf_scenario1_24h.json")
    with open(path_pf, "w") as f:
        json.dump(design_out, f, indent=2)
    print(f"Wrote {path_pf} ({len(design_out)} keys)")

    print("\nGolden values captured successfully.")


if __name__ == "__main__":
    main()
