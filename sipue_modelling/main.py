"""
main.py – Thin re-export facade.

All implementation lives in the ``brave`` package.  This file re-exports every
public name so that ``from main import X`` continues to work for tests and the
sensitivity-analysis scripts.
"""
from __future__ import annotations

# stdlib re-exports used by the sensitivity script (mod.random)
import random  # noqa: F401

# ── brave.config ──────────────────────────────────────────────────────────────
from brave.config import (  # noqa: F401
    Config,
    LOCKED_CALIBRATION,
    _validate_locked_calibration,
)

# ── brave.scenario ────────────────────────────────────────────────────────────
from brave.scenario import (  # noqa: F401
    DATA_CENTER_ARCHETYPES,
    Scenario,
    make_archetype_scenarios,
    make_study_scenarios,
    make_out_of_sample_scenarios,
    make_scenarios,
    apply_global_heat_assumptions,
)

# ── brave.metrics ─────────────────────────────────────────────────────────────
from brave.metrics import (  # noqa: F401
    nanmean,
    spearman_rank_corr,
    compute_sipue,
    compute_land_m2,
    service_revenue_scale,
    land_cost_horizon,
    demand_charge_horizon,
    design_size_proxy,
    eff_capex_from_reduction,
    capex_scale,
)

# ── brave.solver ──────────────────────────────────────────────────────────────
from brave.solver import make_solver_lp  # noqa: F401

# ── brave.network ─────────────────────────────────────────────────────────────
from brave.network import (  # noqa: F401
    _ensure_network_defaults,
    _NETWORK_TS_CACHE,
    load_network_timeseries,
)

# ── brave.timeseries ──────────────────────────────────────────────────────────
from brave.timeseries import (  # noqa: F401
    build_representative_days,
    make_hourly_profiles,
    make_hourly_profiles_from_indices,
    make_overhead_alpha_base,
    apply_services_mode,
)

# ── brave.optimization ────────────────────────────────────────────────────────
from brave.optimization import (  # noqa: F401
    baseline_design_for_scenario,
    solve_ops_given_design,
    solve_perfect_foresight,
    solve_minmax_regret,
)

# ── brave.export ──────────────────────────────────────────────────────────────
from brave.export import (  # noqa: F401
    export_scenarios_csv,
    export_results_csv,
    summarize_interpretation,
)

from brave.validation import (  # noqa: F401
    compute_ere,
    add_derived_validation_metrics,
    build_validation_summary,
    build_ablation_summary,
    export_validation_tables,
    export_out_of_sample_tables,
)

# ── brave.plotting ────────────────────────────────────────────────────────────
from brave.plotting import (  # noqa: F401
    PLOT_FONT_SCALE,
    _fs,
    _pretty_type_label,
    export_plots,
    export_study_plots,
    export_creative_figure_package,
    export_creative_figure_package_from_csv,
    export_sensitivity_plots,
    export_regret_landscapes,
)

# ── brave.runners ─────────────────────────────────────────────────────────────
from brave.runners import (  # noqa: F401
    run_single,
    run_out_of_sample_validation,
    run_train_test_scenario_validation,
)


if __name__ == "__main__":
    cfg = Config()
    run_single(cfg)
