import pytest

from brave.config import Config
from brave.optimization import solve_perfect_foresight, solve_minmax_regret
from brave.runners import run_sipue_objective_comparison


class TestObjectiveModeValidation:
    """Fast, no-solve checks that the new objective_mode branch validates its
    own required inputs instead of silently building an incomplete model."""

    def test_solve_perfect_foresight_rejects_unknown_mode(self, cfg_test, scenario_hyperscale, profile_24h):
        with pytest.raises(ValueError):
            solve_perfect_foresight(scenario_hyperscale, profile_24h, cfg_test, objective_mode="bogus")

    def test_solve_perfect_foresight_sipue_mode_requires_reference_cost(self, cfg_test, scenario_hyperscale, profile_24h):
        with pytest.raises(ValueError):
            solve_perfect_foresight(scenario_hyperscale, profile_24h, cfg_test, objective_mode="sipue")

    def test_solve_minmax_regret_rejects_unknown_mode(self, cfg_test, scenario_hyperscale, profile_24h):
        with pytest.raises(ValueError):
            solve_minmax_regret(
                [scenario_hyperscale], {scenario_hyperscale.sid: profile_24h},
                {scenario_hyperscale.sid: 0.0}, cfg_test, objective_mode="bogus",
            )

    def test_solve_minmax_regret_sipue_mode_requires_benchmarks(self, cfg_test, scenario_hyperscale, profile_24h):
        with pytest.raises(ValueError):
            solve_minmax_regret(
                [scenario_hyperscale], {scenario_hyperscale.sid: profile_24h},
                {scenario_hyperscale.sid: 0.0}, cfg_test, objective_mode="sipue",
            )


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sipue_objective_rows():
    """Small-horizon (T=24) run of both Stage-2 objective modes side by side."""
    return run_sipue_objective_comparison(Config(
        T=24,
        use_representative_days_annual=False,
        enforce_locked_calibration=False,
        export_csv=False,
        export_plots=False,
    ))


def test_covers_four_archetypes_and_ten_shared_states(sipue_objective_rows):
    assert len(sipue_objective_rows) == 40
    counts = {}
    for row in sipue_objective_rows:
        counts[row["archetype_key"]] = counts.get(row["archetype_key"], 0) + 1
    assert counts == {
        "ai_hpc": 10,
        "hyperscale": 10,
        "colocation": 10,
        "enterprise_edge": 10,
    }


def test_both_designs_feasible_and_finite(sipue_objective_rows):
    for row in sipue_objective_rows:
        for suffix in ("cost_regret_robust", "sipue_regret_robust"):
            sipue = float(row[f"sipue_{suffix}"])
            assert sipue == sipue  # not NaN
            assert sipue > 0.0
            cost = float(row[f"market_cost_npv_eur_{suffix}"])
            assert cost == cost


def test_sipue_regret_worst_case_group_regret_not_materially_worse(sipue_objective_rows):
    """Stage 2 optimizes the *group-level worst case* (max over the shared
    10-scenario uncertainty set), not any individual scenario's siPUE. A
    design chosen to directly minimize worst-case siPUE regret can therefore
    still show a higher siPUE than the cost-regret design on some individual
    off-worst-case scenario -- that is not a bug. What must hold is that,
    within each archetype's own scenario group, the siPUE-regret design's
    worst-case regret against the scenario-perfect siPUE benchmark is not
    materially worse than the cost-regret design's. A small tolerance is
    used because both designs are solved via a separate lexicographic
    min-max LP at full representative-day scale (objective values of the
    order of tens of billions of EUR), where solver optimality-gap noise is
    non-negligible relative to siPUE, which is O(1)."""
    by_archetype = {}
    for row in sipue_objective_rows:
        by_archetype.setdefault(row["archetype_key"], []).append(row)

    tolerance = 0.01  # siPUE units
    for archetype_key, group in by_archetype.items():
        regret_cost = max(
            float(r["sipue_cost_regret_robust"]) - float(r["sipue_scenario_perfect_star"])
            for r in group
        )
        regret_sipue = max(
            float(r["sipue_sipue_regret_robust"]) - float(r["sipue_scenario_perfect_star"])
            for r in group
        )
        assert regret_sipue <= regret_cost + tolerance, (
            f"{archetype_key}: sipue-regret worst-case regret {regret_sipue:.5f} exceeds "
            f"cost-regret worst-case regret {regret_cost:.5f} by more than tolerance"
        )


def test_cost_difference_between_objectives_is_small(sipue_objective_rows):
    """The two Stage-2 objectives are expected to select the same or a
    near-identical design in practice (see paper.tex, siPUE-objective
    comparison discussion). A large lifecycle-cost divergence between them
    would indicate a modeling bug rather than the expected near-equivalence,
    so this is a coarse sanity bound, not a tight theoretical guarantee."""
    for row in sipue_objective_rows:
        cost_a = float(row["market_cost_npv_eur_cost_regret_robust"])
        cost_b = float(row["market_cost_npv_eur_sipue_regret_robust"])
        pct_diff = 100.0 * abs(cost_b - cost_a) / max(1.0, abs(cost_a))
        assert pct_diff < 5.0, (
            f"{row['archetype']} {row['scenario']}: lifecycle cost differs by {pct_diff:.2f}% "
            f"between the cost-regret and siPUE-regret designs"
        )
