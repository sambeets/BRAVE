import pytest

from brave.config import Config
from brave.runners import (
    DESIGN_CASES,
    _conclusion_summary,
    _kpi_benchmark_summary,
    _reduction_pct,
    run_single,
)


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def canonical_smoke_rows():
    return run_single(Config(
        T=24,
        use_representative_days_annual=False,
        enforce_locked_calibration=False,
        export_csv=False,
        export_plots=False,
    ))


def test_all_main_cases_use_four_archetypes_and_ten_shared_states(canonical_smoke_rows):
    assert len(canonical_smoke_rows) == 40
    assert {row["scenario_id"] for row in canonical_smoke_rows} == set(range(1, 11))
    counts = {}
    for row in canonical_smoke_rows:
        counts[row["archetype_key"]] = counts.get(row["archetype_key"], 0) + 1
    assert counts == {
        "ai_hpc": 10,
        "hyperscale": 10,
        "colocation": 10,
        "enterprise_edge": 10,
    }
    for case in DESIGN_CASES:
        assert all(f"market_cost_npv_eur_{case}" in row for row in canonical_smoke_rows)


def test_no_services_has_complete_separate_carbon_ledgers(canonical_smoke_rows):
    assert all(row["emissions_tco2_no_services"] > 0.0 for row in canonical_smoke_rows)
    assert all(row["carbon_market_cost_eur_no_services"] > 0.0 for row in canonical_smoke_rows)
    assert all(row["carbon_damage_eur_no_services"] > 0.0 for row in canonical_smoke_rows)


def test_headline_cases_share_carbon_cap(canonical_smoke_rows):
    for row in canonical_smoke_rows:
        cap = float(row["carbon_cap_tco2"])
        for case in ("fair_services_disabled", "baseline", "robust", "scenario_perfect"):
            assert float(row[f"emissions_tco2_{case}"]) <= cap + 1e-6


def test_identity_and_fair_comparator_have_zero_sipue_credit(canonical_smoke_rows):
    for row in canonical_smoke_rows:
        for case in ("no_services", "fair_services_disabled"):
            assert row[f"pue_{case}"] == pytest.approx(row[f"sipue_{case}"])
            assert row[f"system_benefit_eur_{case}"] == pytest.approx(0.0, abs=1e-8)


def test_canonical_schema_excludes_removed_kpis(canonical_smoke_rows):
    banned = ("operator", "settlement", "unitprofit", "profitpct", "co2costshare")
    assert not any(
        any(token in key.lower() for token in banned)
        for row in canonical_smoke_rows
        for key in row
    )


def test_cooling_and_heat_sources_are_identifiable(canonical_smoke_rows):
    for row in canonical_smoke_rows:
        assert row["cooling_mwh_robust"] >= 0.0
        assert row["recovered_heat_it_mwhth_robust"] >= 0.0
        assert row["recovered_heat_cooling_mwhth_robust"] >= 0.0


def test_robust_design_is_normalized_and_uses_site_land_limits(canonical_smoke_rows):
    intensity_fields = (
        "K_HX_per_MW_IT_robust",
        "K_HP_per_MW_IT_robust",
        "K_FLEX_per_MW_IT_robust",
        "K_BAT_E_per_MW_IT_robust",
        "K_BAT_P_per_MW_IT_robust",
    )
    for archetype in {row["archetype_key"] for row in canonical_smoke_rows}:
        subset = [row for row in canonical_smoke_rows if row["archetype_key"] == archetype]
        for field in intensity_fields:
            values = [float(row[field]) for row in subset]
            assert max(values) - min(values) <= 1e-9
    land_values = set()
    for row in canonical_smoke_rows:
        land = float(row["land_m2_robust"])
        limit = float(row["land_limit_m2_robust"])
        land_values.add(round(land, 6))
        assert land <= limit + 1e-6
        assert row["cooling_land_m2_robust"] > 0.0
        assert row["integration_land_m2_robust"] >= 0.0
        assert row["land_utilization_pct_robust"] == pytest.approx(100.0 * land / limit)
    assert len(land_values) > 1


def test_zero_delay_recovers_post_construction_metrics(canonical_smoke_rows):
    zero_delay = [row for row in canonical_smoke_rows if row["construction_delay_months"] == 0.0]
    assert zero_delay
    for row in zero_delay:
        assert row["sipue_robust"] == pytest.approx(row["steady_sipue_robust"])
        assert row["lifecycle_value_loss_eur_robust"] == pytest.approx(0.0, abs=1e-6)
        assert row["sipue_credit_loss_pct_robust"] == pytest.approx(0.0, abs=1e-9)


def test_percentage_sign_convention_and_references(canonical_smoke_rows):
    assert _reduction_pct(100.0, 90.0) == pytest.approx(10.0)
    assert _reduction_pct(100.0, 110.0) == pytest.approx(-10.0)
    for row in canonical_smoke_rows:
        assert row["lifecycle_cost_reduction_pct_vs_no_services_no_services"] == pytest.approx(0.0)
        assert row["lifecycle_cost_reduction_pct_vs_baseline_baseline"] == pytest.approx(0.0)
        for case in DESIGN_CASES:
            assert f"emissions_reduction_pct_vs_no_services_{case}" in row
            assert f"co2_cost_reduction_pct_vs_no_services_{case}" in row


def test_conclusion_summary_distinguishes_ratio_of_averages(canonical_smoke_rows):
    comparison = []
    for row in canonical_smoke_rows:
        for case in DESIGN_CASES:
            regret = 0.0 if case == "robust" else None
            comparison.append({
                "design": case,
                "regret_eur": regret,
                "regret_share_of_attainable_pct": None,
            })
    summary = _conclusion_summary(canonical_smoke_rows, comparison)
    assert len(summary) == len(DESIGN_CASES) * 5
    robust = next(item for item in summary if item["archetype"] == "all" and item["design"] == "robust")
    assert robust["scenario_count"] == 40
    assert "lifecycle_cost_reduction_pct_vs_no_services" in robust
    assert "average_scenario_lifecycle_cost_reduction_pct_vs_no_services" in robust
    assert robust["lifecycle_cost_direction_vs_no_services"] in {"increase", "decrease", "unchanged"}
    assert robust["pue_direction_vs_no_services"] in {"increase", "decrease", "unchanged"}


def test_kpi_benchmark_summary_uses_supported_canonical_metrics(canonical_smoke_rows):
    summary = _kpi_benchmark_summary(canonical_smoke_rows)
    expected_metrics = {
        "pue",
        "sipue",
        "sipue_no_carbon_damage",
        "sipue_carbon_only",
        "steady_sipue",
        "cue",
        "modeled_erf",
        "system_benefit_per_mwh_it",
        "system_benefit_no_carbon_damage_per_mwh_it",
        "system_benefit_carbon_only_per_mwh_it",
        "emissions_tco2",
        "carbon_damage_eur",
    }
    expected_groups = {
        ("all", "all"),
        *{
            (row["archetype"], row["archetype_key"])
            for row in canonical_smoke_rows
        },
    }
    assert {row["metric"] for row in summary} == expected_metrics
    assert {row["design_case"] for row in summary} == set(DESIGN_CASES)
    assert {(row["archetype"], row["archetype_key"]) for row in summary} == expected_groups
    assert len(summary) == len(expected_groups) * len(DESIGN_CASES) * len(expected_metrics)

    for row in summary:
        assert row["scenario_count"] in {10, 40}
        assert row["average_value"] == row["average_value"]
        if row["metric"] == "modeled_erf":
            assert row["average_value"] >= 0.0
            if row["design_case"] == "no_services":
                assert row["average_value"] == pytest.approx(0.0, abs=1e-9)

    all_rows = [row for row in canonical_smoke_rows]
    for design in DESIGN_CASES:
        for metric in ("pue", "sipue", "cue"):
            field = f"{metric}_{design}"
            expected = sum(float(row[field]) for row in all_rows) / len(all_rows)
            actual = next(
                row["average_value"]
                for row in summary
                if row["archetype"] == "all"
                and row["design_case"] == design
                and row["metric"] == metric
            )
            assert actual == pytest.approx(expected)
