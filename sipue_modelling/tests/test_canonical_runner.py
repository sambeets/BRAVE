import pytest

from brave.config import Config
from brave.runners import DESIGN_CASES, _conclusion_summary, _reduction_pct, run_single


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
    assert {row["ScenarioID"] for row in canonical_smoke_rows} == set(range(1, 11))
    counts = {}
    for row in canonical_smoke_rows:
        counts[row["ArchetypeKey"]] = counts.get(row["ArchetypeKey"], 0) + 1
    assert counts == {
        "ai_hpc": 10,
        "hyperscale": 10,
        "colocation": 10,
        "enterprise_edge": 10,
    }
    for case in DESIGN_CASES:
        assert all(f"MarketCostNPVEUR_{case}" in row for row in canonical_smoke_rows)


def test_no_services_has_complete_separate_carbon_ledgers(canonical_smoke_rows):
    assert all(row["EmissionsTCO2_no_services"] > 0.0 for row in canonical_smoke_rows)
    assert all(row["CarbonMarketCostEUR_no_services"] > 0.0 for row in canonical_smoke_rows)
    assert all(row["CarbonDamageEUR_no_services"] > 0.0 for row in canonical_smoke_rows)


def test_canonical_schema_excludes_removed_kpis(canonical_smoke_rows):
    banned = ("operator", "settlement", "unitprofit", "profitpct", "co2costshare")
    assert not any(
        any(token in key.lower() for token in banned)
        for row in canonical_smoke_rows
        for key in row
    )


def test_cooling_and_heat_sources_are_identifiable(canonical_smoke_rows):
    for row in canonical_smoke_rows:
        assert row["CoolingMWh_robust"] >= 0.0
        assert row["RecoveredHeatITMWhth_robust"] >= 0.0
        assert row["RecoveredHeatCoolingMWhth_robust"] >= 0.0


def test_robust_design_is_normalized_and_uses_site_land_limits(canonical_smoke_rows):
    intensity_fields = (
        "K_HX_per_MW_IT_robust",
        "K_HP_per_MW_IT_robust",
        "K_FLEX_per_MW_IT_robust",
        "K_BAT_E_per_MW_IT_robust",
        "K_BAT_P_per_MW_IT_robust",
    )
    for archetype in {row["ArchetypeKey"] for row in canonical_smoke_rows}:
        subset = [row for row in canonical_smoke_rows if row["ArchetypeKey"] == archetype]
        for field in intensity_fields:
            values = [float(row[field]) for row in subset]
            assert max(values) - min(values) <= 1e-9
    land_values = set()
    for row in canonical_smoke_rows:
        land = float(row["LandM2_robust"])
        limit = float(row["LandLimitM2_robust"])
        land_values.add(round(land, 6))
        assert land <= limit + 1e-6
        assert row["CoolingLandM2_robust"] > 0.0
        assert row["IntegrationLandM2_robust"] >= 0.0
        assert row["LandUtilizationPct_robust"] == pytest.approx(100.0 * land / limit)
    assert len(land_values) > 1


def test_zero_delay_recovers_post_construction_metrics(canonical_smoke_rows):
    zero_delay = [row for row in canonical_smoke_rows if row["ConstructionDelayMonths"] == 0.0]
    assert zero_delay
    for row in zero_delay:
        assert row["siPUE_robust"] == pytest.approx(row["SteadySiPUE_robust"])
        assert row["LifecycleValueLossEUR_robust"] == pytest.approx(0.0, abs=1e-6)
        assert row["SiPUECreditLossPct_robust"] == pytest.approx(0.0, abs=1e-9)


def test_percentage_sign_convention_and_references(canonical_smoke_rows):
    assert _reduction_pct(100.0, 90.0) == pytest.approx(10.0)
    assert _reduction_pct(100.0, 110.0) == pytest.approx(-10.0)
    for row in canonical_smoke_rows:
        assert row["LifecycleCostReductionPctVsNoServices_no_services"] == pytest.approx(0.0)
        assert row["LifecycleCostReductionPctVsBaseline_baseline"] == pytest.approx(0.0)
        for case in DESIGN_CASES:
            assert f"EmissionsReductionPctVsNoServices_{case}" in row
            assert f"CO2CostReductionPctVsNoServices_{case}" in row


def test_conclusion_summary_distinguishes_ratio_of_averages(canonical_smoke_rows):
    comparison = []
    for row in canonical_smoke_rows:
        perfect = row["MarketCostNPVEUR_scenario_perfect"]
        baseline = row["MarketCostNPVEUR_baseline"]
        attainable = max(1e-9, baseline - perfect)
        for case in DESIGN_CASES:
            regret = row[f"MarketCostNPVEUR_{case}"] - perfect
            comparison.append({
                "Design": case,
                "RegretEUR": regret,
                "RegretShareOfAttainablePct": 100.0 * regret / attainable,
            })
    summary = _conclusion_summary(canonical_smoke_rows, comparison)
    assert len(summary) == len(DESIGN_CASES) * 5
    robust = next(item for item in summary if item["Archetype"] == "all" and item["Design"] == "robust")
    assert robust["ScenarioCount"] == 40
    assert "LifecycleCostReductionPctVsNoServices" in robust
    assert "AverageScenarioLifecycleCostReductionPctVsNoServices" in robust
    assert robust["LifecycleCostDirectionVsNoServices"] in {"increase", "decrease", "unchanged"}
    assert robust["PUEDirectionVsNoServices"] in {"increase", "decrease", "unchanged"}
