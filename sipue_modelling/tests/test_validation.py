import pytest

from main import build_ablation_summary, build_validation_summary, compute_ere, _make_cv_folds
from brave.config import Config
from brave.scenario import make_scenarios


pytestmark = pytest.mark.unit


def _rows():
    rows = []
    for sid in range(1, 11):
        pue = 1.2 + sid / 100.0
        rows.append({
            "ScenarioID": sid,
            "PUE_no_services": pue,
            "siPUE_no_services": pue,
            "siPUE_baseline": pue - 0.01,
            "siPUE_robust": pue - 0.02,
            "SteadySiPUE_robust": pue - 0.03,
            "siPUE_expected_value": pue - 0.015,
            "siPUE_pure_minmax": pue - 0.018,
            "siPUE_scenario_perfect": pue - 0.025,
            "SystemBenefitEUR_robust": 100.0,
            "CarbonMarketCostEUR_robust": 50.0,
            "CarbonDamageEUR_robust": 75.0,
        })
    return rows


def test_compute_ere_uses_delivered_heat_reuse():
    assert compute_ere(130.0, 100.0, 20.0) == pytest.approx(1.10)


def test_validation_uses_properties_not_constructed_target_correlation():
    summary = build_validation_summary(_rows())
    tests = {row["test"] for row in summary}
    assert {
        "finite_nonnegative",
        "no_services_identity",
        "delay_monotonicity",
        "carbon_ledger_separation",
        "canonical_scenario_count",
    }.issubset(tests)
    assert all(row["passed"] for row in summary)
    assert not any("correlation" in row["test"] for row in summary)


def test_obsolete_market_credit_ablations_are_removed():
    assert build_ablation_summary(_rows()) == []


def test_cv_folds_are_five_eight_train_two_test_splits():
    scenarios = make_scenarios(Config(T=24))
    folds = _make_cv_folds(scenarios, 5)
    held_out_sids = sorted(sc.sid for fold in folds for sc in fold)
    assert held_out_sids == list(range(1, 11))
    assert len(folds) == 5
    assert all(len(fold) == 2 for fold in folds)
    assert all(len(scenarios) - len(fold) == 8 for fold in folds)
