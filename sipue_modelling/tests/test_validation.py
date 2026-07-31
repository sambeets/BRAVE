import pytest

from main import build_validation_summary
from brave.config import Config
from brave.runners import _train_test_split
from brave.scenario import DATA_CENTER_ARCHETYPES, make_archetype_scenarios


pytestmark = pytest.mark.unit


def _rows():
    """Synthetic fixture rows for build_validation_summary's property checks.

    Only two numeric relationships are actually load-bearing here:
      - siPUE_no_services == PUE_no_services (checked by no_services_identity)
      - siPUE_robust >= SteadySiPUE_robust, since SystemBenefitEUR_robust > 0
        (checked by delay_monotonicity: a positive credit can only be
        weakened, not strengthened, by construction delay)
    Every other siPUE_* value just needs to be finite and >= -_TOL; nothing
    asserts its exact magnitude, so a single shared credit is used for them.
    """
    undelayed_credit = 0.02
    delayed_credit = 0.01  # smaller credit realized once delay is applied
    rows = []
    for sid in range(1, 11):
        pue = 1.2 + sid / 100.0
        rows.append({
            "scenario_id": sid,
            "pue_no_services": pue,
            "sipue_no_services": pue,
            "sipue_baseline": pue - undelayed_credit,
            "sipue_robust": pue - delayed_credit,
            "steady_sipue_robust": pue - undelayed_credit,
            "system_benefit_eur_robust": 100.0,
            "carbon_market_cost_eur_robust": 50.0,
            "carbon_damage_eur_robust": 75.0,
        })
    return rows


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


def test_train_test_split_is_seven_train_three_test():
    scenarios = make_archetype_scenarios(Config(T=24), DATA_CENTER_ARCHETYPES[0])
    folds = _train_test_split(scenarios, 7)
    assert len(folds) == 1
    test = folds[0]
    assert sorted(sc.uncertainty_id for sc in test) == [8, 9, 10]
    train = [sc for sc in scenarios if sc not in test]
    assert sorted(sc.uncertainty_id for sc in train) == [1, 2, 3, 4, 5, 6, 7]
