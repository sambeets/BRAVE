import pytest
from main import Config, Scenario, make_overhead_alpha_base

pytestmark = pytest.mark.unit


class TestNonNordicConstant:
    def test_netherlands_constant_alpha(self, cfg_test, scenario_hyperscale):
        alpha = make_overhead_alpha_base(scenario_hyperscale, 24, cfg_test)
        base = scenario_hyperscale.base_pue - 1.0
        assert len(alpha) == 24
        assert all(a == pytest.approx(base) for a in alpha)


class TestNordicTimeVarying:
    def test_norway_has_night_drops(self, cfg_test, scenario_nordic):
        alpha = make_overhead_alpha_base(scenario_nordic, 48, cfg_test)
        base = scenario_nordic.base_pue - 1.0
        night_expected = base * (1.0 - cfg_test.nordic_night_overhead_drop)
        # Hour 3 is a night hour (0-5)
        assert alpha[3] == pytest.approx(night_expected)

    def test_norway_has_day_bump(self, cfg_test, scenario_nordic):
        alpha = make_overhead_alpha_base(scenario_nordic, 48, cfg_test)
        base = scenario_nordic.base_pue - 1.0
        day_expected = base * (1.0 + cfg_test.day_overhead_bump)
        # Hour 12 is a day hour
        assert alpha[12] == pytest.approx(day_expected)


class TestDisabledTimeVarying:
    def test_all_constant_when_disabled(self, scenario_nordic):
        cfg = Config(enforce_locked_calibration=False, enable_time_varying_overhead=False, T=24)
        alpha = make_overhead_alpha_base(scenario_nordic, 24, cfg)
        base = scenario_nordic.base_pue - 1.0
        assert all(a == pytest.approx(base) for a in alpha)


class TestNightDropOverride:
    def test_override(self, cfg_test, scenario_nordic):
        alpha = make_overhead_alpha_base(scenario_nordic, 24, cfg_test, night_drop_override=0.50)
        base = scenario_nordic.base_pue - 1.0
        assert alpha[0] == pytest.approx(base * 0.50)


class TestLengthMatchesT:
    @pytest.mark.parametrize("T", [24, 48, 168])
    def test_length(self, cfg_test, scenario_hyperscale, T):
        alpha = make_overhead_alpha_base(scenario_hyperscale, T, cfg_test)
        assert len(alpha) == T
