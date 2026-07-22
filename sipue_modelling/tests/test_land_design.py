import pytest

from brave.config import Config
from brave.metrics import compute_land_components, compute_land_m2, effective_land_limit_m2
from brave.scenario import Scenario


pytestmark = pytest.mark.unit


def _scenario(limit=800.0):
    return Scenario(
        sid=1,
        name="land-test",
        dc_type="hyperscale",
        location="Netherlands",
        base_it_mw=20.0,
        flex_frac=0.1,
        base_pue=1.2,
        heat_recovery_frac=0.4,
        hp_cop=3.5,
        el_price_level=50.0,
        el_price_vol=10.0,
        heat_price_level=30.0,
        heat_demand_level=20.0,
        flex_price_level=10.0,
        carbon_factor=0.2,
        carbon_price=65.0,
        land_limit_m2=limit,
    )


def test_land_separates_existing_cooling_and_integration_area():
    cfg = Config(enforce_locked_calibration=False)
    parts = compute_land_components(
        2.0, 1.0, 4.0, 2.0, 0.0, cfg,
        K_HP=1.0, base_it_mw=20.0, K_COOL=4.0,
    )
    assert parts["existing_core_m2"] == pytest.approx(160.0)
    assert parts["existing_cooling_m2"] == pytest.approx(20.0)
    assert parts["integration_m2"] == pytest.approx(117.0)
    assert compute_land_m2(
        2.0, 1.0, 4.0, 2.0, 0.0, cfg,
        K_HP=1.0, base_it_mw=20.0, K_COOL=4.0,
    ) == pytest.approx(297.0)


def test_land_limit_multiplier_preserves_site_specific_limit():
    sc = _scenario(800.0)
    cfg = Config(enforce_locked_calibration=False, land_limit_multiplier=0.8)
    assert effective_land_limit_m2(sc, cfg) == pytest.approx(640.0)


def test_global_land_limit_is_only_a_fallback():
    sc = _scenario(None)
    cfg = Config(enforce_locked_calibration=False, land_limit_m2=600.0)
    assert effective_land_limit_m2(sc, cfg) == pytest.approx(600.0)
