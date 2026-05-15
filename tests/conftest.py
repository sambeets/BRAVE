import os
import sys
import random

import pytest

# Add project root to path so we can import main
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from main import (  # noqa: E402
    Config, Scenario, make_scenarios, apply_global_heat_assumptions,
    make_hourly_profiles, apply_services_mode, _ensure_network_defaults,
    _NETWORK_TS_CACHE,
)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "network_3node_timeseries.csv")


@pytest.fixture
def cfg_test():
    """Minimal Config for fast tests (T=24, no file exports, no locked calibration)."""
    return Config(
        T=24,
        seed=42,
        enforce_locked_calibration=False,
        solver_msg=False,
        export_csv=False,
        export_plots=False,
        use_representative_days_annual=False,
        network_timeseries_csv=CSV_PATH,
    )


@pytest.fixture
def cfg_locked():
    """Config with locked calibration enabled (default values)."""
    return Config(
        T=24,
        seed=42,
        solver_msg=False,
        export_csv=False,
        export_plots=False,
        use_representative_days_annual=False,
        network_timeseries_csv=CSV_PATH,
    )


@pytest.fixture
def scenario_hyperscale():
    """S1 Hyperscale-Netherlands (base_it_mw=40)."""
    return Scenario(
        sid=1, name="Hyperscale-Netherlands", dc_type="hyperscale",
        location="Netherlands",
        base_it_mw=40, flex_frac=0.15, base_pue=1.18,
        heat_recovery_frac=0.45, hp_cop=3.5,
        el_price_level=58, el_price_vol=20,
        heat_price_level=34.8, heat_demand_level=65.0,
        flex_price_level=14.5,
        carbon_factor=0.328, carbon_price=65.0,
        land_limit_m2=1100, construction_delay_frac=0.06,
    )


@pytest.fixture
def scenario_nordic():
    """S2 AI-Norway (base_it_mw=25)."""
    return Scenario(
        sid=2, name="AI-Norway", dc_type="ai", location="Norway",
        base_it_mw=25, flex_frac=0.12, base_pue=1.12,
        heat_recovery_frac=0.50, hp_cop=3.6,
        el_price_level=42, el_price_vol=15,
        heat_price_level=32.76, heat_demand_level=45.0,
        flex_price_level=11.55,
        carbon_factor=0.029, carbon_price=65.0,
        land_limit_m2=850, construction_delay_frac=0.02,
    )


@pytest.fixture
def profile_24h(cfg_test, scenario_hyperscale):
    """24-hour seeded profile for scenario 1."""
    random.seed(cfg_test.seed)
    _ensure_network_defaults(cfg_test)
    prof = make_hourly_profiles(scenario_hyperscale, cfg_test.T, cfg_test)
    prof = apply_services_mode(scenario_hyperscale, prof, cfg_test)
    return prof


@pytest.fixture(autouse=True)
def clear_ts_cache():
    """Clear the global timeseries cache between tests."""
    _NETWORK_TS_CACHE.clear()
    yield
    _NETWORK_TS_CACHE.clear()
