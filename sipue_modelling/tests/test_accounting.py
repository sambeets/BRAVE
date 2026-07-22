import pytest

from brave.accounting import (
    build_carbon_ledgers,
    lifecycle_availability_factor,
    lifecycle_market_value,
    lifecycle_operating_factor,
)


pytestmark = pytest.mark.unit


def test_carbon_ledgers_reconcile_without_double_counting():
    ledgers = build_carbon_ledgers(
        resource_cost_eur=100.0,
        transmission_cost_eur=20.0,
        carbon_market_cost_eur=30.0,
        carbon_damage_eur=40.0,
        market_grid_peak_cost_eur=10.0,
        independent_peak_capacity_cost_eur=12.0,
        sla_cost_eur=5.0,
        capex_eur=50.0,
        land_cost_eur=7.0,
        heat_revenue_eur=8.0,
        flex_revenue_eur=9.0,
        avoided_heat_cost_eur=8.0,
        avoided_flex_cost_eur=9.0,
        efficiency_benefit_eur=0.0,
    )
    assert ledgers.market_cost_eur == pytest.approx(205.0)
    assert ledgers.independent_cost_eur == pytest.approx(155.0)
    assert ledgers.market_cost_eur != pytest.approx(245.0)
    assert ledgers.independent_cost_eur != pytest.approx(185.0)


def test_delay_reduces_only_operating_availability():
    full = lifecycle_operating_factor(15, 0.03, 0)
    six = lifecycle_operating_factor(15, 0.03, 6)
    twelve = lifecycle_operating_factor(15, 0.03, 12)
    assert full > six > twelve > 0.0
    assert lifecycle_availability_factor(15, 0.03, 0) == pytest.approx(1.0)


def test_capex_and_land_are_not_delayed():
    zero = lifecycle_market_value(
        annual_market_operating_cost_eur=100.0,
        upfront_capex_eur=1000.0,
        annual_land_cost_eur=10.0,
        years=15,
        real_discount_rate=0.03,
        commissioning_delay_months=0,
    )
    delayed = lifecycle_market_value(
        annual_market_operating_cost_eur=100.0,
        upfront_capex_eur=1000.0,
        annual_land_cost_eur=10.0,
        years=15,
        real_discount_rate=0.03,
        commissioning_delay_months=12,
    )
    assert delayed["capex_npv_eur"] == zero["capex_npv_eur"]
    assert delayed["land_cost_npv_eur"] == pytest.approx(zero["land_cost_npv_eur"])
    assert delayed["operating_cost_npv_eur"] < zero["operating_cost_npv_eur"]
