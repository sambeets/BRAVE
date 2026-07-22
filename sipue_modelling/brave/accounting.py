from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


_EPS = 1e-9


def present_value_annuity_factor(years: int, real_discount_rate: float) -> float:
    """Present value of one end-of-year EUR paid for ``years`` years."""
    if years <= 0:
        return 0.0
    if real_discount_rate <= _EPS:
        return float(years)
    return sum((1.0 + real_discount_rate) ** -year for year in range(1, years + 1))


def lifecycle_operating_factor(
    years: int,
    real_discount_rate: float,
    commissioning_delay_months: float,
) -> float:
    """PV factor for a steady annual operating value after commissioning.

    Operating value accrues monthly. Delayed months are unavailable and are
    not recovered by extending the fixed investment horizon.
    """
    months = max(0, int(round(years * 12)))
    delay = min(months, max(0, int(round(commissioning_delay_months))))
    if months == 0 or delay == months:
        return 0.0
    monthly_value = 1.0 / 12.0
    return sum(
        monthly_value / ((1.0 + real_discount_rate) ** ((month + 1) / 12.0))
        for month in range(delay, months)
    )


def lifecycle_availability_factor(
    years: int,
    real_discount_rate: float,
    commissioning_delay_months: float,
) -> float:
    no_delay = lifecycle_operating_factor(years, real_discount_rate, 0.0)
    if no_delay <= _EPS:
        return 0.0
    delayed = lifecycle_operating_factor(
        years,
        real_discount_rate,
        commissioning_delay_months,
    )
    return min(1.0, max(0.0, delayed / no_delay))


@dataclass(frozen=True)
class CarbonLedgers:
    market_cost_eur: float
    independent_cost_eur: float
    carbon_market_cost_eur: float
    carbon_damage_eur: float

    def validate_no_double_counting(self) -> None:
        if not all(
            value == value
            for value in (
                self.market_cost_eur,
                self.independent_cost_eur,
                self.carbon_market_cost_eur,
                self.carbon_damage_eur,
            )
        ):
            raise ValueError("Carbon ledger values must be finite.")


def build_carbon_ledgers(
    *,
    resource_cost_eur: float,
    transmission_cost_eur: float,
    carbon_market_cost_eur: float,
    carbon_damage_eur: float,
    market_grid_peak_cost_eur: float,
    independent_peak_capacity_cost_eur: float,
    sla_cost_eur: float,
    capex_eur: float,
    land_cost_eur: float,
    heat_revenue_eur: float,
    flex_revenue_eur: float,
    avoided_heat_cost_eur: float,
    avoided_flex_cost_eur: float,
    efficiency_benefit_eur: float = 0.0,
) -> CarbonLedgers:
    """Build separate market and independent ledgers.

    ETS payments appear only in the market ledger. Carbon damage appears only
    in the independent ledger, so the two carbon valuations are never added.
    """
    market_common = resource_cost_eur + transmission_cost_eur
    independent_common = (
        resource_cost_eur
        + transmission_cost_eur
        - avoided_heat_cost_eur
        - avoided_flex_cost_eur
    )
    ledgers = CarbonLedgers(
        market_cost_eur=(
            market_common
            + carbon_market_cost_eur
            + market_grid_peak_cost_eur
            + sla_cost_eur
            + capex_eur
            + land_cost_eur
            - heat_revenue_eur
            - flex_revenue_eur
            - efficiency_benefit_eur
        ),
        independent_cost_eur=(
            independent_common
            + carbon_damage_eur
            + independent_peak_capacity_cost_eur
        ),
        carbon_market_cost_eur=carbon_market_cost_eur,
        carbon_damage_eur=carbon_damage_eur,
    )
    ledgers.validate_no_double_counting()
    return ledgers


def lifecycle_market_value(
    *,
    annual_market_operating_cost_eur: float,
    upfront_capex_eur: float,
    annual_land_cost_eur: float,
    years: int,
    real_discount_rate: float,
    commissioning_delay_months: float,
) -> Dict[str, float]:
    operating_factor = lifecycle_operating_factor(
        years,
        real_discount_rate,
        commissioning_delay_months,
    )
    land_factor = present_value_annuity_factor(years, real_discount_rate)
    operating_cost_npv = annual_market_operating_cost_eur * operating_factor
    land_cost_npv = annual_land_cost_eur * land_factor
    total_cost_npv = operating_cost_npv + upfront_capex_eur + land_cost_npv
    return {
        "operating_factor": operating_factor,
        "availability_factor": lifecycle_availability_factor(
            years,
            real_discount_rate,
            commissioning_delay_months,
        ),
        "operating_cost_npv_eur": operating_cost_npv,
        "capex_npv_eur": upfront_capex_eur,
        "land_cost_npv_eur": land_cost_npv,
        "market_cost_npv_eur": total_cost_npv,
        "market_value_npv_eur": -total_cost_npv,
    }


def decomposition_sum(components: Mapping[str, Optional[float]]) -> float:
    return sum(float(value) for value in components.values() if value is not None)
