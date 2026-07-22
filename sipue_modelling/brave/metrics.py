from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from brave.config import Config
from brave.accounting import lifecycle_availability_factor

if TYPE_CHECKING:
    from brave.scenario import Scenario

# ---------------------------------------------------------------------------
# Non-annual horizon scaling constants used when representative-day annual
# compression is disabled.
# ---------------------------------------------------------------------------
_REP_TOTAL_DAYS: int = 7
_EPS: float = 1e-9


def nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if x == x]  # filter NaN
    return sum(vals) / max(1, len(vals))


def spearman_rank_corr(a: List[float], b: List[float]) -> float:
    """Simple Spearman rank correlation without external deps."""
    n = len(a)
    if n != len(b) or n == 0:
        return float("nan")

    def ranks(x):
        idx = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg_rank
            i = j + 1
        return r

    ra = ranks(a)
    rb = ranks(b)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    den_a = sum((ra[i] - mean_ra) ** 2 for i in range(n))
    den_b = sum((rb[i] - mean_rb) ** 2 for i in range(n))
    if den_a <= 1e-12 or den_b <= 1e-12:
        return float("nan")
    return num / ((den_a ** 0.5) * (den_b ** 0.5))


def compute_sipue(
    E_FAC: float,
    E_IT: float,
    net_system_benefit_eur: float,
    reference_price_eur_per_mwh: float,
    clip_nonnegative: bool = True,
    benefit_availability_factor: float = 1.0,
) -> Tuple[float, float]:
    """
    System-integrated PUE based on independent avoided system cost.

    siPUE = [E_FAC - a * net_system_benefit_eur / reference_price_eur_per_mwh]^+ / E_IT

    The benefit term is a counterfactual system-cost reduction, not private
    heat or flexibility revenue. Positive values lower siPUE; negative values
    raise it. The availability factor a discounts credited system benefit for
    construction delay.
    """
    if E_IT <= _EPS:
        raise ValueError("E_IT must be positive when computing siPUE.")
    if reference_price_eur_per_mwh <= _EPS:
        raise ValueError("reference_price_eur_per_mwh must be positive when computing siPUE.")
    availability = min(1.0, max(0.0, float(benefit_availability_factor)))
    system_credit_mwh = availability * net_system_benefit_eur / reference_price_eur_per_mwh
    adjusted_energy = E_FAC - system_credit_mwh
    if clip_nonnegative:
        adjusted_energy = max(0.0, adjusted_energy)
    return adjusted_energy / E_IT, system_credit_mwh


def compute_land_m2(
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    eff_reduction: float,
    cfg: Config,
    K_HP: float = 0.0,
    base_it_mw: Optional[float] = None,
    K_COOL: float = 0.0,
) -> float:
    return sum(compute_land_components(
        K_HX,
        K_FLEX,
        K_BAT_E,
        K_BAT_P,
        eff_reduction,
        cfg,
        K_HP=K_HP,
        base_it_mw=base_it_mw,
        K_COOL=K_COOL,
    ).values())


def compute_land_components(
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    eff_reduction: float,
    cfg: Config,
    K_HP: float = 0.0,
    base_it_mw: Optional[float] = None,
    K_COOL: float = 0.0,
) -> Dict[str, float]:
    del eff_reduction
    existing_core = (
        cfg.base_land_m2
        if base_it_mw is None
        else cfg.existing_land_m2_per_it_mw * base_it_mw
    )
    return {
        "existing_core_m2": existing_core,
        "existing_cooling_m2": cfg.land_a_cooling * K_COOL,
        "integration_m2": (
            cfg.land_a_hx * K_HX
            + cfg.land_a_hp * K_HP
            + cfg.land_a_flex * K_FLEX
            + cfg.land_a_bat_e * K_BAT_E
            + cfg.land_a_bat_p * K_BAT_P
        ),
    }


def effective_land_limit_m2(sc: Scenario, cfg: Config) -> Optional[float]:
    limit = sc.land_limit_m2 if sc.land_limit_m2 is not None else cfg.land_limit_m2
    if limit is None:
        return None
    return max(0.0, float(limit) * max(0.0, float(cfg.land_limit_multiplier)))


def service_revenue_scale(cfg: Config, sc: Optional[Scenario] = None) -> float:
    """Discounted lifecycle availability used only for siPUE credit.

    Dispatch and full-system costs are never scaled by this factor. The
    deprecated fractional inputs remain as a compatibility conversion to
    months for callers that have not migrated yet.
    """
    delay_months = max(0.0, float(getattr(cfg, "construction_delay_months", 0.0)))
    if delay_months <= _EPS:
        delay_months = 12.0 * max(0.0, float(getattr(cfg, "construction_delay_frac", 0.0)))
    if sc is not None:
        scenario_months = max(0.0, float(getattr(sc, "construction_delay_months", 0.0)))
        if scenario_months <= _EPS:
            scenario_months = 12.0 * max(0.0, float(getattr(sc, "construction_delay_frac", 0.0)))
        delay_months += scenario_months
    return lifecycle_availability_factor(
        int(cfg.lifecycle_years),
        float(cfg.real_discount_rate),
        delay_months,
    )


def land_cost_horizon(land_m2: float, cfg: Config) -> float:
    scale = 1.0
    if cfg.land_cost_is_annual:
        if cfg.use_representative_days_annual:
            scale = 1.0  # weights already annualise operations; keep land annual too
        else:
            scale = _REP_TOTAL_DAYS / max(1, cfg.days_per_year)
    maint = 1.0 + max(0.0, cfg.land_maintenance_frac)
    return land_m2 * cfg.land_cost_eur_per_m2 * scale * maint


def demand_charge_horizon(cfg: Config) -> float:
    """
    Horizon-scaled demand charge rate (EUR per MW of peak import, scaled to the modeled horizon).
    """
    if not cfg.enable_demand_charge:
        return 0.0
    if cfg.use_representative_days_annual:
        return cfg.demand_charge_eur_per_mw_year  # weights already annualise; keep demand charge annual
    scale = _REP_TOTAL_DAYS / max(1, cfg.days_per_year)
    return cfg.demand_charge_eur_per_mw_year * scale


def independent_peak_capacity_horizon(cfg: Config) -> float:
    if cfg.use_representative_days_annual:
        return cfg.independent_peak_capacity_value_eur_per_mw_year
    scale = _REP_TOTAL_DAYS / max(1, cfg.days_per_year)
    return cfg.independent_peak_capacity_value_eur_per_mw_year * scale


def compute_independent_system_cost(
    power_resource_cost_eur: float,
    transmission_cost_eur: float,
    emissions_tco2: float,
    heat_delivered_mwhth: float,
    peak_grid_mw: float,
    cfg: Config,
    avoided_heat_cost_eur_per_mwhth: Optional[List[float]] = None,
    heat_delivered_mwth: Optional[List[float]] = None,
    avoided_flex_cost_eur_per_mw_h: Optional[List[float]] = None,
    flex_delivered_mw: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    dt: float = 1.0,
    carbon_damage_eur_per_tco2: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    External system-cost evaluation independent of private heat/flex payments.

    Heat and flexibility enter only through avoided system-resource costs.
    """
    damage_value = (
        cfg.independent_social_carbon_cost_eur_per_tco2
        if carbon_damage_eur_per_tco2 is None
        else float(carbon_damage_eur_per_tco2)
    )
    carbon_damage_eur = damage_value * emissions_tco2
    if (
        avoided_heat_cost_eur_per_mwhth is not None
        and heat_delivered_mwth is not None
        and len(avoided_heat_cost_eur_per_mwhth) == len(heat_delivered_mwth)
    ):
        w = weights if weights is not None and len(weights) == len(heat_delivered_mwth) else [1.0] * len(heat_delivered_mwth)
        avoided_heat_cost_eur = sum(
            avoided_heat_cost_eur_per_mwhth[t] * heat_delivered_mwth[t] * dt * w[t]
            for t in range(len(heat_delivered_mwth))
        )
    else:
        avoided_heat_cost_eur = cfg.independent_avoided_heat_cost_eur_per_mwhth * heat_delivered_mwhth
    if (
        avoided_flex_cost_eur_per_mw_h is not None
        and flex_delivered_mw is not None
        and len(avoided_flex_cost_eur_per_mw_h) == len(flex_delivered_mw)
    ):
        w = weights if weights is not None and len(weights) == len(flex_delivered_mw) else [1.0] * len(flex_delivered_mw)
        avoided_flex_cost_eur = sum(
            avoided_flex_cost_eur_per_mw_h[t] * flex_delivered_mw[t] * dt * w[t]
            for t in range(len(flex_delivered_mw))
        )
    else:
        avoided_flex_cost_eur = 0.0
    peak_capacity_cost_eur = independent_peak_capacity_horizon(cfg) * peak_grid_mw
    system_cost_eur = (
        power_resource_cost_eur
        + transmission_cost_eur
        + carbon_damage_eur
        + peak_capacity_cost_eur
        - avoided_heat_cost_eur
        - avoided_flex_cost_eur
    )
    return system_cost_eur, avoided_heat_cost_eur, avoided_flex_cost_eur, carbon_damage_eur, peak_capacity_cost_eur


def design_size_proxy(
    sc: Scenario,
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    cfg: Config,
    K_HP: float = 0.0,
) -> float:
    return (
        sc.base_it_mw
        + cfg.ramp_k_hx * K_HX
        + cfg.ramp_k_hp * K_HP
        + cfg.ramp_k_flex * K_FLEX
        + cfg.ramp_k_bat_e * K_BAT_E
        + cfg.ramp_k_bat_p * K_BAT_P
    )


def eff_capex_from_reduction(eff_reduction: float, cfg: Config) -> float:
    if not cfg.eff_levels:
        return 0.0
    idx = min(range(len(cfg.eff_levels)), key=lambda i: abs(cfg.eff_levels[i] - eff_reduction))
    return cfg.eff_capex_levels[idx]


def capex_scale(cfg: Config) -> float:
    if not cfg.capex_scale_with_horizon:
        return 1.0
    if cfg.use_representative_days_annual:
        return 1.0
    return max(0.0, cfg.T / 8760.0)
