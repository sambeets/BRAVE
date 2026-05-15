from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from brave.config import Config

if TYPE_CHECKING:
    from brave.scenario import Scenario

# ---------------------------------------------------------------------------
# Non-annual horizon scaling constants used when representative-day annual
# compression is disabled.
# ---------------------------------------------------------------------------
_REP_TOTAL_DAYS: int = 7


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
    credit_eur: float,
    heat_rev_eur: float,
    flex_rev_eur: float,
    avg_heat_price: float,
    avg_flex_price: float,
    co2_cost_eur: float,
    el_cost: float,
    el_settlement_eur: float,
    E_GRID: float,
    p_el_ref: List[float],
    dc_load_series: Optional[List[float]],
    settle_lmp_series: Optional[List[float]],
    normalization: str,
    credit_price_mode: str,
    clip_nonnegative: bool,
    price_floor_eur_per_mwh: float,
    include_co2_penalty: bool,
    co2_scale: float,
    co2_ref_price_eur_per_mwh: float
) -> Tuple[float, float, float, float, float, float, str, int]:
    """
    Returns (siPUE, credit_mwh, credit_mwh_heat, credit_mwh_flex, co2_mwh,
             price_used, price_source, price_floor_applied).

    normalization:
      - "avg_paid": price_used = el_cost / E_FAC
      - "net_grid_cost": price_used = el_settlement_eur / E_GRID
      - "ref_price": price_used = average(p_el_ref)
      - "settle_lmp_dc": price_used = load-weighted settlement price (DC node LMP from duals)

    credit_price_mode:
      - "lmp": convert service credits using price_used (system price)
      - "service_price": convert using avg service prices (heat/flex)
    """
    price_floor_applied = 0
    if normalization == "avg_paid":
        price_used = el_cost / max(1e-9, E_FAC)
        price_source = "avg_paid (el_cost / E_FAC)"
    elif normalization == "net_grid_cost":
        price_used = el_settlement_eur / max(1e-9, E_GRID)
        if price_used < price_floor_eur_per_mwh:
            price_used = price_floor_eur_per_mwh
            price_floor_applied = 1
        price_source = "net_grid_cost (el_settlement_eur / E_GRID)"
    elif normalization == "ref_price":
        if p_el_ref:
            price_used = sum(p_el_ref) / max(1, len(p_el_ref))
            price_source = "ref_price (avg p_el)"
        else:
            price_used = float("nan")
            price_source = "ref_price (missing)"
    elif normalization == "settle_lmp_dc":
        if not dc_load_series or not settle_lmp_series:
            raise ValueError("settle_lmp_dc requires dc_load_series and settle_lmp_series (DC node LMP)")
        num = 0.0
        den = 0.0
        for load, price in zip(dc_load_series, settle_lmp_series):
            num += load * price
            den += load
        if den <= 1e-9:
            price_used = sum(settle_lmp_series) / max(1, len(settle_lmp_series))
        else:
            price_used = num / den
        if price_used < price_floor_eur_per_mwh:
            price_used = price_floor_eur_per_mwh
            price_floor_applied = 1
        price_source = "settlement LMP at DC node (load-weighted)"
    else:
        raise ValueError("sipue_credit_normalization must be 'avg_paid', 'net_grid_cost', 'ref_price', or 'settle_lmp_dc'")

    credit_mode = (credit_price_mode or "service_price").strip().lower()
    if credit_mode in ("lmp", "price_used", "settle_lmp"):
        heat_price_used = price_used
        flex_price_used = price_used
    else:
        heat_price_used = avg_heat_price if avg_heat_price > 1e-9 else price_used
        flex_price_used = avg_flex_price if avg_flex_price > 1e-9 else price_used

    credit_mwh_heat = heat_rev_eur / max(1e-9, heat_price_used)
    credit_mwh_flex = flex_rev_eur / max(1e-9, flex_price_used)
    if include_co2_penalty:
        co2_price_used = max(1e-9, price_used)  # use electricity price for CO2 normalization
        co2_mwh = (co2_cost_eur * max(0.0, co2_scale)) / co2_price_used
    else:
        co2_mwh = 0.0
    credit_mwh = credit_mwh_heat + credit_mwh_flex
    E_adj = E_FAC - credit_mwh_heat - credit_mwh_flex + co2_mwh
    if clip_nonnegative:
        E_adj = max(0.0, E_adj)

    siPUE = E_adj / max(1e-9, E_IT)
    return siPUE, credit_mwh, credit_mwh_heat, credit_mwh_flex, co2_mwh, price_used, price_source, price_floor_applied


def compute_land_m2(
    K_HX: float,
    K_FLEX: float,
    K_BAT_E: float,
    K_BAT_P: float,
    eff_reduction: float,
    cfg: Config
) -> float:
    return (
        cfg.base_land_m2
        + cfg.land_a_hx * K_HX
        + cfg.land_a_flex * K_FLEX
        + cfg.land_a_bat_e * K_BAT_E
        + cfg.land_a_bat_p * K_BAT_P
    )


def service_revenue_scale(cfg: Config, sc: Optional[Scenario] = None) -> float:
    """
    Scale factor applied to IT + service revenues to represent construction delays.

    - cfg.construction_delay_frac: global delay applied to all scenarios (sensitivity sweeps).
    - sc.construction_delay_frac: scenario-specific delay (optional override/extra delay).

    The combined effect is multiplicative: (1-cfg_delay) * (1-scenario_delay).
    """
    cfg_scale = max(0.0, 1.0 - cfg.construction_delay_frac)
    sc_scale = 1.0
    if sc is not None:
        sc_scale = max(0.0, 1.0 - sc.construction_delay_frac)
    return cfg_scale * sc_scale


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


def design_size_proxy(sc: Scenario, K_HX: float, K_FLEX: float, K_BAT_E: float, K_BAT_P: float, cfg: Config) -> float:
    return (
        sc.base_it_mw
        + cfg.ramp_k_hx * K_HX
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
