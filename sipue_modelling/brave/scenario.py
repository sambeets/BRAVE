from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from brave.config import Config


@dataclass
class Scenario:
    sid: int
    name: str
    dc_type: str
    location: str

    base_it_mw: float
    flex_frac: float
    base_pue: float
    heat_recovery_frac: float
    hp_cop: float

    el_price_level: float
    el_price_vol: float
    heat_price_level: float
    heat_demand_level: float
    flex_price_level: float
    carbon_factor: float
    carbon_price: float
    land_limit_m2: Optional[float] = None
    construction_delay_frac: float = 0.0


def make_scenarios(cfg: Config) -> List[Scenario]:
    def _mult(vals: Tuple[float, ...], sid: int) -> float:
        if vals and len(vals) >= sid:
            return float(vals[sid - 1])
        return 1.0

    def _heat_price(base: float, sid: int) -> float:
        return cfg.heat_price_multiplier * _mult(cfg.heat_price_multipliers_by_sid, sid) * base

    def _flex_price(base: float, sid: int) -> float:
        return cfg.flex_price_multiplier * _mult(cfg.flex_price_multipliers_by_sid, sid) * base

    # Scenarios span 3 network nodes:
    #   N1 (Nordics): Norway, Sweden, Denmark — cheap clean power, high heat demand
    #   N2 (Central Europe): Netherlands, Belgium, Germany, France — mid-price, moderate carbon
    #   N3 (UK/West): United Kingdom — high price, high carbon
    # Land: base_land=500 + 30*K_HX + 15*K_FLEX + 5*K_BAT_E + 5*K_BAT_P
    # Baseline K: HX=0.30*IT, FLEX=0.10*IT, BAT_E=1.0*IT, BAT_P=0.25*IT
    # Baseline land for 40MW = 1170 m², 30MW = 903 m², 20MW = 835 m², 16MW = 768 m²
    return [
        # --- N2 node (Central Europe) ---
        Scenario(1, "Hyperscale-Netherlands", "hyperscale", "Netherlands",
                 base_it_mw=40, flex_frac=0.15, base_pue=1.18,
                 heat_recovery_frac=0.45, hp_cop=3.5,
                 el_price_level=58, el_price_vol=20,
                 heat_price_level=_heat_price(58, 1), heat_demand_level=65.0,
                 flex_price_level=_flex_price(58, 1),
                 carbon_factor=0.328, carbon_price=65.0,
                 land_limit_m2=1100, construction_delay_frac=0.06),

        # --- N1 node (Nordics) ---
        Scenario(2, "AI-Norway", "ai", "Norway",
                 base_it_mw=25, flex_frac=0.12, base_pue=1.12,
                 heat_recovery_frac=0.50, hp_cop=3.6,
                 el_price_level=42, el_price_vol=15,
                 heat_price_level=_heat_price(42, 2), heat_demand_level=45.0,
                 flex_price_level=_flex_price(42, 2),
                 carbon_factor=0.029, carbon_price=65.0,
                 land_limit_m2=850, construction_delay_frac=0.02),

        # --- N2 node (Central Europe) ---
        Scenario(3, "Streaming-Belgium", "streaming", "Belgium",
                 base_it_mw=18, flex_frac=0.22, base_pue=1.32,
                 heat_recovery_frac=0.40, hp_cop=3.2,
                 el_price_level=65, el_price_vol=28,
                 heat_price_level=_heat_price(65, 3), heat_demand_level=38.0,
                 flex_price_level=_flex_price(65, 3),
                 carbon_factor=0.148, carbon_price=65.0,
                 land_limit_m2=750, construction_delay_frac=0.08),

        # --- N1 node (Nordics) ---
        Scenario(4, "Hyperscale-Sweden", "hyperscale", "Sweden",
                 base_it_mw=35, flex_frac=0.14, base_pue=1.08,
                 heat_recovery_frac=0.50, hp_cop=3.4,
                 el_price_level=45, el_price_vol=16,
                 heat_price_level=_heat_price(45, 4), heat_demand_level=50.0,
                 flex_price_level=_flex_price(45, 4),
                 carbon_factor=0.015, carbon_price=65.0,
                 land_limit_m2=1000, construction_delay_frac=0.04),

        # --- N3 node (UK) ---
        Scenario(5, "Streaming-UnitedKingdom", "streaming", "United Kingdom",
                 base_it_mw=16, flex_frac=0.24, base_pue=1.35,
                 heat_recovery_frac=0.30, hp_cop=2.8,
                 el_price_level=82, el_price_vol=30,
                 heat_price_level=_heat_price(82, 5), heat_demand_level=20.0,
                 flex_price_level=_flex_price(82, 5),
                 carbon_factor=0.207, carbon_price=65.0,
                 land_limit_m2=700, construction_delay_frac=0.10),

        # --- N1 node (Nordics) ---
        Scenario(6, "AI-Sweden", "ai", "Sweden",
                 base_it_mw=20, flex_frac=0.10, base_pue=1.14,
                 heat_recovery_frac=0.45, hp_cop=3.3,
                 el_price_level=48, el_price_vol=18,
                 heat_price_level=_heat_price(48, 6), heat_demand_level=30.0,
                 flex_price_level=_flex_price(48, 6),
                 carbon_factor=0.015, carbon_price=65.0,
                 land_limit_m2=800, construction_delay_frac=0.03),

        # --- N2 node (Central Europe) ---
        Scenario(7, "Colocation-Germany", "colocation", "Germany",
                 base_it_mw=22, flex_frac=0.08, base_pue=1.35,
                 heat_recovery_frac=0.35, hp_cop=3.1,
                 el_price_level=62, el_price_vol=22,
                 heat_price_level=_heat_price(62, 7), heat_demand_level=42.0,
                 flex_price_level=_flex_price(62, 7),
                 carbon_factor=0.385, carbon_price=65.0,
                 land_limit_m2=820, construction_delay_frac=0.07),

        # --- N1 node (Nordics) ---
        Scenario(8, "AI-Denmark", "ai", "Denmark",
                 base_it_mw=22, flex_frac=0.13, base_pue=1.13,
                 heat_recovery_frac=0.50, hp_cop=3.5,
                 el_price_level=50, el_price_vol=20,
                 heat_price_level=_heat_price(50, 8), heat_demand_level=40.0,
                 flex_price_level=_flex_price(50, 8),
                 carbon_factor=0.150, carbon_price=65.0,
                 land_limit_m2=830, construction_delay_frac=0.03),

        # --- N2 node (Central Europe) ---
        Scenario(9, "Streaming-France", "streaming", "France",
                 base_it_mw=20, flex_frac=0.20, base_pue=1.30,
                 heat_recovery_frac=0.35, hp_cop=3.0,
                 el_price_level=72, el_price_vol=26,
                 heat_price_level=_heat_price(72, 9), heat_demand_level=28.0,
                 flex_price_level=_flex_price(72, 9),
                 carbon_factor=0.056, carbon_price=65.0,
                 land_limit_m2=780, construction_delay_frac=0.12),

        # --- N3 node (UK) ---
        Scenario(10, "Hyperscale-UnitedKingdom", "hyperscale", "United Kingdom",
                 base_it_mw=32, flex_frac=0.16, base_pue=1.20,
                 heat_recovery_frac=0.35, hp_cop=3.0,
                 el_price_level=78, el_price_vol=25,
                 heat_price_level=_heat_price(78, 10), heat_demand_level=25.0,
                 flex_price_level=_flex_price(78, 10),
                 carbon_factor=0.207, carbon_price=65.0,
                 land_limit_m2=950, construction_delay_frac=0.08),
    ]


def apply_global_heat_assumptions(scenarios: List[Scenario], cfg: Config) -> None:
    if cfg.override_heat_recovery_frac is not None:
        frac = float(cfg.override_heat_recovery_frac)
        frac = max(0.0, min(1.0, frac))
        for sc in scenarios:
            sc.heat_recovery_frac = frac
    else:
        for sc in scenarios:
            sc.heat_recovery_frac = min(sc.heat_recovery_frac, 0.50)

    if cfg.override_hp_cop is not None:
        cop = float(cfg.override_hp_cop)
        cop = max(0.1, cop)
        for sc in scenarios:
            sc.hp_cop = cop
    for sc in scenarios:
        sc.flex_frac *= 0.90
