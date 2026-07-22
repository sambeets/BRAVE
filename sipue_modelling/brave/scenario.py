from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

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
    construction_delay_months: float = 0.0
    uncertainty_id: int = 0
    archetype_key: str = "legacy"
    archetype_name: str = "Legacy mixed case"
    carbon_damage_eur_per_tco2: Optional[float] = None
    congestion_multiplier: float = 1.0
    capex_multiplier: float = 1.0
    land_cost_multiplier: float = 1.0
    cooling_overhead_multiplier: float = 1.0
    free_cooling_multiplier: float = 1.0
    storage_availability: float = 1.0


@dataclass(frozen=True)
class DataCenterArchetype:
    key: str
    name: str
    dc_type: str
    base_it_mw: float
    flex_frac: float
    base_pue: float
    heat_recovery_frac: float
    hp_cop: float
    land_limit_m2: float


@dataclass(frozen=True)
class UncertaintyState:
    uid: int
    name: str
    location: str
    electricity_price_eur_per_mwh: float
    electricity_price_volatility: float
    heat_price_eur_per_mwhth: float
    heat_demand_mwth: float
    flexibility_price_eur_per_mw_h: float
    ets_price_eur_per_tco2: float
    carbon_damage_eur_per_tco2: float
    construction_delay_months: float
    congestion_multiplier: float
    cooling_overhead_multiplier: float
    free_cooling_multiplier: float
    heat_recovery_multiplier: float
    hp_cop_multiplier: float
    flex_availability_multiplier: float
    storage_availability: float
    capex_multiplier: float
    land_cost_multiplier: float


DATA_CENTER_ARCHETYPES: Tuple[DataCenterArchetype, ...] = (
    DataCenterArchetype(
        "ai_hpc", "AI/HPC", "ai", 40.0, 0.24, 1.12, 0.60, 3.8, 1500.0,
    ),
    DataCenterArchetype(
        "hyperscale", "Hyperscale cloud", "hyperscale", 60.0, 0.16, 1.18, 0.45, 3.5, 1900.0,
    ),
    DataCenterArchetype(
        "colocation", "Colocation", "colocation", 25.0, 0.08, 1.30, 0.32, 3.1, 1050.0,
    ),
    DataCenterArchetype(
        "enterprise_edge", "Enterprise/edge", "enterprise_edge", 8.0, 0.05, 1.42, 0.20, 2.8, 620.0,
    ),
)


# These ten states are applied unchanged to every facility archetype. They are
# uncertainty states, not ten different data-center types.
UNCERTAINTY_STATES: Tuple[UncertaintyState, ...] = (
    UncertaintyState(1, "central-base", "Netherlands", 65, 18, 58, 50, 18, 65, 120, 6, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    UncertaintyState(2, "clean-cool", "Norway", 48, 13, 48, 62, 16, 45, 80, 0, 1.20, 0.82, 1.25, 1.10, 1.08, 0.95, 1.10, 0.90, 0.80),
    UncertaintyState(3, "high-ets", "Germany", 72, 24, 62, 48, 21, 105, 160, 6, 0.90, 1.05, 0.95, 0.95, 0.95, 0.90, 0.90, 1.00, 1.10),
    UncertaintyState(4, "heat-rich", "Denmark", 58, 20, 78, 78, 20, 75, 140, 0, 0.85, 0.92, 1.15, 1.15, 1.05, 1.00, 1.00, 1.05, 0.95),
    UncertaintyState(5, "flex-rich", "United Kingdom", 82, 30, 52, 30, 31, 80, 130, 12, 0.75, 1.08, 0.90, 0.90, 0.90, 1.20, 1.00, 1.00, 1.05),
    UncertaintyState(6, "cooling-stress", "France", 70, 22, 55, 38, 18, 70, 110, 6, 1.05, 1.28, 0.70, 0.85, 0.92, 0.90, 0.90, 1.10, 1.00),
    UncertaintyState(7, "congestion-stress", "Belgium", 76, 27, 60, 44, 24, 85, 180, 6, 0.60, 1.12, 0.90, 1.00, 0.95, 1.05, 0.90, 1.05, 1.10),
    UncertaintyState(8, "storage-limited", "Sweden", 54, 19, 50, 55, 17, 60, 100, 0, 1.10, 0.88, 1.20, 1.05, 1.05, 0.95, 0.35, 1.00, 0.90),
    UncertaintyState(9, "cost-stress", "Netherlands", 74, 25, 56, 42, 19, 90, 200, 12, 0.80, 1.15, 0.85, 0.90, 0.90, 0.90, 0.80, 1.35, 1.30),
    UncertaintyState(10, "service-upside", "Denmark", 60, 21, 82, 70, 29, 75, 150, 6, 0.95, 0.95, 1.10, 1.15, 1.08, 1.15, 1.15, 0.90, 0.95),
)


def make_archetype_scenarios(
    cfg: Config,
    archetype: DataCenterArchetype,
) -> List[Scenario]:
    """Compose one facility archetype with the ten shared uncertainty states."""
    archetype_index = next(
        index for index, candidate in enumerate(DATA_CENTER_ARCHETYPES, start=1)
        if candidate.key == archetype.key
    )
    rows: List[Scenario] = []
    for state in UNCERTAINTY_STATES:
        sid = (archetype_index - 1) * len(UNCERTAINTY_STATES) + state.uid
        heat_price = cfg.heat_price_multiplier * state.heat_price_eur_per_mwhth
        flex_price = cfg.flex_price_multiplier * state.flexibility_price_eur_per_mw_h
        rows.append(Scenario(
            sid=sid,
            name=f"{archetype.key}-{state.name}",
            dc_type=archetype.dc_type,
            location=state.location,
            base_it_mw=archetype.base_it_mw,
            flex_frac=min(0.30, archetype.flex_frac * state.flex_availability_multiplier),
            base_pue=archetype.base_pue,
            heat_recovery_frac=min(0.90, archetype.heat_recovery_frac * state.heat_recovery_multiplier),
            hp_cop=max(1.1, archetype.hp_cop * state.hp_cop_multiplier),
            el_price_level=state.electricity_price_eur_per_mwh,
            el_price_vol=state.electricity_price_volatility,
            heat_price_level=heat_price,
            heat_demand_level=state.heat_demand_mwth,
            flex_price_level=flex_price,
            carbon_factor=0.0,
            carbon_price=state.ets_price_eur_per_tco2,
            land_limit_m2=archetype.land_limit_m2,
            construction_delay_months=state.construction_delay_months,
            uncertainty_id=state.uid,
            archetype_key=archetype.key,
            archetype_name=archetype.name,
            carbon_damage_eur_per_tco2=(
                state.carbon_damage_eur_per_tco2 * max(0.0, cfg.carbon_damage_multiplier)
            ),
            congestion_multiplier=state.congestion_multiplier,
            capex_multiplier=state.capex_multiplier,
            land_cost_multiplier=state.land_cost_multiplier,
            cooling_overhead_multiplier=state.cooling_overhead_multiplier,
            free_cooling_multiplier=state.free_cooling_multiplier,
            storage_availability=state.storage_availability,
        ))
    return rows


def make_study_scenarios(cfg: Config) -> Dict[str, List[Scenario]]:
    return {
        archetype.key: make_archetype_scenarios(cfg, archetype)
        for archetype in DATA_CENTER_ARCHETYPES
    }


def make_out_of_sample_scenarios(
    cfg: Config,
    archetype: DataCenterArchetype,
    count: int,
    seed: int,
) -> List[Scenario]:
    """Generate uncertainty cases that are independent of the ten design states."""
    rng = random.Random(seed)
    locations = [state.location for state in UNCERTAINTY_STATES]

    def span(attribute: str) -> Tuple[float, float]:
        values = [float(getattr(state, attribute)) for state in UNCERTAINTY_STATES]
        return min(values), max(values)

    bounds = {
        name: span(name)
        for name in (
            "electricity_price_eur_per_mwh", "electricity_price_volatility",
            "heat_price_eur_per_mwhth", "heat_demand_mwth",
            "flexibility_price_eur_per_mw_h", "ets_price_eur_per_tco2",
            "carbon_damage_eur_per_tco2", "congestion_multiplier",
            "cooling_overhead_multiplier", "free_cooling_multiplier",
            "heat_recovery_multiplier", "hp_cop_multiplier",
            "flex_availability_multiplier", "storage_availability",
            "capex_multiplier", "land_cost_multiplier",
        )
    }

    def draw(name: str) -> float:
        low, high = bounds[name]
        return rng.uniform(low, high)

    archetype_index = next(
        index for index, candidate in enumerate(DATA_CENTER_ARCHETYPES, start=1)
        if candidate.key == archetype.key
    )
    scenarios = []
    for index in range(1, max(0, int(count)) + 1):
        heat_multiplier = draw("heat_recovery_multiplier")
        cop_multiplier = draw("hp_cop_multiplier")
        flex_multiplier = draw("flex_availability_multiplier")
        scenarios.append(Scenario(
            sid=10_000 + archetype_index * 1_000 + index,
            name=f"{archetype.key}-oos-{index:03d}",
            dc_type=archetype.dc_type,
            location=rng.choice(locations),
            base_it_mw=archetype.base_it_mw,
            flex_frac=min(0.30, archetype.flex_frac * flex_multiplier),
            base_pue=archetype.base_pue,
            heat_recovery_frac=min(0.90, archetype.heat_recovery_frac * heat_multiplier),
            hp_cop=max(1.1, archetype.hp_cop * cop_multiplier),
            el_price_level=draw("electricity_price_eur_per_mwh"),
            el_price_vol=draw("electricity_price_volatility"),
            heat_price_level=cfg.heat_price_multiplier * draw("heat_price_eur_per_mwhth"),
            heat_demand_level=draw("heat_demand_mwth"),
            flex_price_level=cfg.flex_price_multiplier * draw("flexibility_price_eur_per_mw_h"),
            carbon_factor=0.0,
            carbon_price=draw("ets_price_eur_per_tco2"),
            land_limit_m2=archetype.land_limit_m2,
            construction_delay_months=rng.choice((0.0, 6.0, 12.0)),
            uncertainty_id=index,
            archetype_key=archetype.key,
            archetype_name=archetype.name,
            carbon_damage_eur_per_tco2=(
                draw("carbon_damage_eur_per_tco2") * max(0.0, cfg.carbon_damage_multiplier)
            ),
            congestion_multiplier=draw("congestion_multiplier"),
            capex_multiplier=draw("capex_multiplier"),
            land_cost_multiplier=draw("land_cost_multiplier"),
            cooling_overhead_multiplier=draw("cooling_overhead_multiplier"),
            free_cooling_multiplier=draw("free_cooling_multiplier"),
            storage_availability=draw("storage_availability"),
        ))
    return scenarios


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
    # Land and integration-capacity assumptions are read from Config.
    return [
        # --- N2 node (Central Europe) ---
        Scenario(1, "Hyperscale-Netherlands", "hyperscale", "Netherlands",
                 base_it_mw=40, flex_frac=0.15, base_pue=1.18,
                 heat_recovery_frac=0.45, hp_cop=3.5,
                 el_price_level=58, el_price_vol=20,
                 heat_price_level=_heat_price(58, 1), heat_demand_level=65.0,
                 flex_price_level=_flex_price(58, 1),
                 carbon_factor=0.328, carbon_price=65.0,
                 land_limit_m2=1100, construction_delay_months=6.0),

        # --- N1 node (Nordics) ---
        Scenario(2, "AI-Norway", "ai", "Norway",
                 base_it_mw=25, flex_frac=0.12, base_pue=1.12,
                 heat_recovery_frac=0.50, hp_cop=3.6,
                 el_price_level=42, el_price_vol=15,
                 heat_price_level=_heat_price(42, 2), heat_demand_level=45.0,
                 flex_price_level=_flex_price(42, 2),
                 carbon_factor=0.029, carbon_price=65.0,
                 land_limit_m2=850, construction_delay_months=0.0),

        # --- N2 node (Central Europe) ---
        Scenario(3, "Streaming-Belgium", "streaming", "Belgium",
                 base_it_mw=18, flex_frac=0.22, base_pue=1.32,
                 heat_recovery_frac=0.40, hp_cop=3.2,
                 el_price_level=65, el_price_vol=28,
                 heat_price_level=_heat_price(65, 3), heat_demand_level=38.0,
                 flex_price_level=_flex_price(65, 3),
                 carbon_factor=0.148, carbon_price=65.0,
                 land_limit_m2=750, construction_delay_months=6.0),

        # --- N1 node (Nordics) ---
        Scenario(4, "Hyperscale-Sweden", "hyperscale", "Sweden",
                 base_it_mw=35, flex_frac=0.14, base_pue=1.08,
                 heat_recovery_frac=0.50, hp_cop=3.4,
                 el_price_level=45, el_price_vol=16,
                 heat_price_level=_heat_price(45, 4), heat_demand_level=50.0,
                 flex_price_level=_flex_price(45, 4),
                 carbon_factor=0.015, carbon_price=65.0,
                 land_limit_m2=1000, construction_delay_months=0.0),

        # --- N3 node (UK) ---
        Scenario(5, "Streaming-UnitedKingdom", "streaming", "United Kingdom",
                 base_it_mw=16, flex_frac=0.24, base_pue=1.35,
                 heat_recovery_frac=0.30, hp_cop=2.8,
                 el_price_level=82, el_price_vol=30,
                 heat_price_level=_heat_price(82, 5), heat_demand_level=20.0,
                 flex_price_level=_flex_price(82, 5),
                 carbon_factor=0.207, carbon_price=65.0,
                 land_limit_m2=700, construction_delay_months=12.0),

        # --- N1 node (Nordics) ---
        Scenario(6, "AI-Sweden", "ai", "Sweden",
                 base_it_mw=20, flex_frac=0.10, base_pue=1.14,
                 heat_recovery_frac=0.45, hp_cop=3.3,
                 el_price_level=48, el_price_vol=18,
                 heat_price_level=_heat_price(48, 6), heat_demand_level=30.0,
                 flex_price_level=_flex_price(48, 6),
                 carbon_factor=0.015, carbon_price=65.0,
                 land_limit_m2=800, construction_delay_months=0.0),

        # --- N2 node (Central Europe) ---
        Scenario(7, "Colocation-Germany", "colocation", "Germany",
                 base_it_mw=22, flex_frac=0.08, base_pue=1.35,
                 heat_recovery_frac=0.35, hp_cop=3.1,
                 el_price_level=62, el_price_vol=22,
                 heat_price_level=_heat_price(62, 7), heat_demand_level=42.0,
                 flex_price_level=_flex_price(62, 7),
                 carbon_factor=0.385, carbon_price=65.0,
                 land_limit_m2=820, construction_delay_months=6.0),

        # --- N1 node (Nordics) ---
        Scenario(8, "AI-Denmark", "ai", "Denmark",
                 base_it_mw=22, flex_frac=0.13, base_pue=1.13,
                 heat_recovery_frac=0.50, hp_cop=3.5,
                 el_price_level=50, el_price_vol=20,
                 heat_price_level=_heat_price(50, 8), heat_demand_level=40.0,
                 flex_price_level=_flex_price(50, 8),
                 carbon_factor=0.150, carbon_price=65.0,
                 land_limit_m2=830, construction_delay_months=0.0),

        # --- N2 node (Central Europe) ---
        Scenario(9, "Streaming-France", "streaming", "France",
                 base_it_mw=20, flex_frac=0.20, base_pue=1.30,
                 heat_recovery_frac=0.35, hp_cop=3.0,
                 el_price_level=72, el_price_vol=26,
                 heat_price_level=_heat_price(72, 9), heat_demand_level=28.0,
                 flex_price_level=_flex_price(72, 9),
                 carbon_factor=0.056, carbon_price=65.0,
                 land_limit_m2=780, construction_delay_months=12.0),

        # --- N3 node (UK) ---
        Scenario(10, "Hyperscale-UnitedKingdom", "hyperscale", "United Kingdom",
                 base_it_mw=32, flex_frac=0.16, base_pue=1.20,
                 heat_recovery_frac=0.35, hp_cop=3.0,
                 el_price_level=78, el_price_vol=25,
                 heat_price_level=_heat_price(78, 10), heat_demand_level=25.0,
                 flex_price_level=_flex_price(78, 10),
                 carbon_factor=0.207, carbon_price=65.0,
                 land_limit_m2=950, construction_delay_months=6.0),
    ]


def apply_global_heat_assumptions(scenarios: List[Scenario], cfg: Config) -> None:
    if cfg.override_heat_recovery_frac is not None:
        frac = float(cfg.override_heat_recovery_frac)
        frac = max(0.0, min(1.0, frac))
        for sc in scenarios:
            sc.heat_recovery_frac = frac
    else:
        for sc in scenarios:
            limit = 0.50 if sc.archetype_key == "legacy" else 0.90
            sc.heat_recovery_frac = min(sc.heat_recovery_frac, limit)

    if cfg.override_hp_cop is not None:
        cop = float(cfg.override_hp_cop)
        cop = max(0.1, cop)
        for sc in scenarios:
            sc.hp_cop = cop
    if cfg.override_construction_delay_months is not None:
        delay = max(0.0, float(cfg.override_construction_delay_months))
        for sc in scenarios:
            sc.construction_delay_frac = 0.0
            sc.construction_delay_months = delay
    for sc in scenarios:
        sc.heat_demand_level *= max(0.0, cfg.heat_demand_multiplier)
        legacy_derating = 0.90 if sc.archetype_key == "legacy" else 1.0
        sc.flex_frac *= legacy_derating * max(0.0, cfg.flex_availability_multiplier)
