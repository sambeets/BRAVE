from __future__ import annotations

import csv
import os
from typing import Dict, List

from brave.config import Config
from brave.scenario import DATA_CENTER_ARCHETYPES


DEA_HEAT = "https://ens.dk/en/analyses-and-statistics/technology-data-generation-electricity-and-district-heating"
DEA_HEAT_BRIEF = "https://ens.dk/media/8616/download"
DEA_STORAGE = "https://ens.dk/media/8622/download"
NREL_BATTERY_2025 = "https://research-hub.nrel.gov/en/publications/cost-projections-for-utility-scale-battery-storage-2025-update/"
FINGRID_FCR = "https://www.fingrid.fi/en/news/news/2024/procurement-of-frequency-containment-reserves-from-annual-market-2025/"
EC_ETS = "https://climate.ec.europa.eu/eu-action/carbon-markets/eu-emissions-trading-system-eu-ets/auctioning-allowances_en"
EC_SHADOW_CARBON = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021XC0916(03)"
DC_HEAT_COP = "https://doi.org/10.54337/ijsepm.8149"
DC_COOLING = "https://www.sciencedirect.com/science/article/abs/pii/S1359431124008573"
DC_DELAY = "https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook-new.pdf"
# Archetype-level heat-recovery / flexibility differentiation (added 2026-07-23,
# see context.md's independent-review pass on why the archetype table needed
# citations: it was previously bare literals with no evidence trail at all).
ACEEE_NADEL_2025 = "https://www.aceee.org/white-paper/2025/10/opportunities-use-energy-efficiency-and-demand-flexibility-reduce-data-center"
DC_COLO_DR_2015 = "https://arxiv.org/pdf/1504.07308"
DC_LIQUID_COOLING_MARKET = "https://www.grandviewresearch.com/industry-analysis/data-center-liquid-cooling-market-report"
# Land-economics evidence pass (added 2026-07-29): only a_B,E had a
# verifiable, fetchable industry figure; land_cost_eur_per_m2, a_core,
# a_cool, a_HX, a_HP, and a_flex remain genuine evidence gaps after a real
# search attempt (JLL/CBRE/Turner & Townsend European data-centre reports
# discuss land scarcity and price trends narratively but do not publish a
# clean EUR/m2/year figure comparable across markets; no equipment-footprint
# study was found for heat exchangers, heat pumps, or flexibility assets).
BESS_FOOTPRINT_BENCHMARK = "https://sunnyplans.com/solar/blog/bess-land-requirements-usa/"


_ARCHETYPES_BY_KEY = {a.key: a for a in DATA_CENTER_ARCHETYPES}


def parameter_provenance(cfg: Config) -> List[Dict]:
    """Return the canonical parameter ledger without silently filling evidence gaps."""
    ai_hpc = _ARCHETYPES_BY_KEY["ai_hpc"]
    hyperscale = _ARCHETYPES_BY_KEY["hyperscale"]
    colocation = _ARCHETYPES_BY_KEY["colocation"]
    enterprise_edge = _ARCHETYPES_BY_KEY["enterprise_edge"]
    return [
        _row("ETS price", "p_CO2", 65.0, "EUR/tCO2", "60-80", 2025, "2025 EUR", "none", EC_ETS, "within source range"),
        _row("carbon damage", "d_CO2", cfg.independent_social_carbon_cost_eur_per_tco2, "EUR/tCO2", "80-250", 2021, "2016 EUR", "range retained; tested in 2025-EUR sensitivity", EC_SHADOW_CARBON, "within policy appraisal range"),
        _row("flexibility value", "v_flex", cfg.independent_avoided_flex_cost_eur_per_mw_h, "EUR/MW-h", "4.00-25.30", 2025, "2025 EUR", "none", FINGRID_FCR, "direct FCR benchmark"),
        _row("avoided heat cost", "v_heat", cfg.independent_avoided_heat_cost_eur_per_mwhth, "EUR/MWhth", "20-80", 2025, "2025 EUR", "none", DEA_HEAT, "scenario assumption within catalogue context"),
        _row("heat-exchanger capex", "c_HX", cfg.capex_HX, "EUR/MWth", "150000-350000", 2025, "2025 EUR", "upfront; no annualization", DEA_HEAT, "study assumption within engineering range"),
        _row("heat-pump capex", "c_HP", cfg.capex_HP, "EUR/MWe", "400000-1400000", 2025, "2025 EUR", "upfront; electrical-input capacity basis", DEA_HEAT_BRIEF, "study assumption; basis disclosed"),
        _row("flex-integration capex", "c_flex", cfg.capex_FLEX, "EUR/MW", "100000-400000", 2025, "2025 EUR", "upfront; no annualization", FINGRID_FCR, "study assumption; sensitivity required"),
        _row("battery energy capex", "c_B,E", cfg.capex_BAT_E, "EUR/MWh", "150000-450000", 2020, "USD", "catalogue USD converted and escalated to 2025 EUR", DEA_STORAGE, "within source-supported range"),
        _row("battery power capex", "c_B,P", cfg.capex_BAT_P, "EUR/MW", "80000-300000", 2020, "USD", "catalogue USD converted and escalated to 2025 EUR", DEA_STORAGE, "within source-supported range"),
        _row("battery capacity headroom multiplier", "m_B,max", cfg.battery_capacity_headroom_multiplier, "multiple of minimum battery floor", "study default 2.0; diagnostic sweep required", 2025, "not monetary", "none", NREL_BATTERY_2025, "study assumption informed by 4-hour utility-scale storage benchmark; sensitivity required"),
        _row("battery charge efficiency", "eta_ch", cfg.eta_ch, "fraction", "0.95-0.99", 2020, "not monetary", "none", DEA_STORAGE, "direct technical range"),
        _row("battery discharge efficiency", "eta_dis", cfg.eta_dis, "fraction", "0.95-0.98", 2020, "not monetary", "none", DEA_STORAGE, "direct technical range"),
        _row("heat-pump COP", "COP", cfg.cop_ref, "dimensionless", "2.2-5.5", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _row("heat recovery fraction", "r_heat", 0.40, "fraction", "0.30-0.50 base range; canonical effective values may exceed this after state multipliers", 2024, "not monetary", "none", DC_HEAT_COP, "base archetype range; state multipliers model local heat-reuse opportunity"),
        # Archetype-level heat_recovery_frac, hp_cop, and flex_frac (added
        # 2026-07-23): these differentiate the four DATA_CENTER_ARCHETYPES in
        # brave/scenario.py and materially affect service-credit diagnostics.
        # They previously had no citation anywhere in the codebase; see
        # scenario.py's comment above DATA_CENTER_ARCHETYPES for the full
        # revision rationale.
        _row("heat recovery fraction (AI/HPC)", "phi_ai_hpc", ai_hpc.heat_recovery_frac, "fraction", "0.30-0.50 base range; effective canonical values up to 0.575 after state multipliers", 2025, "not monetary", "none", DC_LIQUID_COOLING_MARKET, "base value capped at generic cited ceiling; favorable uncertainty states can raise the operational LP input"),
        _row("heat recovery fraction (Hyperscale cloud)", "phi_hyperscale", hyperscale.heat_recovery_frac, "fraction", "0.30-0.50 base range; effective canonical values up to 0.5175 after state multipliers", 2024, "not monetary", "none", DC_HEAT_COP, "base value within generic source range; favorable uncertainty states can raise the operational LP input"),
        _row("heat recovery fraction (Colocation)", "phi_colocation", colocation.heat_recovery_frac, "fraction", "0.30-0.50 base range; effective canonical values remain below 0.50 after state multipliers", 2024, "not monetary", "none", DC_HEAT_COP, "base and effective canonical values remain within generic source range"),
        _row("heat recovery fraction (Enterprise/edge)", "phi_enterprise_edge", enterprise_edge.heat_recovery_frac, "fraction", "evidence gap; below generic 0.30-0.50 floor", 2024, "not monetary", "none", "", "no archetype-specific source found; study assumption; sensitivity required"),
        _row("heat-pump COP (AI/HPC)", "COP_ai_hpc", ai_hpc.hp_cop, "dimensionless", "2.2-5.5 (generic data-centre range)", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _row("heat-pump COP (Hyperscale cloud)", "COP_hyperscale", hyperscale.hp_cop, "dimensionless", "2.2-5.5 (generic data-centre range)", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _row("heat-pump COP (Colocation)", "COP_colocation", colocation.hp_cop, "dimensionless", "2.2-5.5 (generic data-centre range)", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _row("heat-pump COP (Enterprise/edge)", "COP_enterprise_edge", enterprise_edge.hp_cop, "dimensionless", "2.2-5.5 (generic data-centre range)", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _row("flexibility fraction (AI/HPC)", "flex_ai_hpc", ai_hpc.flex_frac, "fraction of IT load", "up to ~0.25 demonstrated", 2025, "not monetary", "none", ACEEE_NADEL_2025, "close to a demonstrated hyperscale/AI figure (Oracle 25% peak-hour reduction via GPU workload orchestration)"),
        _row("flexibility fraction (Hyperscale cloud)", "flex_hyperscale", hyperscale.flex_frac, "fraction of IT load", "evidence gap", 2025, "not monetary", "none", "", "no archetype-specific source found; study assumption; sensitivity required"),
        _row("flexibility fraction (Colocation)", "flex_colocation", colocation.flex_frac, "fraction of IT load", "0.10-0.25 (multi-tenant colocation demand response)", 2015, "not monetary", "none", DC_COLO_DR_2015, "at lower bound of source-supported range"),
        _row("flexibility fraction (Enterprise/edge)", "flex_enterprise_edge", enterprise_edge.flex_frac, "fraction of IT load", "evidence gap", 2025, "not monetary", "none", "", "no archetype-specific source found; study assumption; sensitivity required"),
        _row("cooling electricity share", "e_cool", 0.138, "fraction of facility energy", "0.10-0.45", 2024, "not monetary", "none", DC_COOLING, "field-study benchmark"),
        _row("construction delay", "delta", 6.0, "months", "0-12", 2026, "not monetary", "none", DC_DELAY, "scenario stress range"),
        _row("real discount rate", "r", cfg.real_discount_rate, "fraction/year", "0.03-0.05", 2026, "real", "nominal cash flows excluded", DEA_HEAT_BRIEF, "canonical planning assumption"),
        _row("project horizon", "N", cfg.lifecycle_years, "years", "10-20", 2026, "not monetary", "none", DEA_HEAT_BRIEF, "study assumption"),
        _row("land lease or opportunity cost", "c_land", cfg.land_cost_eur_per_m2, "EUR/m2-year", "evidence gap; sensitivity 17.5-70", 2025, "2025 EUR", "annual cost from project initiation", "", "study assumption; direct source required"),
        _row("existing core footprint intensity", "a_core", cfg.existing_land_m2_per_it_mw, "m2/MW IT", "evidence gap; sensitivity 4-12", 2025, "not monetary", "none", "", "study assumption; direct source required"),
        _row("cooling footprint coefficient", "a_cool", cfg.land_a_cooling, "m2/MW cooling", "evidence gap; sensitivity 2.5-7.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _row("heat-exchanger footprint coefficient", "a_HX", cfg.land_a_hx, "m2/MWth", "evidence gap; sensitivity 15-45", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _row("heat-pump footprint coefficient", "a_HP", cfg.land_a_hp, "m2/MWe", "evidence gap; sensitivity 6-18", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _row("flexibility footprint coefficient", "a_flex", cfg.land_a_flex, "m2/MW", "evidence gap; sensitivity 7.5-22.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _row("battery energy footprint coefficient", "a_B,E", cfg.land_a_bat_e, "m2/MWh", "~93 m2/MWh industry benchmark (container footprint only, excludes site setbacks/roads/fencing)", 2025, "not monetary", "none", BESS_FOOTPRINT_BENCHMARK, "study default is well below this benchmark; not yet reconciled, sensitivity required"),
        _row("battery power footprint coefficient", "a_B,P", cfg.land_a_bat_p, "m2/MW", "evidence gap; sensitivity 2.5-7.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _row("peak capacity value", "c_peak", cfg.independent_peak_capacity_value_eur_per_mw_year, "EUR/MW-year", "25000-100000", 2025, "2025 EUR", "none", "", "study assumption; sensitivity required"),
        _row("transmission cost", "c_grid", cfg.transmission_cost_eur_per_mwh, "EUR/MWh-flow", "4-16", 2025, "2025 EUR", "none", "", "study assumption; sensitivity required"),
        _row("siPUE reference price", "p_ref", cfg.sipue_system_ref_price_eur_per_mwh, "EUR/MWh", "0.5-1.5 times canonical", 2025, "2025 EUR", "endogenous no-DC resource marginal cost; ETS and damage excluded", "", "model-derived and frozen before optimization"),
        _row("line 1-2 capacity", "F12_max", cfg.line_cap_12, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _row("line 2-3 capacity", "F23_max", cfg.line_cap_23, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _row("line 1-3 capacity", "F13_max", cfg.line_cap_13, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _row("physical emissions reduction", "epsilon_CO2", 100.0 * cfg.carbon_cap_reduction_frac, "% of DC-attributable no-services emissions", "0-10", 2025, "not monetary", "steady-state same-IT counterfactual", "", "policy stress target; sensitivity required"),
    ]


def _row(
    parameter: str,
    symbol: str,
    canonical_value: float,
    unit: str,
    source_supported_range: str,
    source_year: int,
    currency_basis: str,
    conversion: str,
    source: str,
    evidence_status: str,
) -> Dict:
    return {
        "parameter": parameter,
        "symbol": symbol,
        "canonical_value": canonical_value,
        "unit": unit,
        "source_supported_range": source_supported_range,
        "source_year": source_year,
        "currency_basis": currency_basis,
        "conversion_to_2025_eur": conversion,
        "source": source,
        "evidence_status": evidence_status,
    }


def export_parameter_provenance(cfg: Config, path: str, run_id: str = "") -> None:
    rows = parameter_provenance(cfg)
    if run_id:
        for row in rows:
            row["run_id"] = run_id
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
