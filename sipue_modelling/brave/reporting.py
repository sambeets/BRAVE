"""CSV reporting artifacts.

Emits the study's documentation and results tables: scenario/results exports,
the canonical parameter-provenance ledger, and the literature comparison matrix
that delimits the siPUE contribution. All three are pure serializers with no
optimization dependency, so they live together here.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List

from brave.config import Config


DEA_HEAT = "https://ens.dk/en/analyses-and-statistics/technology-data-generation-electricity-and-district-heating"
DEA_HEAT_BRIEF = "https://ens.dk/media/8616/download"
DEA_STORAGE = "https://ens.dk/media/8622/download"
FINGRID_FCR = "https://www.fingrid.fi/en/electricity-market-information/reserve-market-information/frequency-controlled-disturbance-reserve/"
NORDIC_FCR_DIM = "https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/SOC%20documents/Nordic/2023/explanatory-document-for-the-amended-nordic-synchronous-area-methodology-for-the-dimensioning-rules-for-fcr.pdf"
DC_FCRD_CASE = "https://www.globenewswire.com/news-release/2026/07/06/vivopower-battery-norway-data-center"
EC_ETS = "https://climate.ec.europa.eu/eu-action/carbon-markets/eu-emissions-trading-system-eu-ets/auctioning-allowances_en"
EC_SHADOW_CARBON = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021XC0916(03)"
DC_HEAT_COP = "https://doi.org/10.54337/ijsepm.8149"
DC_COOLING = "https://www.sciencedirect.com/science/article/abs/pii/S1359431124008573"
DC_DELAY = "https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook-new.pdf"

_BANNED_OUTPUT_TOKENS = ("operator", "settlement", "unitprofit", "profitpct")


# ── generic results / scenario exports ──────────────────────────────────────

def _write_csv(rows: Iterable[Dict], path: str) -> None:
    data = list(rows)
    if not data:
        return
    fields = sorted({key for row in data for key in row})
    banned = [field for field in fields if any(token in field.lower() for token in _BANNED_OUTPUT_TOKENS)]
    if banned:
        raise ValueError(f"Operator-facing fields are not permitted in canonical output: {banned}")
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def export_scenarios_csv(scenarios, path: str) -> None:
    _write_csv((dict(sc.__dict__) for sc in scenarios), path)


def export_results_csv(rows: List[Dict], path: str) -> None:
    _write_csv(rows, path)


def summarize_interpretation(rows: List[Dict], cfg=None) -> None:
    if not rows:
        print("No results available.")
        return
    avg_value = sum(float(row.get("MarketValueNPVEUR_robust", 0.0)) for row in rows) / len(rows)
    avg_benefit = sum(float(row.get("SystemBenefitPerMWhIT_robust", 0.0)) for row in rows) / len(rows)
    carbon_ok = sum(
        float(row.get("EmissionsTCO2_robust", float("inf")))
        <= float(row.get("CarbonCapTCO2", -float("inf"))) + 1e-6
        for row in rows
    )
    print(f"Scenarios: {len(rows)}")
    print(f"Average robust lifecycle market value: EUR {avg_value:,.0f}")
    print(f"Average delay-adjusted system benefit: EUR {avg_benefit:,.2f}/MWh IT")
    print(f"Physical carbon cap satisfied: {carbon_ok}/{len(rows)}")


def _write_documentation_table(rows: List[Dict], path: str, run_id: str = "") -> None:
    if run_id:
        for row in rows:
            row["RunID"] = run_id
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── parameter provenance ledger ─────────────────────────────────────────────

def parameter_provenance(cfg: Config) -> List[Dict]:
    """Return the canonical parameter ledger without silently filling evidence gaps."""
    return [
        _provenance_row("ETS price", "p_CO2", 65.0, "EUR/tCO2", "60-80", 2025, "2025 EUR", "none", EC_ETS, "within source range"),
        _provenance_row("carbon damage", "d_CO2", cfg.independent_social_carbon_cost_eur_per_tco2, "EUR/tCO2", "80-250", 2021, "2016 EUR", "range retained; tested in 2025-EUR sensitivity", EC_SHADOW_CARBON, "within policy appraisal range"),
        _provenance_row("flexibility value", "v_flex", cfg.independent_avoided_flex_cost_eur_per_mw_h, "EUR/MW-h", "4.00-25.30", 2025, "2025 EUR", "none", FINGRID_FCR, "assumed-price benchmark; retained only for sensitivity vs the endogenous reserve channel"),
        _provenance_row("operating-reserve requirement", "gamma_res", cfg.reserve_requirement_frac, "fraction of DC load", "0.03-0.05 total operating reserve; Nordic FCR ~0.034 of peak", 2024, "not monetary", "system reserve ratio applied to the DC's own incremental load", NORDIC_FCR_DIM, "upper end of NERC/UCTE 3-5% operating-reserve standard"),
        _provenance_row("thermal reserve procurement cost", "c_res", cfg.thermal_reserve_cost_eur_per_mw_h, "EUR/MW-h", "FCR-D up 2.8-4.0, down 9.5-10.0 (Fingrid annual 2023-24); FCR-N 19-25 tested in sensitivity", 2024, "2024 EUR", "none", FINGRID_FCR, "central Nordic FCR-D price; FCR-D is the reserve a load-shedding DC provides (41.5 MW VivoPower DC enrols 12 MW FCR-D)"),
        _provenance_row("avoided heat cost", "v_heat", cfg.independent_avoided_heat_cost_eur_per_mwhth, "EUR/MWhth", "20-80", 2025, "2025 EUR", "none", DEA_HEAT, "scenario assumption within catalogue context"),
        _provenance_row("heat-exchanger capex", "c_HX", cfg.capex_HX, "EUR/MWth", "150000-350000", 2025, "2025 EUR", "upfront; no annualization", DEA_HEAT, "study assumption within engineering range"),
        _provenance_row("heat-pump capex", "c_HP", cfg.capex_HP, "EUR/MWe", "400000-1400000", 2025, "2025 EUR", "upfront; electrical-input capacity basis", DEA_HEAT_BRIEF, "study assumption; basis disclosed"),
        _provenance_row("flex-integration capex", "c_flex", cfg.capex_FLEX, "EUR/MW", "100000-400000", 2025, "2025 EUR", "upfront; no annualization", FINGRID_FCR, "study assumption; sensitivity required"),
        _provenance_row("battery energy capex", "c_B,E", cfg.capex_BAT_E, "EUR/MWh", "150000-450000", 2020, "USD", "catalogue USD converted and escalated to 2025 EUR", DEA_STORAGE, "within source-supported range"),
        _provenance_row("battery power capex", "c_B,P", cfg.capex_BAT_P, "EUR/MW", "80000-300000", 2020, "USD", "catalogue USD converted and escalated to 2025 EUR", DEA_STORAGE, "within source-supported range"),
        _provenance_row("battery charge efficiency", "eta_ch", cfg.eta_ch, "fraction", "0.95-0.99", 2020, "not monetary", "none", DEA_STORAGE, "direct technical range"),
        _provenance_row("battery discharge efficiency", "eta_dis", cfg.eta_dis, "fraction", "0.95-0.98", 2020, "not monetary", "none", DEA_STORAGE, "direct technical range"),
        _provenance_row("heat-pump COP", "COP", cfg.cop_ref, "dimensionless", "2.2-5.5", 2024, "not monetary", "none", DC_HEAT_COP, "within published data-centre range"),
        _provenance_row("heat recovery fraction", "r_heat", 0.40, "fraction", "0.30-0.50", 2024, "not monetary", "none", DC_HEAT_COP, "scenario range"),
        _provenance_row("cooling electricity share", "e_cool", 0.138, "fraction of facility energy", "0.10-0.45", 2024, "not monetary", "none", DC_COOLING, "field-study benchmark"),
        _provenance_row("construction delay", "delta", 6.0, "months", "0-12", 2026, "not monetary", "none", DC_DELAY, "scenario stress range"),
        _provenance_row("real discount rate", "r", cfg.real_discount_rate, "fraction/year", "0.03-0.05", 2026, "real", "nominal cash flows excluded", DEA_HEAT_BRIEF, "canonical planning assumption"),
        _provenance_row("project horizon", "N", cfg.lifecycle_years, "years", "10-20", 2026, "not monetary", "none", DEA_HEAT_BRIEF, "study assumption"),
        _provenance_row("land lease or opportunity cost", "c_land", cfg.land_cost_eur_per_m2, "EUR/m2-year", "evidence gap; sensitivity 17.5-70", 2025, "2025 EUR", "annual cost from project initiation", "", "study assumption; direct source required"),
        _provenance_row("existing core footprint intensity", "a_core", cfg.existing_land_m2_per_it_mw, "m2/MW IT", "evidence gap; sensitivity 4-12", 2025, "not monetary", "none", "", "study assumption; direct source required"),
        _provenance_row("cooling footprint coefficient", "a_cool", cfg.land_a_cooling, "m2/MW cooling", "evidence gap; sensitivity 2.5-7.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("heat-exchanger footprint coefficient", "a_HX", cfg.land_a_hx, "m2/MWth", "evidence gap; sensitivity 15-45", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("heat-pump footprint coefficient", "a_HP", cfg.land_a_hp, "m2/MWe", "evidence gap; sensitivity 6-18", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("flexibility footprint coefficient", "a_flex", cfg.land_a_flex, "m2/MW", "evidence gap; sensitivity 7.5-22.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("battery energy footprint coefficient", "a_B,E", cfg.land_a_bat_e, "m2/MWh", "evidence gap; sensitivity 2.5-7.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("battery power footprint coefficient", "a_B,P", cfg.land_a_bat_p, "m2/MW", "evidence gap; sensitivity 2.5-7.5", 2025, "not monetary", "none", "", "study assumption; sensitivity required"),
        _provenance_row("peak capacity value", "c_peak", cfg.independent_peak_capacity_value_eur_per_mw_year, "EUR/MW-year", "25000-100000", 2025, "2025 EUR", "none", "", "study assumption; sensitivity required"),
        _provenance_row("transmission cost", "c_grid", cfg.transmission_cost_eur_per_mwh, "EUR/MWh-flow", "4-16", 2025, "2025 EUR", "none", "", "study assumption; sensitivity required"),
        _provenance_row("siPUE reference price", "p_ref", cfg.sipue_system_ref_price_eur_per_mwh, "EUR/MWh", "0.5-1.5 times canonical", 2025, "2025 EUR", "endogenous no-DC resource marginal cost; ETS and damage excluded", "", "model-derived and frozen before optimization"),
        _provenance_row("line 1-2 capacity", "F12_max", cfg.line_cap_12, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _provenance_row("line 2-3 capacity", "F23_max", cfg.line_cap_23, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _provenance_row("line 1-3 capacity", "F13_max", cfg.line_cap_13, "MW", "calibrated to 5-15% no-DC congestion hours", 2025, "not monetary", "no-DC DC-power-flow calibration", "", "model-derived and frozen before data-centre optimization"),
        _provenance_row("physical emissions reduction", "epsilon_CO2", 100.0 * cfg.carbon_cap_reduction_frac, "% of DC-attributable no-services emissions", "0-10", 2025, "not monetary", "steady-state same-IT counterfactual", "", "policy stress target; sensitivity required"),
    ]


def _provenance_row(
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
        "Parameter": parameter,
        "Symbol": symbol,
        "CanonicalValue": canonical_value,
        "Unit": unit,
        "SourceSupportedRange": source_supported_range,
        "SourceYear": source_year,
        "CurrencyBasis": currency_basis,
        "ConversionTo2025EUR": conversion,
        "Source": source,
        "EvidenceStatus": evidence_status,
    }


def export_parameter_provenance(cfg: Config, path: str, run_id: str = "") -> None:
    _write_documentation_table(parameter_provenance(cfg), path, run_id)


# ── literature comparison matrix ────────────────────────────────────────────

def literature_matrix() -> List[Dict]:
    """Comparison matrix used to delimit the siPUE contribution."""
    return [
        _literature_row(
            "PUE", "ISO/IEC 30134-2:2026", "facility energy / IT energy",
            facility=True, reuse=False, carbon=False, grid=False, economics=False, delay=False,
            source="https://www.iso.org/standard/30134-2",
            distinction="Standard facility-efficiency benchmark; no external system credit.",
        ),
        _literature_row(
            "ERF", "ISO/IEC 30134-6:2021", "reused energy / data-centre energy",
            facility=True, reuse=True, carbon=False, grid=False, economics=False, delay=False,
            source="https://www.iso.org/standard/71717.html",
            distinction="Measures reused energy but not net congestion, peak, reserve or carbon-damage effects.",
        ),
        _literature_row(
            "CUE", "The Green Grid WP#32", "carbon emissions / IT energy",
            facility=True, reuse=False, carbon=True, grid=False, economics=False, delay=False,
            source="https://www.thegreengrid.org/en/resources/library-and-tools/241-WP%2332---Carbon-Usage-Effectiveness-%28CUE%29%3A-A-Green-Grid-Data-Center-Sustainability-Metric",
            distinction="Carbon-intensity indicator; does not aggregate broader local-system burden.",
        ),
        _literature_row(
            "Energy-system integration study", "Energy 2023 Danish case study",
            "system cost and emissions from flexible cooling and waste heat",
            facility=False, reuse=True, carbon=True, grid=True, economics=True, delay=False,
            source="https://doi.org/10.1016/j.energy.2023.128112",
            distinction="Quantifies system outcomes but does not define a PUE-compatible KPI.",
        ),
        _literature_row(
            "Carbon-aware workload scheduling", "CarbonScaler 2023",
            "workload elasticity optimized against grid carbon",
            facility=False, reuse=False, carbon=True, grid=True, economics=False, delay=False,
            source="https://doi.org/10.1145/3626788",
            distinction="Operational carbon scheduling rather than a multi-service lifecycle metric.",
        ),
        _literature_row(
            "AI data-centre flexibility planning", "ACM e-Energy/Sustainability Week 2026",
            "deferrable and geographically shiftable AI workload in grid planning",
            facility=False, reuse=False, carbon=True, grid=True, economics=True, delay=False,
            source="https://doi.org/10.1145/3765611.3815593",
            distinction="Grid-planning formulation; no delay-aware PUE extension.",
        ),
        _literature_row(
            "siPUE", "This study", "delay-adjusted independent system benefit as energy-equivalent facility burden",
            facility=True, reuse=True, carbon=True, grid=True, economics=True, delay=True,
            source="",
            distinction="Proposed model-based indicator; novelty claim remains conditional on structured review.",
        ),
    ]


def _literature_row(
    metric: str,
    reference: str,
    scope: str,
    *,
    facility: bool,
    reuse: bool,
    carbon: bool,
    grid: bool,
    economics: bool,
    delay: bool,
    source: str,
    distinction: str,
) -> Dict:
    return {
        "MetricOrApproach": metric,
        "Reference": reference,
        "PrimaryScope": scope,
        "FacilityEnergy": facility,
        "EnergyReuse": reuse,
        "Carbon": carbon,
        "GridCongestionOrFlexibility": grid,
        "EconomicSystemEffects": economics,
        "ConstructionDelay": delay,
        "Source": source,
        "DistinctionFromSiPUE": distinction,
    }


def export_literature_matrix(path: str, run_id: str = "") -> None:
    _write_documentation_table(literature_matrix(), path, run_id)
