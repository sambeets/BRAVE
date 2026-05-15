from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Dict, List

from brave.config import Config
from brave.metrics import nanmean, spearman_rank_corr

if TYPE_CHECKING:
    from brave.scenario import Scenario


def export_scenarios_csv(scenarios: List[Scenario], path: str):
    fields = [
        "sid", "name", "dc_type", "location",
        "base_it_mw", "flex_frac", "base_pue",
        "heat_recovery_frac", "hp_cop",
        "el_price_level", "el_price_vol",
        "heat_price_level", "heat_demand_level",
        "flex_price_level", "carbon_factor", "carbon_price",
        "land_limit_m2",
        "construction_delay_frac",
    ]
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in scenarios:
            w.writerow(sc.__dict__)


def export_results_csv(rows: List[Dict], path: str):
    fields = [
        "Scenario", "Type", "Loc",
        "PUE_base", "PUE_rob",
        "siPUE_base", "siPUE_rob",
        "Cost_base", "Cost_rob",
        "Profit_base", "Profit_rob",
        "ProfitPct_rob",
        "UnitCost_base", "UnitCost_rob",
        "UnitProfit_base", "UnitProfit_rob",
        "NegProfit_base", "NegProfit_rob",
        "CreditEUR_rob", "Regret_rob",
        "CreditMWh_base", "CreditMWh_rob",
        "siPUE_price_source_rob",
        "E_FAC_rob", "E_IT_rob",
        "Heat_rob_MWhth", "Flex_rob_hours",
        "BatCh_rob", "BatDis_rob",
        "Eff_red_rob",
    ]
    extra_keys = sorted({k for r in rows for k in r.keys()} - set(fields))
    fields = fields + extra_keys
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize_interpretation(rows: List[Dict], cfg: Config) -> None:
    n = len(rows)

    dpue = [r["PUE_rob"] - r["PUE_base"] for r in rows]
    dsipue = [r["siPUE_rob"] - r["siPUE_base"] for r in rows]
    if cfg.objective_mode == "profit_max":
        dobj = [r["Profit_rob"] - r["Profit_base"] for r in rows]
        better = [r for r in rows if r["Profit_rob"] > r["Profit_base"] + 1e-9]
        obj_label = "Profit"
        obj_better = better
    else:
        dobj = [r["Cost_rob"] - r["Cost_base"] for r in rows]
        better = [r for r in rows if r["Cost_rob"] < r["Cost_base"] - 1e-9]
        obj_label = "Cost"
        obj_better = better

    pue_worse = [r for r in rows if r["PUE_rob"] > r["PUE_base"] + 1e-9]
    sipue_better = [r for r in rows if r["siPUE_rob"] < r["siPUE_base"] - 1e-9]

    pue_worse_cost_better = [
        r for r in rows
        if ((r["Profit_rob"] > r["Profit_base"] + 1e-9) if cfg.objective_mode == "profit_max"
            else (r["Cost_rob"] < r["Cost_base"] - 1e-9))
        and (r["PUE_rob"] > r["PUE_base"] + 1e-9)
    ]

    sipue_align_cost_better = [
        r for r in rows
        if ((r["Profit_rob"] > r["Profit_base"] + 1e-9) if cfg.objective_mode == "profit_max"
            else (r["Cost_rob"] < r["Cost_base"] - 1e-9))
        and (r["siPUE_rob"] < r["siPUE_base"] - 1e-9)
    ]

    sipue_fail_cost_better = [
        r for r in rows
        if ((r["Profit_rob"] > r["Profit_base"] + 1e-9) if cfg.objective_mode == "profit_max"
            else (r["Cost_rob"] < r["Cost_base"] - 1e-9))
        and (r["siPUE_rob"] > r["siPUE_base"] + 1e-9)
    ]

    pue_list = [r["PUE_rob"] for r in rows]
    sipue_list = [r["siPUE_rob"] for r in rows]
    if cfg.objective_mode == "profit_max":
        unit_obj_list = [r["UnitProfit_rob"] for r in rows]
    else:
        unit_obj_list = [r["UnitCost_rob"] for r in rows]

    # Delta-based Spearman: correlate change in metric with change in profit
    # This tests the paper's hypothesis: does improving siPUE predict improving profit?
    delta_pue_list = [r["PUE_rob"] - r["PUE_base"] for r in rows]
    delta_sipue_list = [r["siPUE_rob"] - r["siPUE_base"] for r in rows]
    if cfg.objective_mode == "profit_max":
        delta_obj_list = [r["UnitProfit_rob"] - r["UnitProfit_base"] for r in rows]
    else:
        delta_obj_list = [r["UnitCost_base"] - r["UnitCost_rob"] for r in rows]

    rho_pue_unitobj = spearman_rank_corr(delta_pue_list, delta_obj_list)
    rho_sipue_unitobj = spearman_rank_corr(delta_sipue_list, delta_obj_list)

    print("\n=== Data-driven interpretation (single run) ===")
    print(f"Scenarios: {n}")
    print(f"Avg Delta PUE (rob-base):   {sum(dpue)/n:+.3f}")
    print(f"Avg Delta siPUE (rob-base): {sum(dsipue)/n:+.3f}")
    print(f"Avg Delta {obj_label} (rob-base):  {sum(dobj)/n:+.0f} EUR (scaled)")

    print("\nCounts (single run):")
    print(f" - PUE worsens:      {len(pue_worse)}/{n}")
    print(f" - siPUE improves:   {len(sipue_better)}/{n}")
    print(f" - {obj_label} improves:    {len(obj_better)}/{n}")
    print(f" - PUE failure (Delta {obj_label} improves & Delta PUE>0): {len(pue_worse_cost_better)}/{n}")
    if pue_worse_cost_better:
        print("   examples:", ", ".join([x["Scenario"] for x in pue_worse_cost_better[:5]]) +
              ("..." if len(pue_worse_cost_better) > 5 else ""))

    print(f"\nsiPUE validity (conditional on Delta {obj_label} improves):")
    denom = max(1, len(obj_better))
    print(f" - siPUE alignment rate: {len(sipue_align_cost_better)}/{denom} = {100*len(sipue_align_cost_better)/denom:.3f}%")
    print(f" - siPUE failure rate:   {len(sipue_fail_cost_better)}/{denom} = {100*len(sipue_fail_cost_better)/denom:.3f}%")

    print(f" - Spearman rho(Delta_PUE, Delta_Unit{obj_label}):   {rho_pue_unitobj}")
    print(f" - Spearman rho(Delta_siPUE, Delta_Unit{obj_label}):   {rho_sipue_unitobj}")
