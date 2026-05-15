from __future__ import annotations

import os
import random
from dataclasses import replace
from typing import Dict, List

import matplotlib.pyplot as plt

from brave.config import Config, _validate_locked_calibration
from brave.scenario import Scenario, make_scenarios, apply_global_heat_assumptions
from brave.metrics import service_revenue_scale
from brave.network import load_network_timeseries
from brave.timeseries import (
    build_representative_days, make_hourly_profiles,
    make_hourly_profiles_from_indices, apply_services_mode,
)
from brave.optimization import (
    baseline_design_for_scenario, solve_ops_given_design,
    solve_perfect_foresight, solve_minmax_regret,
)
from brave.export import export_scenarios_csv, export_results_csv, summarize_interpretation
from brave.plotting import (
    _ensure_plot_dir, export_plots, export_regret_landscapes,
    plot_lmp_profile, plot_metric_vs_cost_spearman,
)


def run_single(cfg: Config) -> List[Dict]:
    random.seed(cfg.seed)
    if cfg.enforce_locked_calibration:
        _validate_locked_calibration(cfg)
    scenarios = make_scenarios(cfg)
    apply_global_heat_assumptions(scenarios, cfg)

    if cfg.T == 8760:
        # Annual run: always use representative-day compression. Full 8760h runs are disabled.
        if not cfg.use_representative_days_annual:
            raise ValueError(
                "T=8760 requires use_representative_days_annual=True. "
                "Full 8760h runs are disabled; use the representative-day path."
            )
        ts_full = load_network_timeseries(cfg, T=8760)
        rep_indices, rep_weights, rep_day_starts, rep_total_days = build_representative_days(ts_full, cfg)
        cfg_local = replace(cfg, T=len(rep_indices))
        base_profiles = {
            sc.sid: make_hourly_profiles_from_indices(sc, cfg_local, rep_indices, rep_weights, rep_day_starts)
            for sc in scenarios
        }
    else:
        # Short-horizon path (tests only, T < 8760).
        cfg_local = cfg
        base_profiles = {sc.sid: make_hourly_profiles(sc, cfg_local.T, cfg_local) for sc in scenarios}
    profiles = {sc.sid: apply_services_mode(sc, base_profiles[sc.sid], cfg_local) for sc in scenarios}

    if cfg_local.export_csv:
        export_scenarios_csv(scenarios, cfg_local.out_scenarios_csv)

    C_star = {}
    PF_design = {}
    for sc in scenarios:
        cstar, des = solve_perfect_foresight(sc, profiles[sc.sid], cfg_local)
        if cfg_local.validate_solutions and des.get("status") != "Optimal":
            raise RuntimeError(f"Perfect-foresight solve not Optimal for scenario {sc.sid} ({sc.name}): {des.get('status')}")
        C_star[sc.sid] = cstar
        PF_design[sc.sid] = des

    robust = solve_minmax_regret(scenarios, profiles, C_star, cfg_local)
    if cfg_local.validate_solutions and robust.get("status") != "Optimal":
        raise RuntimeError(f"Robust min-max regret solve not Optimal: {robust.get('status')}")

    print("\n=== Robust (min-max regret) shared design (single run) ===")
    print(f"K_HX (MW_th)  = {robust['K_HX']:.2f}")
    print(f"K_FLEX (MW)   = {robust['K_FLEX']:.2f}")
    print(f"K_BAT_E (MWh) = {robust['K_BAT_E']:.2f}")
    print(f"K_BAT_P (MW)  = {robust['K_BAT_P']:.2f}")
    print(f"Eff level idx = {robust['eff_level_index']}  (reduction={robust['eff_reduction']:.2f}, capex={robust['eff_capex']:.0f})")
    print(f"Worst regret Z= {robust['Z']:.2f} EUR (scaled)")
    print(f"siPUE norm='{cfg.sipue_credit_normalization}', clip={cfg.sipue_clip_nonnegative}")
    print("siPUE price source: see per-scenario output (sipue_price_source)")
    print(f"Overhead time-varying: {cfg.enable_time_varying_overhead}\n")

    rows = []
    sample_ops = None
    for sc in scenarios:
        prof = profiles[sc.sid]
        dh_cap = prof.get("dh_cap", [])
        weights = prof.get("w", [1.0] * len(dh_cap))
        heat_potential = sum(dh_cap[t] * weights[t] for t in range(len(dh_cap))) if dh_cap else 0.0

        b_k_hx, b_k_flex, b_k_be, b_k_bp, b_eff = baseline_design_for_scenario(sc, prof, cfg)
        baseline = solve_ops_given_design(sc, prof, cfg_local, b_k_hx, b_k_flex, b_k_be, b_k_bp, eff_reduction=b_eff)

        rob_ops = solve_ops_given_design(
            sc, prof, cfg_local,
            robust["K_HX"], robust["K_FLEX"], robust["K_BAT_E"], robust["K_BAT_P"],
            eff_reduction=robust["eff_reduction"]
        )
        if sample_ops is None:
            sample_ops = rob_ops

        cost_base = baseline["total_cost"]
        cost_rob = rob_ops["total_cost"]
        profit_base = baseline["total_profit"]
        profit_rob = rob_ops["total_profit"]

        rows.append({
            "Scenario": sc.name,
            "Type": sc.dc_type,
            "Loc": sc.location,
            "ConstructionDelayFrac": getattr(sc, "construction_delay_frac", 0.0),
            "ServiceRevScale": service_revenue_scale(cfg, sc),
            "LandLimitM2": getattr(sc, "land_limit_m2", None),

            "PUE_base": baseline["PUE"],
            "siPUE_base": baseline["siPUE"],
            "Cost_base": cost_base,
            "Profit_base": profit_base,
            "ProfitPct_base": 0.0,
            "LMP_settle_avg_dc_base": baseline.get("lmp_settle_avg_dc"),
            "GridUtil_avg_base": baseline.get("grid_util_avg"),
            "GridUtil_max_base": baseline.get("grid_util_max"),
            "HeatPipeUtil_avg_base": baseline.get("heat_pipe_util_avg"),
            "HeatPipeUtil_max_base": baseline.get("heat_pipe_util_max"),
            "ElCostEUR_base": baseline.get("el_cost_eur"),
            "ElSettlementEUR_base": baseline.get("el_settlement_eur"),
            "CO2CostEUR_base": baseline.get("co2_cost_eur"),
            "TransCostEUR_base": baseline.get("trans_cost_eur"),
            "DemandChargeEUR_base": baseline.get("demand_charge_eur"),
            "PeakImportMW_base": baseline.get("peak_import_mw"),
            "HeatRevEUR_base": baseline.get("heat_rev_eur"),
            "FlexRevEUR_base": baseline.get("flex_rev_eur"),
            "ITRevEUR_base": baseline.get("it_rev_eur"),
            "SLAPenEUR_base": baseline.get("sla_penalty_eur"),
            "LandM2_base": baseline.get("land_m2"),
            "LandCostEUR_base": baseline.get("land_cost_eur"),
            "EffCapexEUR_base": baseline.get("eff_capex_eur"),
            "CreditMWh_base": baseline.get("credit_mwh"),
            "CreditMWhHeat_base": baseline.get("credit_mwh_heat"),
            "CreditMWhFlex_base": baseline.get("credit_mwh_flex"),
            "CO2MWh_base": baseline.get("co2_mwh"),
            "siPUE_price_used_base": baseline.get("sipue_price_used"),
            "siPUE_price_source_base": baseline.get("sipue_price_source"),
            "sipue_price_floor_applied_base": baseline.get("sipue_price_floor_applied"),
            "E_FAC_base": baseline.get("E_FAC"),
            "E_IT_base": baseline.get("E_IT"),
            "E_GRID_base": baseline.get("E_GRID"),
            "E_LOAD_base": baseline.get("E_LOAD"),
            "CreditEUR_base": baseline.get("credit_eur"),
            "Heat_base_MWhth": baseline.get("heat_MWhth"),
            "Flex_base_hours": baseline.get("flex_cap_hours"),
            "BatCh_base": baseline.get("bat_charge_mwh"),
            "BatDis_base": baseline.get("bat_discharge_mwh"),
            "Eff_red_base": baseline.get("eff_reduction"),

            "PUE_rob": rob_ops["PUE"],
            "siPUE_rob": rob_ops["siPUE"],
            "Cost_rob": cost_rob,
            "Profit_rob": profit_rob,
            "ProfitPct_rob": (profit_rob - profit_base) / max(1e-6, abs(profit_base)) * 100.0,
            "CreditEUR_rob": rob_ops["credit_eur"],
            "CreditMWh_rob": rob_ops.get("credit_mwh"),
            "CreditMWhHeat_rob": rob_ops.get("credit_mwh_heat"),
            "CreditMWhFlex_rob": rob_ops.get("credit_mwh_flex"),
            "CO2MWh_rob": rob_ops.get("co2_mwh"),
            "Regret_rob": cost_rob - C_star[sc.sid],
            "siPUE_price_source_rob": rob_ops.get("sipue_price_source", ""),
            "siPUE_price_used_rob": rob_ops.get("sipue_price_used"),
            "sipue_price_floor_applied_rob": rob_ops.get("sipue_price_floor_applied"),
            "LMP_settle_avg_dc_rob": rob_ops.get("lmp_settle_avg_dc"),
            "GridUtil_avg_rob": rob_ops.get("grid_util_avg"),
            "GridUtil_max_rob": rob_ops.get("grid_util_max"),
            "HeatPipeUtil_avg_rob": rob_ops.get("heat_pipe_util_avg"),
            "HeatPipeUtil_max_rob": rob_ops.get("heat_pipe_util_max"),
            "ElCostEUR_rob": rob_ops.get("el_cost_eur"),
            "ElSettlementEUR_rob": rob_ops.get("el_settlement_eur"),
            "CO2CostEUR_rob": rob_ops.get("co2_cost_eur"),
            "TransCostEUR_rob": rob_ops.get("trans_cost_eur"),
            "DemandChargeEUR_rob": rob_ops.get("demand_charge_eur"),
            "PeakImportMW_rob": rob_ops.get("peak_import_mw"),
            "HeatRevEUR_rob": rob_ops.get("heat_rev_eur"),
            "FlexRevEUR_rob": rob_ops.get("flex_rev_eur"),
            "ITRevEUR_rob": rob_ops.get("it_rev_eur"),
            "SLAPenEUR_rob": rob_ops.get("sla_penalty_eur"),
            "LandM2_rob": rob_ops.get("land_m2"),
            "LandCostEUR_rob": rob_ops.get("land_cost_eur"),
            "EffCapexEUR_rob": rob_ops.get("eff_capex_eur"),

            "E_FAC_rob": rob_ops["E_FAC"],
            "E_IT_rob": rob_ops["E_IT"],
            "E_GRID_rob": rob_ops["E_GRID"],
            "E_LOAD_rob": rob_ops["E_LOAD"],
            "Heat_rob_MWhth": rob_ops["heat_MWhth"],
            "HeatPotential_MWhth": heat_potential,
            "Flex_rob_hours": rob_ops["flex_cap_hours"],
            "BatCh_rob": rob_ops["bat_charge_mwh"],
            "BatDis_rob": rob_ops["bat_discharge_mwh"],
            "Eff_red_rob": rob_ops["eff_reduction"],
            "UnitCost_base": cost_base / max(1e-9, baseline["E_IT"]),
            "UnitCost_rob": cost_rob / max(1e-9, rob_ops["E_IT"]),
            "UnitProfit_base": profit_base / max(1e-9, baseline["E_IT"]),
            "UnitProfit_rob": profit_rob / max(1e-9, rob_ops["E_IT"]),
            "NegProfit_base": -profit_base,
            "NegProfit_rob": -profit_rob,
        })

    print("=== Scenario summary (single run) ===")
    obj_label = "Cost" if cfg.objective_mode == "cost_min" else "Profit"
    header = (f"Scenario | Type | Loc | PUE base->rob | siPUE base->rob | {obj_label} base->rob | CreditEUR(rob) | Regret(rob)")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Scenario'][:22]:22} | {r['Type'][:10]:10} | {r['Loc']:2} | "
            f"{r['PUE_base']:.3f}->{r['PUE_rob']:.3f} | "
            f"{r['siPUE_base']:.3f}->{r['siPUE_rob']:.3f} | "
            f"{(r['Cost_base'] if cfg.objective_mode == 'cost_min' else r['Profit_base']):.0f}"
            f"->{(r['Cost_rob'] if cfg.objective_mode == 'cost_min' else r['Profit_rob']):.0f} | "
            f"{r['CreditEUR_rob']:.0f} | "
            f"{r['Regret_rob']:.0f}"
        )

    summarize_interpretation(rows, cfg_local)

    if cfg_local.export_csv:
        export_results_csv(rows, cfg_local.out_results_csv)
        print(f"\nCSV exported: {cfg_local.out_scenarios_csv}, {cfg_local.out_results_csv}")
    if cfg_local.export_plots:
        export_plots(rows, cfg_local)
        try:
            if sample_ops:
                out_dir = _ensure_plot_dir(cfg_local)
                plot_lmp_profile(
                    sample_ops,
                    os.path.join(out_dir, f"electricity_price_profile_week.{cfg_local.plot_format}"),
                    hours=min(168, cfg_local.T),
                )
        except Exception as exc:
            print(f"[warn] Skipping LMP price profile plot: {exc}")
        print(f"Plots exported to: {cfg_local.plot_output_dir}")
    if cfg_local.export_regret_landscape:
        export_regret_landscapes(scenarios, profiles, C_star, robust, cfg_local)
        print(f"Regret landscapes exported to: {cfg_local.plot_output_dir}")
    if cfg_local.export_diagnostic_plots:
        out_dir = _ensure_plot_dir(cfg_local)
        if cfg_local.objective_mode == "profit_max":
            fig = plot_metric_vs_cost_spearman(rows, "PUE_rob", cost_key="NegProfit_rob", title="Robust design: PUE vs -Profit")
            fig.savefig(os.path.join(out_dir, f"diag_pue_vs_negprofit.{cfg_local.plot_format}"))
            plt.close(fig)
            fig = plot_metric_vs_cost_spearman(rows, "siPUE_rob", cost_key="NegProfit_rob", title="Robust design: siPUE vs -Profit")
            fig.savefig(os.path.join(out_dir, f"diag_sipue_vs_negprofit.{cfg_local.plot_format}"))
            plt.close(fig)
        else:
            fig = plot_metric_vs_cost_spearman(rows, "PUE_rob", cost_key="UnitCost_rob", title="Robust design: PUE vs Unit Cost")
            fig.savefig(os.path.join(out_dir, f"diag_pue_vs_unitcost.{cfg_local.plot_format}"))
            plt.close(fig)
            fig = plot_metric_vs_cost_spearman(rows, "siPUE_rob", cost_key="UnitCost_rob", title="Robust design: siPUE vs Unit Cost")
            fig.savefig(os.path.join(out_dir, f"diag_sipue_vs_unitcost.{cfg_local.plot_format}"))
            plt.close(fig)



    return rows
