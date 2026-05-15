import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
import importlib.util
import sys

PLOT_FONT_SCALE = 1.0


def _fs(size: float) -> int:
    return int(round(size * PLOT_FONT_SCALE))

def _load_main():
    root = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root, "main.py")
    spec = importlib.util.spec_from_file_location("sipue_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sipue_main"] = mod
    spec.loader.exec_module(mod)
    return mod, root


def _run_case(
    mod,
    cfg,
    case_name,
    co2_scale=1.0,
    heat_scale=1.0,
    flex_scale=1.0,
    line_cap_scale=1.0,
    heat_price_override=None,
    construction_delay=0.0,
):
    scenarios = mod.make_scenarios(cfg)
    mod.apply_global_heat_assumptions(scenarios, cfg)

    for sc in scenarios:
        sc.carbon_price *= co2_scale
        if heat_price_override is None:
            sc.heat_price_level *= heat_scale
        else:
            sc.heat_price_level = float(heat_price_override)
        sc.flex_price_level *= flex_scale
        sc.heat_demand_level *= heat_scale
        sc.flex_frac *= flex_scale

    cfg_case = replace(cfg)
    cfg_case.co2_timeseries_scale = co2_scale
    cfg_case.line_cap_12 *= line_cap_scale
    cfg_case.line_cap_23 *= line_cap_scale
    cfg_case.line_cap_13 *= line_cap_scale
    if line_cap_scale != 1.0:
        cfg_case.transmission_cost_eur_per_mwh = 8.0
        cfg_case.th_eff_N1 = 0.52
        cfg_case.th_eff_N2 = 0.47
        cfg_case.th_eff_N3 = 0.42
        cfg_case.th_vom_N1 = 8.0
        cfg_case.th_vom_N2 = 14.0
        cfg_case.th_vom_N3 = 22.0
    cfg_case.export_csv = False
    cfg_case.export_plots = False
    cfg_case.export_regret_landscape = False
    cfg_case.export_diagnostic_plots = False
    cfg_case.construction_delay_frac = construction_delay

    mod.random.seed(cfg_case.seed)
    # Annual run: always use representative-day compression. Full 8760h runs are disabled.
    if cfg_case.T != 8760:
        raise ValueError(
            f"Sensitivity script requires T=8760 (got T={cfg_case.T}). "
            "Full non-annual runs are disabled."
        )
    ts_full = mod.load_network_timeseries(cfg_case, T=8760)
    rep_indices, rep_weights, rep_day_starts, rep_total_days = mod.build_representative_days(ts_full, cfg_case)
    cfg_local = replace(cfg_case, T=len(rep_indices))
    base_profiles = {
        sc.sid: mod.make_hourly_profiles_from_indices(sc, cfg_local, rep_indices, rep_weights, rep_day_starts)
        for sc in scenarios
    }
    profiles = {sc.sid: mod.apply_services_mode(sc, base_profiles[sc.sid], cfg_local) for sc in scenarios}

    C_star = {}
    for sc in scenarios:
        cstar, des = mod.solve_perfect_foresight(sc, profiles[sc.sid], cfg_local)
        if cfg_local.validate_solutions and des.get("status") != "Optimal":
            raise RuntimeError(f"PF not Optimal for {sc.sid} {sc.name}: {des.get('status')}")
        C_star[sc.sid] = cstar

    robust = mod.solve_minmax_regret(scenarios, profiles, C_star, cfg_local)
    if cfg_local.validate_solutions and robust.get("status") != "Optimal":
        raise RuntimeError(f"Robust not Optimal: {robust.get('status')}")

    rows = []
    for sc in scenarios:
        prof = profiles[sc.sid]
        b_k_hx, b_k_flex, b_k_be, b_k_bp, b_eff = mod.baseline_design_for_scenario(sc, prof, cfg_local)
        baseline = mod.solve_ops_given_design(sc, prof, cfg_local, b_k_hx, b_k_flex, b_k_be, b_k_bp, eff_reduction=b_eff)
        rob_ops = mod.solve_ops_given_design(
            sc, prof, cfg_local,
            robust["K_HX"], robust["K_FLEX"], robust["K_BAT_E"], robust["K_BAT_P"],
            eff_reduction=robust["eff_reduction"]
        )
        rows.append({
            "Profit_base": baseline["total_profit"],
            "Profit_rob": rob_ops["total_profit"],
            "PUE_base": baseline["PUE"],
            "PUE_rob": rob_ops["PUE"],
            "siPUE_base": baseline["siPUE"],
            "siPUE_rob": rob_ops["siPUE"],
            "GridUtil_avg_rob": rob_ops.get("grid_util_avg"),
            "CO2CostEUR_rob": rob_ops.get("co2_cost_eur"),
            "CreditMWh_rob": rob_ops.get("credit_mwh"),
        })

    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    n = len(rows)
    avg_profit_base = sum(f(r["Profit_base"]) for r in rows) / max(1, n)
    avg_profit_rob = sum(f(r["Profit_rob"]) for r in rows) / max(1, n)
    avg_profit_delta = avg_profit_rob - avg_profit_base
    avg_profit_pct = (avg_profit_delta / max(abs(avg_profit_base), 1e-6)) * 100.0

    avg_pue_base = sum(f(r["PUE_base"]) for r in rows) / max(1, n)
    avg_pue_rob = sum(f(r["PUE_rob"]) for r in rows) / max(1, n)
    avg_pue_delta = avg_pue_rob - avg_pue_base

    avg_sipue_base = sum(f(r["siPUE_base"]) for r in rows) / max(1, n)
    avg_sipue_rob = sum(f(r["siPUE_rob"]) for r in rows) / max(1, n)
    avg_sipue_delta = avg_sipue_rob - avg_sipue_base
    avg_sipue_pct = (avg_sipue_delta / max(abs(avg_sipue_base), 1e-6)) * 100.0

    sipue_pass = 0
    pue_fail = 0
    for r in rows:
        d_profit = f(r["Profit_rob"]) - f(r["Profit_base"])
        d_sipue = f(r["siPUE_rob"]) - f(r["siPUE_base"])
        d_pue = f(r["PUE_rob"]) - f(r["PUE_base"])
        if d_profit > 1e-9 and d_sipue < -1e-9:
            sipue_pass += 1
        if d_profit > 1e-9 and d_pue > 1e-9:
            pue_fail += 1

    avg_grid_util_rob = sum(f(r.get("GridUtil_avg_rob")) for r in rows) / max(1, n)
    avg_co2_cost_rob = sum(f(r.get("CO2CostEUR_rob")) for r in rows) / max(1, n)
    avg_credit_mwh_rob = sum(f(r.get("CreditMWh_rob")) for r in rows) / max(1, n)

    return {
        "case": case_name,
        "avg_profit_base": avg_profit_base,
        "avg_profit_rob": avg_profit_rob,
        "avg_profit_delta": avg_profit_delta,
        "avg_profit_pct": avg_profit_pct,
        "avg_pue_base": avg_pue_base,
        "avg_pue_rob": avg_pue_rob,
        "avg_pue_delta": avg_pue_delta,
        "avg_sipue_base": avg_sipue_base,
        "avg_sipue_rob": avg_sipue_rob,
        "avg_sipue_delta": avg_sipue_delta,
        "avg_sipue_pct": avg_sipue_pct,
        "sipue_pass_rate": sipue_pass / max(1, n),
        "pue_fail_rate": pue_fail / max(1, n),
        "avg_grid_util_rob": avg_grid_util_rob,
        "avg_co2_cost_rob": avg_co2_cost_rob,
        "avg_credit_mwh_rob": avg_credit_mwh_rob,
    }


def main():
    mod, root = _load_main()
    cfg = mod.Config(T=8760)
    cfg.use_representative_days_annual = True
    cfg.fixed_eff_reduction = 0.15
    cfg.sipue_credit_normalization = "settle_lmp_dc"
    cfg.sipue_credit_price_mode = "lmp"

    # (name, co2, heat, flex, line_cap, heat_price_override, delay)
    cases = [
        ("baseline",   1.0, 1.0, 1.0, 1.0, None, 0.0),
        ("co2_low",    0.5, 1.0, 1.0, 1.0, None, 0.0),
        ("co2_high",   2.0, 1.0, 1.0, 1.0, None, 0.0),
        ("heat_low",   1.0, 0.5, 1.0, 1.0, None, 0.0),
        ("heat_high",  1.0, 2.0, 1.0, 1.0, None, 0.0),
        ("flex_low",   1.0, 1.0, 0.5, 1.0, None, 0.0),
        ("flex_high",  1.0, 1.0, 2.0, 1.0, None, 0.0),
        ("cong_med",   1.0, 1.0, 1.0, 0.5, None, 0.0),
        ("cong_high",  1.0, 1.0, 1.0, 0.3, None, 0.0),
        ("delay_0.0",  1.0, 1.0, 1.0, 1.0, None, 0.0),
        ("delay_0.25", 1.0, 1.0, 1.0, 1.0, None, 0.25),
        ("delay_0.5",  1.0, 1.0, 1.0, 1.0, None, 0.5),
    ]

    rows = []
    for name, co2, heat, flex, line_cap, heat_override, delay in cases:
        rows.append(_run_case(mod, cfg, name, co2, heat, flex, line_cap, heat_override, delay))

    out_path = os.path.join(root, "output", "sensitivity_prices_congestion_repdays_annual_10scen_final.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sensitivity table: {out_path}")

    fig_dir = os.path.join(root, "output", "figures_365day_repdays_final")
    os.makedirs(fig_dir, exist_ok=True)

    rows_by_case = {r["case"]: r for r in rows}

    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    groups = [
        ("CO$_2$",     ["co2_low", "baseline", "co2_high"],    ["0.5x", "1.0x", "2.0x"]),
        ("Heat",       ["heat_low", "baseline", "heat_high"],  ["0.5x", "1.0x", "2.0x"]),
        ("Flex",       ["flex_low", "baseline", "flex_high"],  ["0.5x", "1.0x", "2.0x"]),
        ("Congestion", ["baseline", "cong_med", "cong_high"],  ["1.0x", "0.5x", "0.3x"]),
        ("Delay",      ["delay_0.0", "delay_0.25", "delay_0.5"], ["0.00", "0.25", "0.50"]),
    ]
    # Requested palette (no hatch/strike lines)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#999999"]
    hatches = ["", "", ""]

    plt.rcParams.update({
        "font.size": _fs(12),
        "axes.labelsize": _fs(13),
        "axes.titlesize": _fs(13),
        "xtick.labelsize": _fs(11),
        "ytick.labelsize": _fs(11),
        "legend.fontsize": _fs(11),
    })
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))
    x = range(len(groups))
    width = 0.20
    gap = 0.03
    offsets = [-(width + gap), 0.0, (width + gap)]

    for gi, (label, keys, bar_labels) in enumerate(groups):
        sip_vals = [-f(rows_by_case[k]["avg_sipue_delta"]) * 1000 for k in keys]
        for bi in range(3):
            ax.bar(
                gi + offsets[bi], sip_vals[bi], width=width,
                color=colors[gi], edgecolor="none", hatch=hatches[bi]
            )
            ax.text(gi + offsets[bi], sip_vals[bi], bar_labels[bi],
                         ha="center", va="bottom", fontsize=_fs(10), rotation=90, fontweight="bold")

    ax.set_ylabel(r"$\Delta$siPUE improvement ($\times 10^{-3}$)", fontsize=_fs(13), fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups], fontweight="bold", fontsize=_fs(12))
    ax.tick_params(axis="both", labelsize=_fs(11), width=1.0)

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "sensitivity_prices_congestion.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote sensitivity plot: {out_path}")

    # ── Dedicated CO₂ zoomed plot (3 panels, no siPUE level) ───────────
    co2_keys = ["co2_low", "baseline", "co2_high"]
    co2_labels = ["0.5×", "1.0×", "2.0×"]
    co2_sip = [-f(rows_by_case[k]["avg_sipue_delta"]) * 1000 for k in co2_keys]
    co2_prof = [f(rows_by_case[k]["avg_profit_pct"]) for k in co2_keys]
    co2_co2cost = [f(rows_by_case[k]["avg_co2_cost_rob"]) / 1e6 for k in co2_keys]  # M EUR

    co2_color = "#0072B2"
    fig2, (ax_c0, ax_c1, ax_c2) = plt.subplots(1, 3, figsize=(10.0, 3.6))

    # (a): siPUE improvement (zoomed)
    ax_c0.bar(range(3), co2_sip, color=co2_color, edgecolor="none", width=0.5)
    ax_c0.set_ylabel(r"$\Delta$siPUE improvement ($\times 10^{-3}$)", fontsize=_fs(9))
    ax_c0.set_xticks(range(3))
    ax_c0.set_xticklabels(co2_labels, fontsize=_fs(8))
    ax_c0.grid(axis="y", alpha=0.25)

    # (b): Profit uplift % (zoomed)
    ax_c1.bar(range(3), co2_prof, color=co2_color, edgecolor="none", width=0.5)
    ax_c1.set_ylabel("Profit uplift (%)", fontsize=_fs(9))
    ax_c1.set_xticks(range(3))
    ax_c1.set_xticklabels(co2_labels, fontsize=_fs(8))
    ax_c1.grid(axis="y", alpha=0.25)

    # (c): CO2 cost (M EUR)
    ax_c2.bar(range(3), co2_co2cost, color=co2_color, edgecolor="none", width=0.5)
    ax_c2.set_ylabel(r"CO$_2$ cost (M EUR)", fontsize=_fs(9))
    ax_c2.set_xticks(range(3))
    ax_c2.set_xticklabels(co2_labels, fontsize=_fs(8))
    ax_c2.grid(axis="y", alpha=0.25)

    for a in [ax_c0, ax_c1, ax_c2]:
        a.tick_params(axis="both", labelsize=_fs(8), width=1.0)
    fig2.tight_layout()
    out_co2 = os.path.join(fig_dir, "sensitivity_co2_zoomed.pdf")
    fig2.savefig(out_co2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote CO2 zoomed plot: {out_co2}")

    # ── Tornado diagram: siPUE sensitivity to each factor ────────────────
    # Show low/high swing relative to baseline for each factor
    tornado_factors = [
        ("CO$_2$",     "co2_low",   "co2_high",   "#0072B2"),
        ("Heat",       "heat_low",  "heat_high",  "#D55E00"),
        ("Flex",       "flex_low",  "flex_high",  "#009E73"),
        ("Congestion", "cong_med",  "cong_high",  "#CC79A7"),
        ("Delay",      "delay_0.25","delay_0.5",  "#999999"),
    ]

    baseline_sipue_delta = -f(rows_by_case["baseline"]["avg_sipue_delta"]) * 1000
    baseline_profit_pct = f(rows_by_case["baseline"]["avg_profit_pct"])

    fig3, (ax_sip, ax_prof) = plt.subplots(1, 2, figsize=(10, 4.5))

    y_pos = list(range(len(tornado_factors)))
    for i, (label, low_key, high_key, color) in enumerate(tornado_factors):
        sip_low = -f(rows_by_case[low_key]["avg_sipue_delta"]) * 1000
        sip_high = -f(rows_by_case[high_key]["avg_sipue_delta"]) * 1000
        prof_low = f(rows_by_case[low_key]["avg_profit_pct"])
        prof_high = f(rows_by_case[high_key]["avg_profit_pct"])

        # siPUE panel
        left_sip = min(sip_low, sip_high)
        right_sip = max(sip_low, sip_high)
        ax_sip.barh(i, right_sip - left_sip, left=left_sip, height=0.6, color=color, edgecolor="none", alpha=0.85)
        ax_sip.text(left_sip - 0.3, i, f"{left_sip:.1f}", va="center", ha="right", fontsize=_fs(7))
        ax_sip.text(right_sip + 0.3, i, f"{right_sip:.1f}", va="center", ha="left", fontsize=_fs(7))

        # Profit panel
        left_prof = min(prof_low, prof_high)
        right_prof = max(prof_low, prof_high)
        ax_prof.barh(i, right_prof - left_prof, left=left_prof, height=0.6, color=color, edgecolor="none", alpha=0.85)
        ax_prof.text(left_prof - 0.5, i, f"{left_prof:.1f}%", va="center", ha="right", fontsize=_fs(7))
        ax_prof.text(right_prof + 0.5, i, f"{right_prof:.1f}%", va="center", ha="left", fontsize=_fs(7))

    for ax in [ax_sip, ax_prof]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[0] for t in tornado_factors], fontsize=_fs(9))
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="both", labelsize=_fs(8), width=1.0)

    ax_sip.axvline(baseline_sipue_delta, color="#333333", lw=1.2, ls="--", label=f"Baseline ({baseline_sipue_delta:.1f})")
    ax_prof.axvline(baseline_profit_pct, color="#333333", lw=1.2, ls="--", label=f"Baseline ({baseline_profit_pct:.1f}%)")

    ax_sip.set_xlabel(r"$\Delta$siPUE improvement ($\times 10^{-3}$)", fontsize=_fs(9))
    ax_sip.legend(fontsize=_fs(8), loc="lower right")

    ax_prof.set_xlabel("Profit uplift (%)", fontsize=_fs(9))
    ax_prof.legend(fontsize=_fs(8), loc="lower right")
    fig3.tight_layout()
    out_tornado = os.path.join(fig_dir, "sensitivity_tornado.pdf")
    fig3.savefig(out_tornado, bbox_inches="tight")
    plt.close(fig3)
    print(f"Wrote tornado plot: {out_tornado}")


if __name__ == "__main__":
    main()
