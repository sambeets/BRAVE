from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib.ticker import ScalarFormatter

from brave.config import Config
from brave.scenario import Scenario
from brave.metrics import nanmean, spearman_rank_corr
from brave.optimization import solve_ops_given_design


def _ensure_plot_dir(cfg: Config) -> str:
    out_dir = cfg.plot_output_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _pretty_type_label(dc_type: str) -> str:
    mapping = {
        "hyperscale": "Hyperscale",
        "ai": "AI",
        "streaming": "Streaming",
        "colocation": "Colocation",
    }
    return mapping.get(dc_type, dc_type.replace("_", " ").title())


def _apply_sci(ax, axis: str = "y") -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "z":
        ax.zaxis.set_major_formatter(formatter)


PLOT_FONT_SCALE = 1.6


def _fs(size: float) -> int:
    return int(round(size * PLOT_FONT_SCALE))


def _apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": _fs(13),
        "axes.labelsize": _fs(16),
        "axes.titlesize": _fs(16),
        "axes.labelweight": "bold",
        "xtick.labelsize": _fs(13),
        "ytick.labelsize": _fs(13),
        "legend.fontsize": _fs(12),
        "lines.linewidth": 2.6,
        "lines.markersize": 9,
    })


def _bold_ticklabels(ax, size: Optional[int] = None) -> None:
    target_size = _fs(size) if size is not None else _fs(13)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(target_size)


def plot_lmp_profile(ops: Dict, out_path: str, hours: int = 168) -> None:
    settle = ops.get("settle_lmp_series") or []
    util = ops.get("grid_util_series") or []
    if not settle:
        raise ValueError("No settle_lmp_series available for LMP plot.")
    h = min(hours, len(settle))
    if h <= 0:
        raise ValueError("No data available for LMP plot.")
    settle = settle[:h]
    util = util[:h] if util else [float("nan")] * h

    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    x = np.arange(h)
    axes[0].plot(x, settle, lw=2.2)
    axes[0].set_ylabel("Electricity price (EUR/MWh)", fontweight="bold", fontsize=_fs(9))
    axes[0].grid(alpha=0.2)
    _bold_ticklabels(axes[0], size=9)

    if util and util[0] == util[0]:
        util_pct = [100.0 * u for u in util]
        axes[1].fill_between(x, 0.0, util_pct, color="#888888", alpha=0.25)
        axes[1].plot(x, util_pct, color="#666666", lw=2.0)
        axes[1].set_ylabel("Grid util. (%)", fontweight="bold", fontsize=_fs(9))
    else:
        axes[1].set_ylabel("Grid util. (%)", fontweight="bold", fontsize=_fs(9))
    axes[1].set_xlabel("Hour (one-week slice)", fontweight="bold", fontsize=_fs(9))
    axes[1].grid(alpha=0.2)
    _bold_ticklabels(axes[1], size=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_vs_cost_spearman(rows, metric_key, cost_key="UnitCost_rob", title=""):
    _apply_plot_style()
    x = [r[metric_key] for r in rows]
    y = [r[cost_key] for r in rows]

    rho = spearman_rank_corr(x, y)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=60, alpha=0.9)
    ax.set_xlabel(metric_key, fontweight="bold")
    ax.set_ylabel(cost_key, fontweight="bold")
    ax.set_title(f"{title}\nSpearman rho = {rho:.3f}", fontweight="bold")
    _bold_ticklabels(ax, size=12)
    fig.tight_layout()
    return fig


def export_plots(rows: List[Dict], cfg: Config) -> None:
    if not cfg.export_plots:
        return

    _apply_plot_style()
    out_dir = _ensure_plot_dir(cfg)

    def _short_label(name: str) -> str:
        parts = name.replace("_", " ").split("-")
        if len(parts) >= 2:
            return f"{parts[0][:3].title()}-{parts[-1][:3].title()}"
        return name[:10]

    prof_pct = []
    for r in rows:
        base = r["Profit_base"]
        prof_pct.append((r["Profit_rob"] - base) / max(1e-6, abs(base)) * 100.0)
    labels = [_short_label(r["Scenario"]) for r in rows]
    values = prof_pct

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(values))
    ax.hlines(y, 0.0, values, color="#4cc9f0", lw=3.0, alpha=0.9)
    ax.scatter(values, y, s=90, color="#4cc9f0", edgecolors="black", linewidths=0.6, zorder=3)
    ax.axvline(0.0, color="#cccccc", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=_fs(10))
    ax.set_xlabel("Profit uplift (%)", fontweight="bold", fontsize=_fs(12))
    ax.grid(axis="x", alpha=0.25)
    _bold_ticklabels(ax, size=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"profit_uplift_lollipop.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if cfg.objective_mode == "profit_max":
        d_obj_trend = [r["UnitProfit_rob"] - r["UnitProfit_base"] for r in rows]
        y_label = "Δ Unit profit (rob−base, EUR/MWh IT)"
    else:
        d_obj_trend = [r["UnitCost_base"] - r["UnitCost_rob"] for r in rows]
        y_label = "Δ Unit cost saving (base−rob, EUR/MWh IT)"
    x_vals = [r["siPUE_rob"] - r["siPUE_base"] for r in rows]
    y_vals = d_obj_trend
    ax.scatter(x_vals, y_vals, s=90, color="#4cc9f0", edgecolors="black", linewidths=0.6)
    if len(rows) >= 2:
        fit = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(min(x_vals), max(x_vals), 100)
        ys = fit[0] * xs + fit[1]
        ax.plot(xs, ys, color="#222222", lw=2.0)
    rho = spearman_rank_corr(x_vals, y_vals)
    ax.text(0.02, 0.96, f"Spearman $\\rho$={rho:.3f}", transform=ax.transAxes, fontsize=_fs(12), va="top", fontweight="bold")
    ax.set_xlabel("Δ siPUE (rob−base)", fontweight="bold", fontsize=_fs(12))
    ax.set_ylabel(y_label, fontweight="bold", fontsize=_fs(12))
    ax.grid(alpha=0.2)
    _bold_ticklabels(ax, size=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sipue_vs_profit_trend.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_vals = [r["siPUE_rob"] - r["siPUE_base"] for r in rows]
    y_vals = [r["UnitProfit_rob"] - r["UnitProfit_base"] for r in rows]
    ax.scatter(x_vals, y_vals, s=90, color="#4cc9f0", edgecolors="black", linewidths=0.6)
    if len(rows) >= 2:
        fit = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(min(x_vals), max(x_vals), 100)
        ys = fit[0] * xs + fit[1]
        ax.plot(xs, ys, color="#222222", lw=2.0)
    rho = spearman_rank_corr(x_vals, y_vals)
    ax.text(0.02, 0.96, f"Spearman $\\rho$={rho:.3f}", transform=ax.transAxes, fontsize=_fs(12), va="top", fontweight="bold")
    ax.set_xlabel("Δ siPUE (rob−base)", fontweight="bold", fontsize=_fs(12))
    ax.set_ylabel("Δ Unit profit (rob−base, EUR/MWh IT)", fontweight="bold", fontsize=_fs(12))
    ax.grid(alpha=0.2)
    _bold_ticklabels(ax, size=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sipue_vs_unitprofit.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    type_colors = {"ai": "#4cc9f0", "hyperscale": "#90be6d", "streaming": "#f4a261", "colocation": "#577590"}
    types = sorted(set(r.get("Type", "unknown") for r in rows))
    for t in types:
        R = [r for r in rows if r.get("Type") == t]
        ax.scatter(
            [r.get("CreditMWhHeat_rob", 0.0) for r in R],
            [r.get("CreditMWhFlex_rob", 0.0) for r in R],
            s=90,
            label=_pretty_type_label(t),
            color=type_colors.get(t, "#999999"),
            edgecolors="black",
            linewidths=0.6,
        )
    ax.set_xlabel("Heat credit (MWh eq.)", fontweight="bold", fontsize=_fs(12))
    ax.set_ylabel("Flex credit (MWh eq.)", fontweight="bold", fontsize=_fs(12))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10, loc="upper left")
    _bold_ticklabels(ax, size=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"heat_vs_flex_credit.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pairs = [(r["Scenario"], r["Regret_rob"] / max(1e-9, r["E_IT_base"] / max(1e-9, float(cfg.T)))) for r in rows]
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels = [_short_label(p[0]) for p in pairs]
    values = [p[1] for p in pairs]
    y = np.arange(len(values))
    ax.barh(y, values, color="#4cc9f0", edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_fs(9), fontweight="bold")
    ax.set_xlabel("Regret per avg IT MW (EUR, scaled)", fontweight="bold", fontsize=_fs(10))
    ax.ticklabel_format(style="plain", axis="x")
    _bold_ticklabels(ax, size=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"regret_per_itmw_bar.{cfg.plot_format}"))
    plt.close(fig)

    if any(r.get("HeatPotential_MWhth", 0.0) > 0 for r in rows):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x_vals = [r.get("HeatPotential_MWhth", 0.0) for r in rows]
        y_vals = [r.get("Heat_rob_MWhth", 0.0) for r in rows]
        ax.scatter(x_vals, y_vals, s=90, color="#4cc9f0", edgecolors="black", linewidths=0.6)
        ax.set_xlabel("Potential heat (MWhth)", fontweight="bold", fontsize=_fs(10))
        ax.set_ylabel("Delivered heat (MWhth)", fontweight="bold", fontsize=_fs(10))
        if x_vals and y_vals:
            xmin, xmax = min(x_vals), max(x_vals)
            ymin, ymax = min(y_vals), max(y_vals)
            xpad = max(1e-9, (xmax - xmin) * 0.07)
            ypad = max(1e-9, (ymax - ymin) * 0.07)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
        _apply_sci(ax, "x")
        _apply_sci(ax, "y")
        ax.grid(alpha=0.2)
        _bold_ticklabels(ax, size=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"heat_delivered_vs_potential.{cfg.plot_format}"))
        plt.close(fig)

    # ── Plot: PUE vs siPUE Direction (split panels for scale) ───────────
    from matplotlib.patches import Patch
    fig, (ax_pue, ax_sipue) = plt.subplots(1, 2, figsize=(8, 5), sharey=True,
                                           gridspec_kw={"width_ratios": [1, 2.5]})
    scenario_labels = [_short_label(r["Scenario"]) for r in rows]
    y_pos = np.arange(len(rows))
    delta_pue_dir = [r["PUE_rob"] - r["PUE_base"] for r in rows]
    delta_sipue_dir = [r["siPUE_rob"] - r["siPUE_base"] for r in rows]
    delta_profit_dir = [r["Profit_rob"] - r["Profit_base"] for r in rows]

    colors_pue_dir = ["#e63946" if dp > 0 else "#2a9d8f" for dp in delta_pue_dir]
    colors_sipue_dir = ["#2a9d8f" if ds < 0 else "#e63946" for ds in delta_sipue_dir]

    ax_pue.barh(y_pos, delta_pue_dir, height=0.6, color=colors_pue_dir, edgecolor="none", alpha=0.85)
    ax_pue.axvline(0, color="#333333", lw=1)
    ax_pue.set_xlabel("\u0394PUE (rob\u2212base)", fontweight="bold", fontsize=_fs(10))
    ax_pue.set_yticks(y_pos)
    ax_pue.set_yticklabels(scenario_labels, fontsize=_fs(7), fontweight="bold")
    ax_pue.grid(axis="x", alpha=0.2)
    _bold_ticklabels(ax_pue, size=8)

    ax_sipue.barh(y_pos, delta_sipue_dir, height=0.6, color=colors_sipue_dir, edgecolor="none", alpha=0.85)
    ax_sipue.axvline(0, color="#333333", lw=1)
    ax_sipue.set_xlabel("\u0394siPUE (rob\u2212base)", fontweight="bold", fontsize=_fs(10))
    ax_sipue.grid(axis="x", alpha=0.2)
    _bold_ticklabels(ax_sipue, size=8)
    for i in range(len(rows)):
        symbol = "\u25B2" if delta_profit_dir[i] > 0 else "\u25BC"
        clr = "#2a9d8f" if delta_profit_dir[i] > 0 else "#e63946"
        ax_sipue.text(min(delta_sipue_dir) * 1.15, y_pos[i], symbol, va="center",
                      ha="center", fontsize=_fs(10), color=clr, fontweight="bold")

    legend_dir = [Patch(facecolor="#2a9d8f", label="Improves (aligns with profit)"),
                  Patch(facecolor="#e63946", label="Worsens (misaligns with profit)")]
    ax_sipue.legend(handles=legend_dir, loc="lower right", fontsize=_fs(8))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pue_vs_sipue_direction.{cfg.plot_format}"),
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot: Profit Waterfall (stacked, DC-level) ───────────────────────
    fig, ax = plt.subplots(figsize=(12, 5.5))
    scenario_short_dec = [_short_label(r["Scenario"]) for r in rows]
    x_pos_dec = np.arange(len(rows))

    it_margin_dec, heat_rev_dec, flex_rev_dec = [], [], []
    demand_neg_dec, capex_neg_dec, residual_dec, net_profit_dec = [], [], [], []
    for r in rows:
        srv = 1.0 - float(r.get("ConstructionDelayFrac", 0))
        it_m = r.get("ITRevEUR_rob", 0) - srv * r.get("ElSettlementEUR_rob", 0)
        h = r.get("HeatRevEUR_rob", 0)
        f = r.get("FlexRevEUR_rob", 0)
        dem = r.get("DemandChargeEUR_rob", 0)
        cap = r.get("EffCapexEUR_rob", 0) + r.get("LandCostEUR_rob", 0)
        prof = r.get("Profit_rob", 0)
        res = prof - (it_m + h + f - dem - cap)
        it_margin_dec.append(it_m / 1e6)
        heat_rev_dec.append(h / 1e6)
        flex_rev_dec.append(f / 1e6)
        demand_neg_dec.append(-dem / 1e6)
        capex_neg_dec.append(-cap / 1e6)
        residual_dec.append(res / 1e6)
        net_profit_dec.append(prof / 1e6)

    bottom_pos = [0.0] * len(rows)
    ax.bar(x_pos_dec, it_margin_dec, width=0.5, bottom=bottom_pos, color="#264653",
           label="IT Margin (ITRev\u2212ElSettle)")
    bottom_pos = it_margin_dec[:]
    ax.bar(x_pos_dec, heat_rev_dec, width=0.5, bottom=bottom_pos, color="#2a9d8f",
           label="Heat Revenue")
    bottom_pos = [b + h for b, h in zip(bottom_pos, heat_rev_dec)]
    ax.bar(x_pos_dec, flex_rev_dec, width=0.5, bottom=bottom_pos, color="#e9c46a",
           label="Flex Revenue")
    bottom_neg = [0.0] * len(rows)
    ax.bar(x_pos_dec, demand_neg_dec, width=0.5, bottom=bottom_neg, color="#e76f51",
           label="Demand Charge")
    bottom_neg = demand_neg_dec[:]
    ax.bar(x_pos_dec, capex_neg_dec, width=0.5, bottom=bottom_neg, color="#577590",
           label="CAPEX+Land")
    bottom_neg = [b + c for b, c in zip(bottom_neg, capex_neg_dec)]
    ax.bar(x_pos_dec, residual_dec, width=0.5, bottom=bottom_neg, color="#999999",
           label="Other (base CAPEX, SLA, eff)")
    ax.scatter(x_pos_dec, net_profit_dec, marker="D", s=70, color="#111111", zorder=5,
               label="Net Profit")

    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x_pos_dec)
    ax.set_xticklabels(scenario_short_dec, fontsize=_fs(8), fontweight="bold")
    ax.set_ylabel("EUR (millions)", fontweight="bold", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7), ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    _bold_ticklabels(ax, size=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"profit_decomposition.{cfg.plot_format}"))
    plt.close(fig)

    # ── Plot: siPUE Credit Decomposition (fraction of E_IT) ──────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    scenario_short_cr = [_short_label(r["Scenario"]) for r in rows]
    x_pos_cr = np.arange(len(rows))

    heat_frac_cr = [-r.get("CreditMWhHeat_rob", 0) / max(1e-9, r["E_IT_rob"]) for r in rows]
    flex_frac_cr = [-r.get("CreditMWhFlex_rob", 0) / max(1e-9, r["E_IT_rob"]) for r in rows]
    co2_frac_cr = [r.get("CO2MWh_rob", 0) / max(1e-9, r["E_IT_rob"]) for r in rows]
    net_frac_cr = [h + f + c for h, f, c in zip(heat_frac_cr, flex_frac_cr, co2_frac_cr)]

    bottom_neg_cr = [0.0] * len(rows)
    ax.bar(x_pos_cr, heat_frac_cr, width=0.5, bottom=bottom_neg_cr, color="#2a9d8f",
           label="Heat credit (reduces siPUE)")
    bottom_neg_cr = heat_frac_cr[:]
    ax.bar(x_pos_cr, flex_frac_cr, width=0.5, bottom=bottom_neg_cr, color="#e9c46a",
           label="Flex credit (reduces siPUE)")
    ax.bar(x_pos_cr, co2_frac_cr, width=0.5, color="#e76f51",
           label="CO\u2082 penalty (increases siPUE)")
    ax.plot(x_pos_cr, net_frac_cr, "ko-", markersize=7, lw=2.0, label="Net (siPUE \u2212 PUE)",
            zorder=5)
    for i, nf in enumerate(net_frac_cr):
        ax.text(x_pos_cr[i], nf + 0.003, f"{nf:+.3f}", ha="center", va="bottom",
                fontsize=_fs(7), fontweight="bold")

    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x_pos_cr)
    ax.set_xticklabels(scenario_short_cr, fontsize=_fs(8), fontweight="bold")
    ax.set_ylabel("Contribution to siPUE (fraction of E_IT)", fontweight="bold", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.2)
    _bold_ticklabels(ax, size=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sipue_credit_breakdown.{cfg.plot_format}"))
    plt.close(fig)


def export_regret_landscapes(
    scenarios: List[Scenario],
    profiles: Dict[int, Dict[str, List[float]]],
    c_star: Dict[int, float],
    robust: Dict[str, float],
    cfg: Config,
) -> None:
    if not cfg.export_regret_landscape:
        return

    out_dir = _ensure_plot_dir(cfg)
    nx = max(3, int(cfg.landscape_grid_nx))
    ny = max(3, int(cfg.landscape_grid_ny))

    def linspace(a: float, b: float, n: int) -> List[float]:
        if n <= 1:
            return [a]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    base_k_hx = robust["K_HX"]
    base_k_bat_e = robust["K_BAT_E"]
    base_k_bat_p = robust["K_BAT_P"]
    base_k_flex = robust["K_FLEX"]
    base_eff = robust["eff_reduction"]

    grids = [
        ("hx_bat", "HX capacity (MW_th)", "Battery energy (MWh)",
         linspace(0.0, cfg.K_HX_max, nx), linspace(0.0, cfg.K_BAT_E_max, ny),
         lambda x, y: (x, base_k_flex, y, base_k_bat_p)),
        ("hx_flex", "HX capacity (MW_th)", "Flex capacity (MW)",
         linspace(0.0, cfg.K_HX_max, nx), linspace(0.0, cfg.K_FLEX_max, ny),
         lambda x, y: (x, y, base_k_bat_e, base_k_bat_p)),
        ("hx_flex_bat", "HX capacity (MW_th)", "Flex capacity (MW)",
         linspace(0.0, cfg.K_HX_max, nx), linspace(0.0, cfg.K_FLEX_max, ny),
         lambda x, y: (x, y, base_k_bat_e, base_k_bat_p)),
        ("flex_bat", "Flex capacity (MW)", "Battery energy (MWh)",
         linspace(0.0, cfg.K_FLEX_max, nx), linspace(0.0, cfg.K_BAT_E_max, ny),
         lambda x, y: (base_k_hx, x, y, base_k_bat_p)),
    ]

    tags = getattr(cfg, "regret_landscape_tags", None)
    if tags is not None:
        tags = set(tags)
    for tag, xlabel, ylabel, xs, ys, mapper in grids:
        if tags is not None and tag not in tags:
            continue
        Z = [[0.0 for _ in ys] for _ in xs]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                k_hx, k_flex, k_bat_e, k_bat_p = mapper(x, y)
                worst_regret = -1e18
                for sc in scenarios:
                    prof = profiles[sc.sid]
                    ops = solve_ops_given_design(sc, prof, cfg, k_hx, k_flex, k_bat_e, k_bat_p, base_eff)
                    regret = ops["total_cost"] - c_star[sc.sid]
                    if regret > worst_regret:
                        worst_regret = regret
                Z[i][j] = worst_regret

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        X, Y = [], []
        for i in range(len(xs)):
            X.append([xs[i] for _ in ys])
            Y.append([ys[j] for j in range(len(ys))])
        X = np.array(X)
        Y = np.array(Y)
        Z_arr = np.array(Z)
        ax.plot_surface(X, Y, Z_arr, cmap="viridis", edgecolor="none", alpha=0.9)
        if tag == "hx_bat":
            ax.scatter([base_k_hx], [base_k_bat_e], [max(c_star.values()) + robust["Z"] if "Z" in robust else max(max(row) for row in Z)],
                       color="white", edgecolors="black", s=60, marker="*")
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=_fs(9))
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=_fs(9))
        ax.set_zlabel("Worst-case regret (EUR, scaled)", fontweight="bold", fontsize=_fs(9), labelpad=12)
        _apply_sci(ax, "z")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            for lbl in axis.get_ticklabels():
                lbl.set_fontweight("bold")
                lbl.set_fontsize(9)
        fig.subplots_adjust(right=0.86)
        fig.savefig(os.path.join(out_dir, f"regret_surface_{tag}.{cfg.plot_format}"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        c = ax.contourf(X, Y, Z_arr, levels=12, cmap="viridis")
        cb = fig.colorbar(c, ax=ax)
        cb.set_label("Worst-case regret (EUR, scaled)", fontweight="bold", fontsize=_fs(9))
        cb.ax.tick_params(labelsize=_fs(9))
        for lbl in cb.ax.get_yticklabels():
            lbl.set_fontweight("bold")
        if tag == "hx_bat":
            ax.scatter([base_k_hx], [base_k_bat_e], color="white", edgecolors="black", s=50, marker="*")
            x0, y0 = base_k_hx, base_k_bat_e
        elif tag == "hx_flex":
            ax.scatter([base_k_hx], [base_k_flex], color="white", edgecolors="black", s=50, marker="*")
            x0, y0 = base_k_hx, base_k_flex
        elif tag == "hx_flex_bat":
            ax.scatter([base_k_hx], [base_k_flex], color="white", edgecolors="black", s=50, marker="*")
            x0, y0 = base_k_hx, base_k_flex
        elif tag == "flex_bat":
            ax.scatter([base_k_flex], [base_k_bat_e], color="white", edgecolors="black", s=50, marker="*")
            x0, y0 = base_k_flex, base_k_bat_e
        else:
            x0, y0 = None, None

        if x0 is not None and y0 is not None:
            x_min, x_max = float(np.min(X)), float(np.max(X))
            y_min, y_max = float(np.min(Y)), float(np.max(Y))
            if x0 > 0:
                span = max(0.35 * x0, 0.2 * (x_max - x_min))
                ax.set_xlim(max(x_min, x0 - span), min(x_max, x0 + span))
            if y0 > 0:
                span = max(0.35 * y0, 0.2 * (y_max - y_min))
                ax.set_ylim(max(y_min, y0 - span), min(y_max, y0 + span))
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=_fs(9))
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=_fs(9))
        _apply_sci(cb.ax, "y")
        _bold_ticklabels(ax, size=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"regret_surface_{tag}_contour.{cfg.plot_format}"))
        plt.close(fig)
