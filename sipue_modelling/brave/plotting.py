from __future__ import annotations

import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brave.config import Config
from brave.optimization import solve_ops_given_design
from brave.scenario import Scenario


PLOT_FONT_SCALE = 1.0


def _fs(size: float) -> int:
    return int(round(size * PLOT_FONT_SCALE))


def _pretty_type_label(dc_type: str) -> str:
    labels = {
        "hyperscale": "Hyperscale",
        "colocation": "Colocation",
        "ai": "AI",
        "streaming": "Streaming",
    }
    return labels.get(dc_type, dc_type.replace("_", " ").title())


def _ensure_plot_dir(cfg: Config) -> str:
    os.makedirs(cfg.plot_output_dir, exist_ok=True)
    return cfg.plot_output_dir


def _num(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _labels(rows: List[Dict]) -> List[str]:
    return [str(row["Scenario"]).replace("Hyperscale-", "H-").replace("Streaming-", "S-").replace("Colocation-", "C-").replace("AI-", "AI-") for row in rows]


def export_plots(rows: List[Dict], cfg: Config) -> None:
    """Export descriptive plots only; no constructed-target correlations."""
    if not cfg.export_plots or not rows:
        return
    out_dir = _ensure_plot_dir(cfg)
    labels = _labels(rows)
    x = np.arange(len(rows))
    case = "robust"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    pue_delta = [_num(row, f"PUE_{case}") - _num(row, "PUE_no_services") for row in rows]
    sipue_delta = [_num(row, f"siPUE_{case}") - _num(row, "siPUE_no_services") for row in rows]
    axes[0].bar(x, pue_delta, color="#4C78A8")
    axes[1].bar(x, sipue_delta, color="#59A14F")
    for ax, ylabel in zip(axes, ("PUE change", "delay-aware siPUE change")):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pue_sipue_system_direction.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    credit_loss = [_num(row, f"SiPUECreditLossMWh_{case}") for row in rows]
    ax.bar(x, credit_loss, color="#E15759")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("siPUE credit loss (MWh)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"delay_credit_loss.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    width = 0.36
    market = [_num(row, f"CarbonMarketCostEUR_{case}") / 1e6 for row in rows]
    damage = [_num(row, f"CarbonDamageEUR_{case}") / 1e6 for row in rows]
    ax.bar(x - width / 2, market, width, label="CO2 market cost", color="#F28E2B")
    ax.bar(x + width / 2, damage, width, label="carbon damage", color="#B07AA1")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Separate carbon ledgers (million EUR)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"carbon_ledgers.{cfg.plot_format}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    cooling = [_num(row, f"CoolingMWh_{case}") for row in rows]
    heat_it = [_num(row, f"RecoveredHeatITMWhth_{case}") for row in rows]
    heat_cooling = [_num(row, f"RecoveredHeatCoolingMWhth_{case}") for row in rows]
    ax.bar(x, cooling, label="cooling electricity", color="#4C78A8")
    ax.bar(x, heat_it, bottom=cooling, label="IT-source recovered heat", color="#59A14F")
    ax.bar(x, heat_cooling, bottom=np.array(cooling) + np.array(heat_it), label="cooling-source recovered heat", color="#9C755F")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Energy (MWh)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cooling_heat_sources.{cfg.plot_format}"))
    plt.close(fig)


def export_study_plots(
    rows: List[Dict],
    comparison: List[Dict],
    decomposition: List[Dict],
    cfg: Config,
) -> None:
    """Export the manuscript-candidate figure set for the four-archetype study."""
    if not cfg.export_plots or not rows:
        return
    out_dir = _ensure_plot_dir(cfg)
    archetypes = [
        name for name in ("AI/HPC", "Hyperscale cloud", "Colocation", "Enterprise/edge")
        if any(row.get("Archetype") == name for row in rows)
    ]
    colors = ["#2F6B8A", "#3A7D44", "#C17C2C", "#8E5C7A"]
    x = np.arange(len(archetypes))

    def values(archetype: str, key: str) -> List[float]:
        return [_num(row, key) for row in rows if row.get("Archetype") == archetype]

    def mean_range(archetype: str, key: str):
        data = values(archetype, key)
        mean = float(np.mean(data)) if data else 0.0
        return mean, mean - min(data, default=mean), max(data, default=mean) - mean

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, ylabel in (
        (axes[0], "PUEReductionPctVsNoServices_robust", "PUE reduction vs no-services (%)"),
        (axes[1], "SiPUEReductionPctVsNoServices_robust", "siPUE reduction vs no-services (%)"),
    ):
        stats = [mean_range(name, key) for name in archetypes]
        means = [item[0] for item in stats]
        errors = np.array([[item[1] for item in stats], [item[2] for item in stats]])
        ax.bar(x, means, color=colors, width=0.68)
        ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, archetypes, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"metric_change_by_archetype.{cfg.plot_format}"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, ylabel in (
        (axes[0], "LifecycleCostReductionPctVsNoServices_robust", "Lifecycle cost reduction (%)"),
        (axes[1], "DCAttributedEmissionsReductionPctVsNoServices_robust", "DC-attributable emissions reduction (%)"),
    ):
        stats = [mean_range(name, key) for name in archetypes]
        means = [item[0] for item in stats]
        errors = np.array([[item[1] for item in stats], [item[2] for item in stats]])
        ax.bar(x, means, color=colors, width=0.68)
        ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, archetypes, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cost_emissions_change_by_archetype.{cfg.plot_format}"))
    plt.close(fig)

    design_order = ["deterministic", "expected_value", "pure_minmax", "robust"]
    design_labels = ["Deterministic", "Expected value", "Absolute min-max", "Normalized carbon min-max"]
    fig, ax = plt.subplots(figsize=(9.3, 4.4))
    width = 0.18
    for index, (design, label) in enumerate(zip(design_order, design_labels)):
        means = []
        for archetype in archetypes:
            data = [
                _num(row, "RegretPerMWhIT")
                for row in comparison
                if row.get("Archetype") == archetype and row.get("Design") == design
            ]
            means.append(float(np.mean(data)) if data else 0.0)
        ax.bar(x + (index - 1.5) * width, means, width, label=label)
    ax.set_xticks(x, archetypes, rotation=20, ha="right")
    ax.set_ylabel("Lifecycle regret (EUR/discounted MWh IT)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"regret_comparison_by_archetype.{cfg.plot_format}"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    delay_loss = [float(np.mean(values(name, "DelayLossPctOfZeroDelayGain_robust"))) for name in archetypes]
    credit_loss = [float(np.mean(values(name, "SiPUECreditLossPct_robust"))) for name in archetypes]
    axes[0].bar(x, delay_loss, color=colors)
    axes[1].bar(x, credit_loss, color=colors)
    for ax, ylabel in zip(axes, ("Delay loss (% of zero-delay gain)", "siPUE credit lost to delay (%)")):
        ax.set_xticks(x, archetypes, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"construction_delay_effects.{cfg.plot_format}"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    market = [float(np.mean(values(name, "CarbonMarketCostEUR_robust"))) / 1e6 for name in archetypes]
    damage = [float(np.mean(values(name, "CarbonDamageEUR_robust"))) / 1e6 for name in archetypes]
    axes[0].bar(x - 0.18, market, 0.36, label="ETS payment", color="#C17C2C")
    axes[0].bar(x + 0.18, damage, 0.36, label="carbon damage", color="#8E5C7A")
    axes[0].set_ylabel("Annual value (million EUR)")
    axes[0].legend(frameon=False)
    congestion = [float(np.mean(values(name, "CongestionHours_robust"))) for name in archetypes]
    curtailment = [float(np.mean(values(name, "RenewableCurtailmentMWh_robust"))) for name in archetypes]
    axes[1].bar(x - 0.18, congestion, 0.36, label="congestion hours", color="#2F6B8A")
    axes[1].bar(x + 0.18, curtailment, 0.36, label="renewable curtailment (MWh)", color="#3A7D44")
    axes[1].set_ylabel("Annual system outcome")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, archetypes, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"carbon_and_network_outcomes.{cfg.plot_format}"))
    plt.close(fig)

    components = [
        "heating revenue", "flexibility revenue", "resource electricity",
        "cooling electricity", "CO2 market", "capex", "land", "construction delay",
    ]
    component_colors = ["#3A7D44", "#2F6B8A", "#6B6B6B", "#64A6BD", "#C17C2C", "#B44B4B", "#8E5C7A", "#D7A84B"]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    positive = np.zeros(len(archetypes))
    negative = np.zeros(len(archetypes))
    for component, color in zip(components, component_colors):
        data = []
        for archetype in archetypes:
            found = [
                _num(row, "ValueEUR") / 1e6
                for row in decomposition
                if row.get("Archetype") == archetype
                and row.get("Design") == "robust"
                and row.get("Ledger") == "market"
                and row.get("Component") == component
            ]
            data.append(float(np.mean(found)) if found else 0.0)
        data_array = np.array(data)
        bottoms = np.where(data_array >= 0.0, positive, negative)
        ax.bar(x, data_array, bottom=bottoms, label=component, color=color)
        positive += np.where(data_array >= 0.0, data_array, 0.0)
        negative += np.where(data_array < 0.0, data_array, 0.0)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, archetypes, rotation=20, ha="right")
    ax.set_ylabel("Average lifecycle value contribution (million EUR)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"market_value_decomposition.{cfg.plot_format}"))
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
    n = max(3, int(cfg.landscape_grid_nx))
    hx_values = np.linspace(0.0, cfg.K_HX_max, n)
    flex_values = np.linspace(0.0, cfg.K_FLEX_max, n)
    regret = np.zeros((n, n))
    for i, k_hx in enumerate(hx_values):
        for j, k_flex in enumerate(flex_values):
            regret[i, j] = max(
                solve_ops_given_design(
                    sc,
                    profiles[sc.sid],
                    cfg,
                    k_hx,
                    k_flex,
                    robust["K_BAT_E"],
                    robust["K_BAT_P"],
                    robust["eff_reduction"],
                    K_HP=robust.get("K_HP", 0.0),
                )["market_planning_cost_npv_eur"]
                - c_star[sc.sid]
                for sc in scenarios
            )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    contour = ax.contourf(hx_values, flex_values, regret.T, levels=12, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="worst-case lifecycle regret (EUR)")
    ax.scatter([robust["K_HX"]], [robust["K_FLEX"]], marker="*", color="white", edgecolor="black", s=90)
    ax.set_xlabel("heat-exchanger capacity (MWth)")
    ax.set_ylabel("flexibility capacity (MW)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"regret_surface_hx_flex_contour.{cfg.plot_format}"))
    plt.close(fig)
