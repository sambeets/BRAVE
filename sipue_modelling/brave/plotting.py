from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.path import Path

from brave.config import Config
from brave.network import load_network_timeseries
from brave.timeseries import build_representative_days


PLOT_FONT_SCALE = 1.7


def _fs(size: float) -> int:
    return int(round(size * PLOT_FONT_SCALE))


plt.rcParams.update({
    "font.size": _fs(12),
    "font.weight": "medium",
    "axes.titlesize": _fs(15),
    "axes.titleweight": "bold",
    "axes.labelsize": _fs(12),
    "axes.labelweight": "bold",
    "axes.linewidth": 1.3,
    "xtick.labelsize": _fs(11),
    "ytick.labelsize": _fs(11),
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.fontsize": _fs(11),
    "legend.title_fontsize": _fs(12),
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "figure.titlesize": _fs(18),
    "figure.titleweight": "bold",
    "patch.linewidth": 1.2,
})

CASE_COLORS = {
    "no_services": "#6B7280",
    "fair_services_disabled": "#4B5563",
    "baseline": "#2F6B8A",
    "robust": "#3A7D44",
}

ARCHETYPE_COLORS = {
    "AI/HPC": "#2F6B8A",
    "Hyperscale cloud": "#3A7D44",
    "Colocation": "#C17C2C",
    "Enterprise/edge": "#8E5C7A",
}

LEDGER_COLORS = {
    "heat": "#3A7D44",
    "flex": "#2F6B8A",
    "electricity": "#6B7280",
    "carbon": "#C17C2C",
    "capex": "#B44B4B",
    "land": "#8E5C7A",
    "delay": "#D7A84B",
    "policy": "#8E5C7A",
    "peak": "#946C4D",
    "transmission": "#5B7A99",
}

DESIGN_LABELS = {
    "no_services": "Identity no-services",
    "fair_services_disabled": "Services disabled",
    "baseline": "Baseline",
    "robust": "Robust",
}

ALLOWED_SENSITIVITY_FACTORS = {
    "canonical",
    "heat price",
    "flexibility price",
    "cooling overhead",
    "land availability",
    "construction delay",
}


def _wrap(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def _plot_formats(cfg: Config) -> Tuple[str, ...]:
    formats = [cfg.plot_format, *getattr(cfg, "plot_extra_formats", ())]
    seen = []
    for fmt in formats:
        fmt = str(fmt).strip().lower()
        if fmt and fmt not in seen:
            seen.append(fmt)
    return tuple(seen or ["pdf"])


def _save_figure(fig, out_dir: str, name: str, cfg: Config) -> None:
    for fmt in _plot_formats(cfg):
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)


def _text_box(ax, xy, text: str, color: str = "#F8FAFC", edge: str = "#CBD5E1", size: float = 9.0, zorder: float = 3):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=_fs(size),
        weight="medium",
        zorder=zorder,
        bbox={
            "boxstyle": "round,pad=0.4,rounding_size=0.09",
            "facecolor": color,
            "edgecolor": edge,
            "linewidth": 1.3,
        },
    )


def _read_csv_rows(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _norm_key(key: str) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _value(row: Dict, key: str, default=None):
    if key in row:
        return row[key]
    wanted = _norm_key(key)
    for actual, value in row.items():
        if _norm_key(actual) == wanted:
            return value
    return default


def _mean_numeric(rows: Sequence[Dict], key: str) -> float:
    values = [_num(row, key) for row in rows if _value(row, key) not in (None, "")]
    return float(np.mean(values)) if values else 0.0


def _median_numeric(rows: Sequence[Dict], key: str) -> float:
    values = [_num(row, key) for row in rows if _value(row, key) not in (None, "")]
    return float(np.median(values)) if values else 0.0


def _archetype_order(rows: Sequence[Dict]) -> List[str]:
    preferred = ["AI/HPC", "Hyperscale cloud", "Colocation", "Enterprise/edge"]
    return [name for name in preferred if any(_value(row, "Archetype") == name for row in rows)]


def _all_archetypes(rows: Sequence[Dict]) -> List[str]:
    preferred = _archetype_order(rows)
    extras = sorted({str(_value(row, "Archetype")) for row in rows if _value(row, "Archetype")} - set(preferred))
    return preferred + extras


def _case_change(rows: Sequence[Dict], metric: str, case: str) -> float:
    return _mean_numeric(rows, f"{metric}_{case}") - _mean_numeric(rows, f"{metric}_fair_services_disabled")


def _case_pct_change(rows: Sequence[Dict], metric: str, case: str) -> float:
    ref = _mean_numeric(rows, f"{metric}_fair_services_disabled")
    if abs(ref) <= 1e-9:
        return 0.0
    return 100.0 * (_mean_numeric(rows, f"{metric}_{case}") - ref) / abs(ref)


def _format_value(value: float, unit: str = "") -> str:
    if abs(value) >= 1e9:
        text = f"{value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        text = f"{value / 1e6:.2f}M"
    elif abs(value) >= 1000:
        text = f"{value / 1000:.1f}k"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.3g}"
    return f"{text} {unit}".strip()


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
        value = float(_value(row, key, default))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _labels(rows: List[Dict]) -> List[str]:
    return [str(row["scenario"]).replace("Hyperscale-", "H-").replace("Streaming-", "S-").replace("Colocation-", "C-").replace("AI-", "AI-") for row in rows]


def _placeholder_figure(out_dir: str, name: str, cfg: Config, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=_fs(15), weight="bold")
    ax.text(0.5, 0.43, _wrap(message, 90), ha="center", va="center", fontsize=_fs(10), color="#475569")
    _save_figure(fig, out_dir, name, cfg)


def _fig_05_optimization_formulation_map(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    boxes = {
        "benchmark": (0.18, 0.72, "Stage 1 benchmark\nC*_s\nscenario-perfect cost", "#FEF3C7"),
        "objective": (0.50, 0.72, "Stage 2 objective\nminimize max normalized regret", "#DCFCE7"),
        "decision": (0.82, 0.72, "Shared design variables\nHX, HP, FLEX, battery\n(efficiency retrofit fixed\nat 0.15 in canonical rows)", "#DBEAFE"),
        "constraints": (0.28, 0.38, "Constraints\ncarbon cap, land, heat,\nflex, battery, DC-OPF", "#F3E8FF"),
        "ops": (0.62, 0.38, "Operational evaluation\nhourly weighted energy,\ncost, emissions", "#E0F2FE"),
        "metrics": (0.50, 0.12, "Reported KPIs\nPUE, siPUE, CUE,\nsystem benefit, regret", "#FFE4E6"),
    }
    arrows = [
        ("benchmark", "objective"),
        ("decision", "objective"),
        ("objective", "constraints"),
        ("constraints", "ops"),
        ("decision", "ops"),
        ("ops", "metrics"),
    ]
    for start, end in arrows:
        x1, y1 = boxes[start][0], boxes[start][1]
        x2, y2 = boxes[end][0], boxes[end][1]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            zorder=2,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.8,
                "color": "#334155",
                "shrinkA": 46,
                "shrinkB": 46,
                "connectionstyle": "arc3,rad=0.0",
            },
        )
    for key, (xpos, ypos, label, face) in boxes.items():
        _text_box(ax, (xpos, ypos), label, face, "#94A3B8", 10.0, zorder=5)
    ax.text(0.5, 0.96, "Stage 2 minimizes worst-case normalized cost regret; siPUE is computed on the resulting design.", ha="center", fontsize=_fs(10.5), color="#334155", weight="medium")
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.02, right=0.98)
    _save_figure(fig, out_dir, "final_01_optimization_formulation_map", cfg)


def _fig_07_representative_day_compression(out_dir: str, cfg: Config) -> None:
    try:
        ts = load_network_timeseries(cfg, T=8760)
        indices, weights, starts, total_days = build_representative_days(ts, cfg)
    except Exception as exc:
        _placeholder_figure(out_dir, "final_03_representative_day_weights", cfg, "Representative-day compression", f"Representative-day diagnostic unavailable: {exc}")
        _placeholder_figure(out_dir, "final_04_representative_day_load_trace", cfg, "Representative-day compression", f"Representative-day diagnostic unavailable: {exc}")
        return
    day_ids = [indices[start] // 24 + 1 for start in starts]
    day_weights = [sum(weights[start:start + 24]) / 24.0 for start in starts]
    hours = list(range(min(168, len(ts["load_N1_mw"]))))
    load = [ts["load_N1_mw"][i] + ts["load_N2_mw"][i] + ts["load_N3_mw"][i] for i in hours]
    ren = [ts["ren_avail_N1_mw"][i] + ts["ren_avail_N2_mw"][i] + ts["ren_avail_N3_mw"][i] for i in hours]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(day_ids, day_weights, width=8.0, color=CASE_COLORS["baseline"], alpha=0.85)
    ax.set_ylabel("day weight")
    ax.set_xlabel("calendar day")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.12)
    _save_figure(fig, out_dir, "final_03_representative_day_weights", cfg)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(hours, load, label="network load", color=CASE_COLORS["baseline"], lw=2.2)
    ax.plot(hours, ren, label="renewable availability", color=CASE_COLORS["robust"], lw=2.2)
    ax.set_ylabel("MW")
    ax.set_xlabel("first-week hour")
    ax.legend(frameon=False)
    _style_axis(ax)
    _save_figure(fig, out_dir, "final_04_representative_day_load_trace", cfg)


def _fig_08b_load_and_renewables(out_dir: str, cfg: Config) -> None:
    """Network topology is drawn directly in paper.tex as a TikZ figure
    (Fig.~nodal-power), so no image-based topology figure is generated here."""
    try:
        ts = load_network_timeseries(cfg, T=8760)
        hours = list(range(min(240, len(ts["load_N1_mw"]))))
        load = [ts["load_N1_mw"][i] + ts["load_N2_mw"][i] + ts["load_N3_mw"][i] for i in hours]
        ren = [ts["ren_avail_N1_mw"][i] + ts["ren_avail_N2_mw"][i] + ts["ren_avail_N3_mw"][i] for i in hours]
    except Exception:
        load, ren, hours = [], [], []
    fig, ax_load = plt.subplots(figsize=(9.0, 5.0))
    if hours:
        ax_load.plot(hours, load, color=CASE_COLORS["baseline"], label="load")
        ax_load.plot(hours, ren, color=CASE_COLORS["robust"], label="renewables")
    ax_load.set_ylabel("MW")
    ax_load.set_xlabel("hour")
    ax_load.legend(frameon=False)
    _style_axis(ax_load)
    _save_figure(fig, out_dir, "final_05_load_and_renewables", cfg)


def _fig_03_pue_sipue_tradeoff_compass(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    tails_x, tails_y, heads_x, heads_y = [], [], [], []
    for row in rows:
        x0 = _num(row, "pue_baseline") - _num(row, "pue_fair_services_disabled")
        y0 = _num(row, "sipue_baseline") - _num(row, "sipue_fair_services_disabled")
        x1 = _num(row, "pue_robust") - _num(row, "pue_fair_services_disabled")
        y1 = _num(row, "sipue_robust") - _num(row, "sipue_fair_services_disabled")
        color = ARCHETYPE_COLORS.get(str(_value(row, "Archetype")), "#64748B")
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            zorder=3,
            arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": color, "alpha": 0.75, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 13},
        )
        ax.scatter([x0], [y0], s=22, color=color, alpha=0.55, edgecolor="none", zorder=2)
        tails_x.append(x0)
        tails_y.append(y0)
        heads_x.append(x1)
        heads_y.append(y1)
    ax.axhline(0.0, color="#111827", lw=1.0)
    ax.axvline(0.0, color="#111827", lw=1.0)
    ax.set_xlabel("ΔPUE (vs. services-disabled)")
    ax.set_ylabel("ΔsiPUE (vs. services-disabled)")
    all_x = tails_x + heads_x
    all_y = tails_y + heads_y
    if all_x:
        pad_x = 0.1 * ((max(all_x) - min(all_x)) or 1.0)
        pad_y = 0.1 * ((max(all_y) - min(all_y)) or 1.0)
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Neutral grey avoids colliding with archetype colors used below.
    quadrant_color = "#475569"
    ax.text(xlim[1], ylim[0], "PUE higher\nsiPUE lower", ha="right", va="bottom", fontsize=_fs(9), color=quadrant_color)
    ax.text(xlim[1], ylim[1], "PUE higher\nsiPUE higher", ha="right", va="top", fontsize=_fs(9), color=quadrant_color)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94A3B8", label="Baseline (tail)", markersize=7, alpha=0.75),
        Line2D([0], [0], marker=">", color="#334155", label="Robust (arrowhead)", markersize=9),
        *[
            Line2D([0], [0], marker="s", color="w", markerfacecolor=color, label=name, markersize=7)
            for name, color in ARCHETYPE_COLORS.items()
            if any(_value(row, "Archetype") == name for row in rows)
        ],
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=_fs(8.2))
    _style_axis(ax)
    fig.subplots_adjust(left=0.19, right=0.76, bottom=0.11, top=0.97)
    _save_figure(fig, out_dir, "final_09_pue_sipue_tradeoff_compass", cfg)


def _kpi_rows_from_results(rows: Sequence[Dict]) -> List[Dict]:
    metrics = [
        ("pue", "dimensionless", "pue"),
        ("sipue", "dimensionless", "sipue"),
        ("steady_sipue", "dimensionless", "steady_sipue"),
        ("cue", "tCO2/MWh IT", "cue"),
        ("modeled_erf", "heat/facility electricity", "modeled_erf"),
        ("system_benefit_per_mwh_it", "EUR/MWh IT", "system_benefit_per_mwh_it"),
        ("emissions_tco2", "tCO2", "emissions_tco2"),
        ("carbon_damage_eur", "EUR", "carbon_damage_eur"),
    ]
    output: List[Dict] = []
    for archetype in [*_all_archetypes(rows), "all"]:
        subset = rows if archetype == "all" else [row for row in rows if _value(row, "Archetype") == archetype]
        for design in ("no_services", "fair_services_disabled", "baseline", "robust"):
            for metric, unit, stem in metrics:
                values = []
                refs = []
                for row in subset:
                    if metric == "modeled_erf":
                        values.append(_num(row, f"heat_mwhth_{design}") / max(1e-9, _num(row, f"e_fac_mwh_{design}", 1.0)))
                        refs.append(_num(row, "heat_mwhth_no_services") / max(1e-9, _num(row, "e_fac_mwh_no_services", 1.0)))
                    else:
                        values.append(_num(row, f"{stem}_{design}"))
                        refs.append(_num(row, f"{stem}_no_services"))
                weights = [_num(row, f"e_it_mwh_{design}", 1.0) for row in subset]
                total_weight = sum(weights)
                avg = float(np.mean(values)) if values else 0.0
                ref = float(np.mean(refs)) if refs else 0.0
                avg_wt = (sum(v * w for v, w in zip(values, weights)) / total_weight) if total_weight > 0 else avg
                ref_wt = (sum(v * w for v, w in zip(refs, weights)) / total_weight) if total_weight > 0 else ref
                output.append({
                    "archetype": archetype,
                    "design_case": design,
                    "metric": metric,
                    "unit": unit,
                    "average_value": avg,
                    "average_no_services_value": ref,
                    "average_absolute_change_vs_no_services": avg - ref,
                    "average_pct_change_vs_no_services": (100.0 * (avg - ref) / abs(ref)) if abs(ref) > 1e-9 else "",
                    "average_value_it_weighted": avg_wt,
                    "average_no_services_value_it_weighted": ref_wt,
                })
    return output


KPI_PANEL_FILENAMES = {
    "pue": "final_06_kpi_pue",
    "sipue": "final_07_kpi_sipue",
    "system_benefit_per_mwh_it": "final_08_kpi_system_benefit_per_mwh_it",
}

# PUE and siPUE are dimensionless ratios, so omit the CSV unit string.
KPI_PANEL_YLABELS = {
    "pue": "PUE (–)",
    "sipue": "siPUE (–)",
}

# Share one y-axis for PUE/siPUE; system benefit uses a separate unit.
SHARED_YLIM_METRICS = ("pue", "sipue")


def _fig_04_kpi_benchmark_scoreboard(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.canonical_output_dir, "kpi_benchmark_summary.csv")
    kpi_rows = _read_csv_rows(path) or _kpi_rows_from_results(rows)
    metrics = ["pue", "sipue", "system_benefit_per_mwh_it"]
    archetypes = _all_archetypes(rows)
    designs = ("no_services", "fair_services_disabled", "baseline", "robust")
    x = np.arange(len(designs))

    def _series(subset: Sequence[Dict], archetype: str) -> List[float]:
        # Use IT weighting only for the pooled "all" series.
        value_key = "average_value_it_weighted" if archetype == "all" else "average_value"
        out = []
        for design in designs:
            match = [r for r in subset if _value(r, "Archetype") == archetype and _value(r, "DesignCase") == design]
            if not match:
                return []
            if value_key not in match[0]:
                value_key = "average_value"
            out.append(_num(match[0], value_key))
        return out

    # Collect metric series before plotting so PUE/siPUE can share y-limits.
    per_metric: Dict[str, Tuple[List[Dict], Dict[str, List[float]]]] = {}
    for metric in metrics:
        subset = [row for row in kpi_rows if _value(row, "Metric") == metric]
        if not subset:
            continue
        series_by_archetype = {}
        for archetype in [*archetypes, "all"]:
            values = _series(subset, archetype)
            if values:
                series_by_archetype[archetype] = values
        per_metric[metric] = (subset, series_by_archetype)

    shared_values = [
        v
        for metric in SHARED_YLIM_METRICS
        if metric in per_metric
        for series in per_metric[metric][1].values()
        for v in series
    ]
    shared_ylim = None
    if shared_values:
        lo, hi = min(shared_values), max(shared_values)
        pad = 0.08 * max(hi - lo, 1e-9)
        shared_ylim = (lo - pad, hi + pad)

    for metric in metrics:
        if metric not in per_metric:
            continue
        subset, series_by_archetype = per_metric[metric]
        unit = KPI_PANEL_YLABELS.get(metric, _value(subset[0], "Unit", ""))
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        drawn = False
        endpoint_labels = []
        for archetype in archetypes:
            values = series_by_archetype.get(archetype)
            if not values:
                continue
            color = ARCHETYPE_COLORS.get(archetype, "#64748B")
            ax.plot(x, values, color=color, lw=3.6, marker="o", markersize=10.0, markeredgecolor="white", markeredgewidth=1.3, alpha=0.95, zorder=3)
            endpoint_labels.append((float(values[-1]), archetype, color))
            drawn = True
        overall = series_by_archetype.get("all")
        if overall:
            ax.plot(x, overall, color="#111827", lw=2.8, linestyle="--", marker="s", markersize=8, zorder=2)
            endpoint_labels.append((float(overall[-1]), "All (IT-weighted)", "#111827"))
            drawn = True
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xticks(x, [DESIGN_LABELS.get(d, d) for d in designs], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.15, len(designs) - 1 + 0.65)
        ax.set_ylabel(unit)
        if metric in SHARED_YLIM_METRICS and shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        else:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        ymin, ymax = ax.get_ylim()
        yrange = max(ymax - ymin, 1e-9)
        min_sep = 0.13 * yrange
        adjusted = []
        for y_raw, archetype, color in sorted(endpoint_labels, key=lambda item: item[0]):
            y_text = y_raw if not adjusted else max(y_raw, adjusted[-1][0] + min_sep)
            adjusted.append((y_text, y_raw, archetype, color))
        if adjusted:
            overflow = adjusted[-1][0] - (ymax - 0.02 * yrange)
            if overflow > 0:
                adjusted = [(y_text - overflow, y_raw, archetype, color) for y_text, y_raw, archetype, color in adjusted]
            ymin_text = ymin + 0.02 * yrange
            adjusted = [(max(y_text, ymin_text), y_raw, archetype, color) for y_text, y_raw, archetype, color in adjusted]
            for y_text, y_raw, archetype, color in adjusted:
                ax.plot([x[-1], x[-1] + 0.06], [y_raw, y_text], color=color, lw=0.9, alpha=0.65, zorder=2)
                ax.text(x[-1] + 0.08, y_text, archetype, color=color, fontsize=_fs(9.0), weight="bold", va="center", ha="left")
        _style_axis(ax)
        fig.subplots_adjust(right=0.72, bottom=0.28)
        _save_figure(fig, out_dir, KPI_PANEL_FILENAMES[metric], cfg)


def _fig_09_design_portfolio_by_archetype(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    archetypes = _all_archetypes(rows)
    metrics = [
        ("HX", "K_HX_per_MW_IT"),
        ("HP", "K_HP_per_MW_IT"),
        ("Flex", "K_FLEX_per_MW_IT"),
        ("Battery E", "K_BAT_E_per_MW_IT"),
        ("Battery P", "K_BAT_P_per_MW_IT"),
    ]
    cases = ("baseline", "robust")
    raw = {
        archetype: {
            case: [_mean_numeric([row for row in rows if _value(row, "Archetype") == archetype], f"{stem}_{case}") for _, stem in metrics]
            for case in cases
        }
        for archetype in archetypes
    }
    if not archetypes:
        return
    n_axes = len(metrics)
    mins = [min(raw[a][c][j] for a in archetypes for c in cases) for j in range(n_axes)]
    maxs = [max(raw[a][c][j] for a in archetypes for c in cases) for j in range(n_axes)]
    ranges = [max(maxs[j] - mins[j], 1e-9) for j in range(n_axes)]

    def _norm(values: List[float]) -> List[float]:
        return [(v - mins[j]) / ranges[j] for j, v in enumerate(values)]

    x = np.arange(n_axes)
    case_style = {
        "baseline": {"marker": "o", "linestyle": "-", "markersize": 7.5},
        "robust": {"marker": "D", "linestyle": "--", "markersize": 7.0},
    }
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for xi in x:
        ax.axvline(xi, color="#CBD5E1", lw=1.0, zorder=1)
    for archetype in archetypes:
        color = ARCHETYPE_COLORS.get(archetype, "#64748B")
        for case in cases:
            ax.plot(
                x,
                _norm(raw[archetype][case]),
                color=color,
                alpha=0.9,
                lw=2.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
                **case_style[case],
            )
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_xlim(-0.3, n_axes - 1 + 0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized value (0-1)")
    handles = [Line2D([0], [0], color=ARCHETYPE_COLORS.get(a, "#64748B"), lw=2.6, label=a) for a in archetypes] + [
        Line2D([0], [0], color="#334155", lw=2.0, marker="o", linestyle="-", markersize=7, label="Baseline"),
        Line2D([0], [0], color="#334155", lw=2.0, marker="D", linestyle="--", markersize=6.5, label="Robust"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=_fs(8.5))
    _style_axis(ax, "y")
    ax.spines["bottom"].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.78, bottom=0.10, top=0.97)
    _save_figure(fig, out_dir, "final_10_design_portfolio_parallel", cfg)


def _ribbon(ax, p0: Tuple[float, float], w0: float, p1: Tuple[float, float], w1: float, color: str, alpha: float = 0.55, zorder: float = 1) -> None:
    """A width-varying flow ribbon between two node anchors (mirrored cubic Beziers with
    horizontal control tangents, the same construction d3-sankey uses for link paths)."""
    x0, y0 = p0
    x1, y1 = p1
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0 + w0 / 2), (xm, y0 + w0 / 2), (xm, y1 + w1 / 2), (x1, y1 + w1 / 2),
        (x1, y1 - w1 / 2), (xm, y1 - w1 / 2), (xm, y0 - w0 / 2), (x0, y0 - w0 / 2),
        (x0, y0 + w0 / 2),
    ]
    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY,
    ]
    ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))


def _fig_10_system_value_sankey(decomposition: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    """Decomposes the independent system-cost ledger that defines the siPUE
    credit (Eq. independent-cost-ledger / Table credit-ablation), not the
    private market cash flow. Avoided heat/flexibility cost, carbon damage,
    peak capacity, resource electricity, and transmission are the same six
    components reported in the credit-ablation table; construction delay is
    a separate scalar applied to their sum (Eq. sipue-credit) and is not part
    of this ledger, so it is excluded here to avoid double-counting."""
    components = [
        ("heat system value", "heat"),
        ("flexibility system value", "flex"),
        ("carbon damage", "carbon"),
        ("peak capacity", "peak"),
        ("resource electricity", "electricity"),
        ("transmission", "transmission"),
    ]
    # Pool archetypes with IT-energy weights to match reported means.
    values: Dict[str, float] = {}
    for component, _ in components:
        found = [
            (_num(row, "value_eur") / 1e6, _num(row, "e_it_mwh", 1.0))
            for row in decomposition
            if _value(row, "Design") == "robust" and _value(row, "Ledger") == "independent" and _value(row, "Component") == component
        ]
        total_weight = sum(w for _, w in found)
        values[component] = (sum(v * w for v, w in found) / total_weight) if total_weight > 0 else 0.0

    inflows = [(c, k, values[c]) for c, k in components if values[c] > 0]
    outflows = [(c, k, values[c]) for c, k in components if values[c] < 0]
    if not inflows:
        _placeholder_figure(out_dir, "final_11_system_value_sankey", cfg, "System benefit flow", "No positive value components found in the independent decomposition ledger.")
        return
    total_in = sum(value for _, _, value in inflows)
    total_out = -sum(value for _, _, value in outflows)
    net_value = total_in - total_out

    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.14, 1.04)

    max_w = 0.14
    gross_xy = (0.38, 0.68)
    net_xy = (0.94, 0.68)
    costs_xy = (0.60, 0.16)

    def _width(value: float) -> float:
        return max(0.014, max_w * abs(value) / total_in)

    n_in = len(inflows)
    in_ys = np.linspace(0.98, 0.38, n_in) if n_in > 1 else [0.68]
    for (component, key, value), y in zip(inflows, in_ys):
        color = LEDGER_COLORS.get(key, "#64748B")
        _ribbon(ax, (0.07, y), _width(value), gross_xy, _width(value), color, zorder=1)
        _text_box(ax, (0.07, y), f"{_wrap(component.title(), 14)}\n{value:+.1f} M EUR", "#F8FAFC", color, 11.0, zorder=3)

    _ribbon(ax, gross_xy, _width(net_value), net_xy, _width(net_value), CASE_COLORS["robust"], zorder=2)
    if outflows:
        _ribbon(ax, gross_xy, _width(total_out), costs_xy, _width(total_out), LEDGER_COLORS["peak"], zorder=2)

    _text_box(ax, gross_xy, f"Avoided\nindependent cost\n{total_in:+.1f} M EUR", "#DCFCE7", CASE_COLORS["robust"], 12.5, zorder=4)
    _text_box(ax, net_xy, f"Undelayed\nsystem benefit\n{net_value:+.1f} M EUR", "#DBEAFE", CASE_COLORS["baseline"], 12.5, zorder=4)

    if outflows:
        _text_box(ax, costs_xy, f"Additional\nindependent burden\n{-total_out:+.1f} M EUR", "#FEE2E2", LEDGER_COLORS["peak"], 11.5, zorder=4)
        n_out = len(outflows)
        out_ys = np.linspace(0.32, -0.10, n_out) if n_out > 1 else [0.05]
        for (component, key, value), y in zip(outflows, out_ys):
            color = LEDGER_COLORS.get(key, "#64748B")
            _ribbon(ax, costs_xy, _width(value), (0.88, y), _width(value), color, zorder=1)
            _text_box(ax, (0.88, y), f"{_wrap(component.title(), 16)}\n{value:+.1f} M EUR", "#FFF7ED", color, 10.5, zorder=3)

    _save_figure(fig, out_dir, "final_11_system_value_sankey", cfg)


def _fig_11_carbon_ledger_separation(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    archetypes = _all_archetypes(rows)
    metrics = [
        ("final_13_carbon_ledger_power_emissions", "Power emissions", "dc_power_emissions_tco2_robust", "tCO2", False),
        ("final_14_carbon_ledger_avoided_heat_emissions", "Avoided heat emissions", "avoided_heat_emissions_tco2_robust", "tCO2", False),
    ]
    y = np.arange(len(archetypes))
    # Shared x-limits make power-sector and avoided-heat profiles comparable.
    all_values: Dict[str, List[float]] = {}
    for slug, label, key, unit, is_pct in metrics:
        values = []
        for archetype in archetypes:
            subset = [row for row in rows if _value(row, "Archetype") == archetype]
            values.append(_mean_numeric(subset, key) / 1000.0)
        all_values[slug] = values
    shared_max = max((v for values in all_values.values() for v in values), default=0.0)
    shared_xlim = (0.0, shared_max * 1.08 if shared_max > 0 else 1.0)
    for slug, label, key, unit, is_pct in metrics:
        values = all_values[slug]
        color = LEDGER_COLORS["policy"] if is_pct else LEDGER_COLORS["carbon"]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.barh(y, values, color=color, alpha=0.88)
        if is_pct:
            ax.axvline(0.0, color="#222222", linewidth=1.2)
        ax.set_yticks(y, archetypes)
        ax.invert_yaxis()
        ax.set_xlim(*shared_xlim)
        ax.set_xlabel(f"{unit} (×10³)")
        _style_axis(ax, "x")
        fig.subplots_adjust(bottom=0.14, left=0.24)
        _save_figure(fig, out_dir, slug, cfg)


def _fig_12_delay_credit_bridge(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    """Plots the absolute credit lost to construction delay per archetype
    (MWh-equivalent/year), not the raw post-construction vs. delay-adjusted
    siPUE values: the percentage loss is uniform across archetypes (delay is
    a single scenario-level availability factor, not archetype-specific, per
    Eq. sipue-credit), so the raw siPUE values sit almost on top of each
    other and a percentage bar chart would show four equal bars. The
    absolute MWh lost varies by archetype because it scales with each
    archetype's own undelayed credit."""
    archetypes = _all_archetypes(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    y = np.arange(len(archetypes))
    loss_mwh = [
        _mean_numeric([row for row in rows if _value(row, "Archetype") == archetype], "sipue_credit_loss_mwh_robust")
        for archetype in archetypes
    ]
    ax.barh(y, loss_mwh, color=LEDGER_COLORS["delay"], alpha=0.9, height=0.6, zorder=2)
    for yi, value in zip(y, loss_mwh):
        ax.text(value, yi, f"  {value:,.0f}", va="center", ha="left", fontsize=_fs(11), weight="bold", color="#0F172A")
    ax.set_yticks(y, archetypes)
    ax.invert_yaxis()
    max_loss = max(loss_mwh, default=0.0)
    ax.set_xlim(0, max_loss * 1.2 if max_loss > 0 else 1.0)
    ax.set_xlabel("siPUE credit lost to construction delay (MWh-equivalent/year)")
    _style_axis(ax, "x")
    fig.subplots_adjust(left=0.24)
    _save_figure(fig, out_dir, "final_15_delay_credit_bridge", cfg)


def _fig_14_train_test_validation(out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.validation_output_dir, "train-test", f"train_test_1_{cfg.train_test_max_train_id}_vs_rest.csv")
    rows = _read_csv_rows(path)
    if not rows:
        _placeholder_figure(out_dir, "final_17_train_test_regret_distribution", cfg, "Train/test validation", "Train/test validation output is not present. Run run_train_test_scenario_validation.py before generating this diagnostic.")
        return
    archetypes = _all_archetypes(rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # Prefer EUR/MWh IT; use percent-of-lifecycle-cost only for older CSVs.
    regret_key = "regret_eur_per_mwh_it" if any(_value(row, "regret_eur_per_mwh_it") not in (None, "") for row in rows) else "regret_share_of_lifecycle"
    data = [[_num(row, regret_key) for row in rows if _value(row, "Archetype") == archetype and _value(row, regret_key) not in (None, "")] for archetype in archetypes]
    ax.boxplot(data, labels=archetypes, patch_artist=True, boxprops={"facecolor": "#DBEAFE", "color": CASE_COLORS["baseline"]}, medianprops={"color": "#111827", "linewidth": 2.0})
    ax.set_ylabel("Regret (% of lifecycle cost)" if regret_key == "regret_share_of_lifecycle" else "Regret (EUR/MWh IT)")
    ax.tick_params(axis="x", rotation=18)
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.2)
    _save_figure(fig, out_dir, "final_17_train_test_regret_distribution", cfg)


def _fig_15_out_of_sample_validation(out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.validation_output_dir, "out-of-sample", "out_of_sample_validation.csv")
    rows = _read_csv_rows(path)
    if not rows:
        _placeholder_figure(out_dir, "final_18_out_of_sample_regret_distribution", cfg, "Out-of-sample validation", "Out-of-sample validation output is not present. Run run_out_of_sample_validation.py before generating this diagnostic.")
        return
    archetypes = _all_archetypes(rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    regret_key = "regret_eur_per_mwh_it" if any(_value(row, "regret_eur_per_mwh_it") not in (None, "") for row in rows) else "regret_share_of_lifecycle"
    data = [[_num(row, regret_key) for row in rows if _value(row, "Archetype") == archetype and _value(row, regret_key) not in (None, "")] for archetype in archetypes]
    # Match the held-out panel style for direct comparison.
    ax.boxplot(data, labels=archetypes, patch_artist=True, boxprops={"facecolor": "#DCFCE7", "color": CASE_COLORS["robust"]}, medianprops={"color": "#111827", "linewidth": 2.0})
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylabel("Regret (% of lifecycle cost)" if regret_key == "regret_share_of_lifecycle" else "Regret (EUR/MWh IT)")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.2)
    _save_figure(fig, out_dir, "final_18_out_of_sample_regret_distribution", cfg)


def _fig_17_sensitivity_tornado(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    base = [row for row in rows if _value(row, "Factor") == "canonical" and _value(row, "Archetype") == "all"]
    if not base:
        return
    # Use IT weighting for pooled means; fall back for older CSVs.
    sipue_key = "average_sipue_it_weighted" if "average_sipue_it_weighted" in base[0] else "average_sipue"
    base_sipue = _num(base[0], sipue_key)
    entries = []
    for factor in sorted({str(_value(row, "Factor")) for row in rows if _value(row, "Factor") not in ("canonical", None, "")}):
        if factor not in ALLOWED_SENSITIVITY_FACTORS:
            continue
        subset = {
            str(_value(row, "Level")).strip().lower(): _num(row, sipue_key) - base_sipue
            for row in rows
            if _value(row, "Factor") == factor and _value(row, "Archetype") == "all"
        }
        # Label bars by input level, even when the output direction reverses.
        low_delta = subset.get("low")
        high_delta = subset.get("high")
        if low_delta is None or high_delta is None:
            # Construction delay uses "0 months"/"12 months" instead of low/high.
            level_values = sorted(
                (
                    (str(_value(row, "Level")), _num(row, sipue_key) - base_sipue)
                    for row in rows
                    if _value(row, "Factor") == factor and _value(row, "Archetype") == "all"
                ),
                key=lambda item: item[0],
            )
            if len(level_values) < 2:
                continue
            low_delta, high_delta = level_values[0][1], level_values[-1][1]
        entries.append((str(factor), low_delta, high_delta))
    if not entries:
        return
    entries.sort(key=lambda item: abs(item[2] - item[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    y = np.arange(len(entries))
    low_color, high_color = "#2F6B8A", "#C17C2C"
    label_color = "#0F172A"
    for yi, (factor, low_delta, high_delta) in zip(y, entries):
        ax.barh(yi, low_delta, color=low_color, alpha=0.85, height=0.55, zorder=2)
        ax.barh(yi, high_delta, color=high_color, alpha=0.85, height=0.55, zorder=2)
        low_text = "0.00" if abs(low_delta) < 0.005 else f"{low_delta:+.2f}"
        high_text = "0.00" if abs(high_delta) < 0.005 else f"{high_delta:+.2f}"
        ax.annotate(
            low_text, xy=(low_delta, yi), xytext=(-6 if low_delta < 0 else 6, 0), textcoords="offset points",
            va="center", ha="right" if low_delta < 0 else "left", fontsize=_fs(9.5), color=label_color, weight="medium", zorder=4,
        )
        ax.annotate(
            high_text, xy=(high_delta, yi), xytext=(-6 if high_delta < 0 else 6, 0), textcoords="offset points",
            va="center", ha="right" if high_delta < 0 else "left", fontsize=_fs(9.5), color=label_color, weight="medium", zorder=4,
        )
    ax.axvline(0.0, color="#111827", lw=1.2, label="canonical value")
    tips = [v for _, low_delta, high_delta in entries for v in (low_delta, high_delta)] + [0.0]
    lo, hi = min(tips), max(tips)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.24 * span, hi + 0.24 * span)
    ax.set_yticks(y, [label.title() for label, _, _ in entries])
    ax.invert_yaxis()
    ax.set_xlabel("robust siPUE change vs canonical sensitivity case")
    handles = [
        patches.Patch(facecolor=low_color, label="low input level"),
        patches.Patch(facecolor=high_color, label="high input level"),
    ]
    # Place legend above the axes to avoid the near-zero land bar.
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=_fs(9))
    _style_axis(ax, "x")
    fig.subplots_adjust(top=0.88)
    _save_figure(fig, out_dir, "final_19_sensitivity_tornado", cfg)


def _fig_18_sensitivity_archetype_response(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    factors = [
        factor for factor in ("heat price", "flexibility price", "cooling overhead", "land availability", "construction delay")
        if any(_value(row, "Factor") == factor for row in rows)
    ]
    archetypes = [name for name in _all_archetypes(rows) if name != "all"]
    if not factors or not archetypes:
        return
    matrix = []
    for archetype in archetypes:
        row_values = []
        base = [row for row in rows if _value(row, "Factor") == "canonical" and _value(row, "Archetype") == archetype]
        base_value = _num(base[0], "average_sipue") if base else 0.0
        for factor in factors:
            subset = [row for row in rows if _value(row, "Factor") == factor and _value(row, "Archetype") == archetype]
            vals = [_num(row, "average_sipue") - base_value for row in subset]
            row_values.append(max(vals, default=0.0) - min(vals, default=0.0))
        matrix.append(row_values)
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(data, cmap="cividis", aspect="auto")
    ax.set_yticks(np.arange(len(archetypes)), archetypes)
    ax.set_xticks(np.arange(len(factors)), [label.title() for label in factors], rotation=25, ha="right")
    vmax = float(data.max()) if data.size else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "#F8FAFC" if (vmax <= 0 or data[i, j] / vmax > 0.55) else "#0F172A"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=_fs(9.5), weight="bold", color=text_color)
    fig.colorbar(im, ax=ax, label="siPUE response range")
    _save_figure(fig, out_dir, "final_20_sensitivity_archetype_response", cfg)


def export_sensitivity_plots(cfg: Config) -> None:
    if not cfg.export_plots:
        return
    rows = _read_csv_rows(os.path.join(cfg.canonical_output_dir, "sensitivity_annual_summary.csv"))
    if not rows:
        return
    rows = [row for row in rows if _value(row, "Factor") in ALLOWED_SENSITIVITY_FACTORS]
    if not rows:
        return
    out_dir = _ensure_plot_dir(cfg)
    _fig_17_sensitivity_tornado(rows, out_dir, cfg)
    _fig_18_sensitivity_archetype_response(rows, out_dir, cfg)


def export_creative_figure_package(
    rows: Sequence[Dict],
    comparison: Sequence[Dict],
    decomposition: Sequence[Dict],
    cfg: Config,
) -> None:
    if not cfg.export_plots or not rows:
        return
    out_dir = _ensure_plot_dir(cfg)
    _fig_03_pue_sipue_tradeoff_compass(rows, out_dir, cfg)
    _fig_04_kpi_benchmark_scoreboard(rows, out_dir, cfg)
    _fig_05_optimization_formulation_map(out_dir, cfg)
    _fig_07_representative_day_compression(out_dir, cfg)
    _fig_08b_load_and_renewables(out_dir, cfg)
    _fig_09_design_portfolio_by_archetype(rows, out_dir, cfg)
    _fig_10_system_value_sankey(decomposition, out_dir, cfg)
    _fig_11_carbon_ledger_separation(rows, out_dir, cfg)
    _fig_12_delay_credit_bridge(rows, out_dir, cfg)
    _fig_14_train_test_validation(out_dir, cfg)
    _fig_15_out_of_sample_validation(out_dir, cfg)
    export_sensitivity_plots(cfg)


def export_creative_figure_package_from_csv(cfg: Config) -> None:
    rows = _read_csv_rows(os.path.join(cfg.canonical_output_dir, "results_4_archetypes_10_states.csv"))
    comparison = _read_csv_rows(os.path.join(cfg.canonical_output_dir, "design_comparison.csv"))
    decomposition = _read_csv_rows(os.path.join(cfg.canonical_output_dir, "decomposition.csv"))
    if not rows:
        raise FileNotFoundError(
            os.path.join(cfg.canonical_output_dir, "results_4_archetypes_10_states.csv")
        )
    export_creative_figure_package(rows, comparison, decomposition, cfg)


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
        if any(_value(row, "Archetype") == name for row in rows)
    ]
    x = np.arange(len(archetypes))

    def values(archetype: str, key: str) -> List[float]:
        return [_num(row, key) for row in rows if _value(row, "Archetype") == archetype]

    regret_means = []
    for archetype in archetypes:
        data = [
            _num(row, "regret_per_mwh_it")
            for row in comparison
            if _value(row, "Archetype") == archetype and _value(row, "Design") == "robust"
        ]
        regret_means.append(float(np.mean(data)) if data else 0.0)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(x, regret_means, s=160, color=CASE_COLORS["robust"], zorder=3, edgecolor="white", linewidth=1.2)
    ax.vlines(x, 0.0, regret_means, color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.axhline(0.0, color="#222222", linewidth=1.1)
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Robust design regret (EUR / discounted MWh IT)")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "final_16_regret_by_archetype", cfg)

    congestion = [float(np.mean(values(name, "congestion_hours_robust"))) for name in archetypes]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(x, congestion, s=160, color=CASE_COLORS["baseline"], zorder=3, edgecolor="white", linewidth=1.2)
    ax.vlines(x, 0.0, congestion, color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Congestion hours per year")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "final_12_network_congestion_by_archetype", cfg)

    export_creative_figure_package(rows, comparison, decomposition, cfg)
