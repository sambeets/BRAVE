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
from brave.optimization import solve_ops_given_design
from brave.scenario import Scenario
from brave.timeseries import build_representative_days


PLOT_FONT_SCALE = 1.55

plt.rcParams.update({
    "font.size": 12,
    "font.weight": "medium",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.3,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "figure.titlesize": 18,
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


def _fs(size: float) -> int:
    return int(round(size * PLOT_FONT_SCALE))


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


def _pct_true(rows: Sequence[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(str(_value(row, key, "")).lower() == "true" for row in rows) / len(rows)


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


def _add_takeaway(fig, text: str) -> None:
    fig.text(
        0.5,
        0.015,
        _wrap(text, 130),
        ha="center",
        va="bottom",
        fontsize=_fs(8.5),
        color="#334155",
    )


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


def _fig_01_sipue_metric_story(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")
    ax.set_title("siPUE extends PUE only when system services exist", fontsize=_fs(16), weight="bold", pad=12)
    y = 0.62
    blocks = [
        (0.12, "Facility electricity\nE_fac", "#E5E7EB", "#6B7280"),
        (0.32, "IT electricity\nE_IT", "#DBEAFE", CASE_COLORS["baseline"]),
        (0.52, "System benefit\nconverted to MWh", "#DCFCE7", CASE_COLORS["robust"]),
        (0.72, "Construction delay\navailability factor", "#FEF3C7", LEDGER_COLORS["delay"]),
        (0.90, "delay-aware\nsiPUE", "#F3E8FF", LEDGER_COLORS["policy"]),
    ]
    for idx, (xpos, label, face, edge) in enumerate(blocks):
        _text_box(ax, (xpos, y), label, face, edge, 10)
        if idx < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(blocks[idx + 1][0] - 0.075, y),
                xytext=(xpos + 0.075, y),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"},
            )
    ax.text(0.5, 0.30, "PUE = E_fac / E_IT", ha="center", fontsize=_fs(13), color="#111827")
    ax.text(
        0.5,
        0.19,
        "siPUE = (E_fac - delay-adjusted system credit) / E_IT",
        ha="center",
        fontsize=_fs(13),
        color="#111827",
    )
    ax.text(
        0.5,
        0.08,
        "No active heat, flexibility, or battery service => system credit = 0 => siPUE = PUE",
        ha="center",
        fontsize=_fs(10),
        color="#475569",
    )
    _save_figure(fig, out_dir, "fig_01_sipue_metric_story", cfg)


def _fig_01b_why_sipue_matters(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Why siPUE: the same integration assets read differently under PUE and siPUE", fontsize=_fs(14.5), weight="bold", pad=14)

    def _stack_bar(x_center: float, segments: List[Tuple[float, str]]) -> float:
        width = 0.16
        bottom = 0.12
        for height, color in segments:
            ax.add_patch(patches.Rectangle((x_center - width / 2, bottom), width, height, facecolor=color, edgecolor="#334155", linewidth=1.0, zorder=2))
            bottom += height
        return bottom

    top_a = _stack_bar(0.22, [(0.30, "#DBEAFE"), (0.28, "#94A3B8")])
    ax.text(0.22, top_a + 0.04, "PUE view", ha="center", va="bottom", fontsize=_fs(10.5), weight="bold")
    ax.text(0.22, 0.20, "IT electricity", ha="center", va="center", fontsize=_fs(8.0), weight="medium")
    ax.text(0.22, 0.51, "Integration overhead\n(HP, HX, storage, cooling)", ha="center", va="center", fontsize=_fs(7.6), weight="medium")
    ax.text(0.22, 0.05, "All non-IT electricity is\ncounted as pure overhead", ha="center", va="top", fontsize=_fs(8.0), color="#475569")
    _text_box(ax, (0.22, 0.90), "PUE = E_fac / E_IT\nhigher = penalized", "#FEE2E2", "#991B1B", 9.0, zorder=4)

    top_b = _stack_bar(0.72, [(0.30, "#DBEAFE"), (0.14, "#94A3B8"), (0.14, "#DCFCE7")])
    ax.text(0.72, top_b + 0.04, "siPUE view", ha="center", va="bottom", fontsize=_fs(10.5), weight="bold")
    ax.text(0.72, 0.20, "IT electricity", ha="center", va="center", fontsize=_fs(8.0), weight="medium")
    ax.text(0.72, 0.44, "Remaining\noverhead", ha="center", va="center", fontsize=_fs(7.6), weight="medium")
    ax.text(0.72, 0.58, "Credited system value\n(heat + flexibility)", ha="center", va="center", fontsize=_fs(7.6), weight="medium")
    ax.text(0.72, 0.05, "Independently valued heat and\nflexibility are credited back", ha="center", va="top", fontsize=_fs(8.0), color="#475569")
    _text_box(ax, (0.72, 0.90), "siPUE = (E_fac - credit) / E_IT\nlower = rewarded", "#DCFCE7", CASE_COLORS["robust"], 9.0, zorder=4)

    ax.annotate("", xy=(0.60, 0.5), xytext=(0.34, 0.5), arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#334155"}, zorder=3)
    ax.text(0.47, 0.545, "same physical assets", ha="center", va="bottom", fontsize=_fs(8.2), color="#334155", weight="medium")

    _add_takeaway(fig, "Same integration equipment, different KPI verdict: PUE structurally penalizes heat and flexibility investment, while siPUE credits the independently valued system benefit -- this is the KPI gap siPUE closes.")
    _save_figure(fig, out_dir, "fig_01b_why_sipue_matters", cfg)


def _fig_02_model_pipeline_subway(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.suptitle("From data to policy-facing KPI bundle", fontsize=_fs(17), weight="bold", y=0.985)
    xs = np.linspace(0.08, 0.92, 7)
    stages = [
        ("Inputs", "network, archetypes,\nprices, carbon, land"),
        ("Rep. days", "7 weighted days\ncover annual hours"),
        ("Scenarios", "4 archetypes x\n10 shared states"),
        ("Stage 1", "scenario-perfect\nbenchmark C*_s"),
        ("Stage 2", "shared robust\nmin-max regret design"),
        ("Operations", "heat, flex, battery,\ncarbon, DC network"),
        ("Outputs", "PUE, siPUE, CUE,\nERF-style heat, value"),
    ]
    colors = ["#E5E7EB", "#DBEAFE", "#E0F2FE", "#FEF3C7", "#DCFCE7", "#F3E8FF", "#FFE4E6"]
    for i, (xpos, (title, detail)) in enumerate(zip(xs, stages)):
        ax.plot([xpos, xpos], [0.24, 0.68], color="#CBD5E1", lw=5, solid_capstyle="round")
        ax.scatter([xpos], [0.46], s=620, color=colors[i], edgecolor="#334155", linewidth=1.6, zorder=3)
        ax.text(xpos, 0.46, str(i + 1), ha="center", va="center", fontsize=_fs(12), weight="bold")
        ax.text(xpos, 0.80, title, ha="center", va="center", fontsize=_fs(12.5), weight="bold")
        ax.text(xpos, 0.13, detail, ha="center", va="center", fontsize=_fs(9.5), color="#334155", weight="medium")
        if i < len(xs) - 1:
            ax.plot([xpos + 0.035, xs[i + 1] - 0.035], [0.46, 0.46], color="#334155", lw=2.4, zorder=1)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.02, right=0.98)
    _add_takeaway(fig, "The process separates scenario design, optimization, operation, and KPI accounting so policy and business claims can be traced.")
    _save_figure(fig, out_dir, "fig_02_model_pipeline_subway", cfg)


def _fig_02b_full_study_roadmap(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.suptitle("Full study roadmap: optimization, validation, and sensitivity together", fontsize=_fs(16), weight="bold", y=0.985)

    xs = np.linspace(0.10, 0.90, 5)
    stages = [
        ("Inputs &\nscenarios", "network, archetypes, prices;\n4 archetypes x 10 states"),
        ("Stage 1", "scenario-perfect\nbenchmark C*_s"),
        ("Stage 2", "shared robust\nmin-max regret design"),
        ("Operations\n& KPIs", "heat, flex, battery;\nPUE, siPUE, CUE"),
        ("Outputs", "manuscript-facing\nKPI bundle"),
    ]
    colors = ["#E5E7EB", "#FEF3C7", "#DCFCE7", "#E0F2FE", "#FFE4E6"]
    y_main = 0.74
    for i, (xpos, (title, detail)) in enumerate(zip(xs, stages)):
        ax.plot([xpos, xpos], [0.60, y_main + 0.08], color="#CBD5E1", lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([xpos], [y_main], s=620, color=colors[i], edgecolor="#334155", linewidth=1.6, zorder=3)
        ax.text(xpos, y_main, str(i + 1), ha="center", va="center", fontsize=_fs(12), weight="bold", zorder=4)
        ax.text(xpos, y_main + 0.17, title, ha="center", va="center", fontsize=_fs(11.5), weight="bold")
        ax.text(xpos, y_main - 0.15, detail, ha="center", va="center", fontsize=_fs(8.6), color="#334155", weight="medium")
        if i < len(xs) - 1:
            ax.plot([xpos + 0.035, xs[i + 1] - 0.035], [y_main, y_main], color="#334155", lw=2.4, zorder=1)

    branch_y = 0.16
    branches = [
        (xs[3] - 0.10, "Validation", "train/test +\nout-of-sample checks", "#DBEAFE", CASE_COLORS["baseline"]),
        (xs[3] + 0.10, "Sensitivity", "5-factor tornado:\nprice, cooling, land, delay", "#F3E8FF", LEDGER_COLORS["policy"]),
    ]
    for bx, title, detail, face, edge in branches:
        ax.annotate("", xy=(bx, branch_y + 0.10), xytext=(xs[3], y_main - 0.11), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#94A3B8"}, zorder=1)
        _text_box(ax, (bx, branch_y), f"{title}\n{detail}", face, edge, 8.2, zorder=3)
        ax.annotate("", xy=(xs[4] - 0.03, y_main - 0.11), xytext=(bx, branch_y + 0.10), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#94A3B8"}, zorder=1)

    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98)
    _add_takeaway(fig, "Held-out validation and a five-factor sensitivity tornado feed back into the reported KPI bundle alongside the two-stage optimization.")
    _save_figure(fig, out_dir, "fig_02b_full_study_roadmap", cfg)


def _fig_05_optimization_formulation_map(out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.suptitle("Uncertainty-aware design is optimized before KPI accounting", fontsize=_fs(16.5), weight="bold", y=0.985)
    boxes = {
        "benchmark": (0.18, 0.72, "Stage 1 benchmark\nC*_s\nscenario-perfect cost", "#FEF3C7"),
        "objective": (0.50, 0.72, "Stage 2 objective\nminimize max normalized regret\nthen planning cost", "#DCFCE7"),
        "decision": (0.82, 0.72, "Shared design variables\nHX, HP, FLEX, battery,\nefficiency", "#DBEAFE"),
        "constraints": (0.28, 0.38, "Constraints\ncarbon cap, land, heat,\nflex, battery, DC-OPF", "#F3E8FF"),
        "ops": (0.62, 0.38, "Operational evaluation\nhourly weighted energy,\ncost, emissions", "#E0F2FE"),
        "metrics": (0.50, 0.12, "Reported KPIs\nPUE, siPUE, CUE,\nModeled ERF, value, regret", "#FFE4E6"),
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
    ax.text(0.5, 0.93, "Stage 2 minimizes worst-case normalized cost regret; siPUE is computed on the resulting design.", ha="center", fontsize=_fs(10.5), color="#334155", weight="medium")
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.02, right=0.98)
    _save_figure(fig, out_dir, "fig_05_optimization_formulation_map", cfg)


def _fig_06_input_scenario_fingerprint(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    archetypes = _all_archetypes(rows)
    metrics = [
        ("IT energy", "e_it_mwh_no_services"),
        ("cooling overhead", "pue_no_services"),
        ("heat reuse", "heat_mwhth_robust"),
        ("flex output", "flex_load_mwh_robust"),
        ("land utilization", "land_utilization_pct_robust"),
        ("carbon cap", "carbon_cap_tco2"),
    ]
    matrix = []
    for archetype in archetypes:
        subset = [row for row in rows if _value(row, "Archetype") == archetype]
        matrix.append([_mean_numeric(subset, key) for _, key in metrics])
    data = np.array(matrix, dtype=float)
    if data.size:
        mins = data.min(axis=0)
        ranges = np.maximum(data.max(axis=0) - mins, 1e-9)
        data = (data - mins) / ranges
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Scenario fingerprint: what differs by archetype", fontsize=_fs(15.5), weight="bold", pad=14)
    ax.set_yticks(np.arange(len(archetypes)), archetypes)
    ax.set_xticks(np.arange(len(metrics)), [label for label, _ in metrics], rotation=30, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "#F8FAFC" if data[i, j] > 0.55 else "#0F172A"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=_fs(10), weight="bold", color=text_color)
    fig.colorbar(im, ax=ax, label="normalized relative level")
    fig.subplots_adjust(bottom=0.32, top=0.88)
    _add_takeaway(fig, "The heatmap is normalized within each column to show relative scenario structure.")
    _save_figure(fig, out_dir, "fig_06_input_scenario_fingerprint", cfg)


def _fig_07_representative_day_compression(out_dir: str, cfg: Config) -> None:
    try:
        ts = load_network_timeseries(cfg, T=8760)
        indices, weights, starts, total_days = build_representative_days(ts, cfg)
    except Exception as exc:
        _placeholder_figure(out_dir, "fig_07a_representative_day_weights", cfg, "Representative-day compression", f"Representative-day diagnostic unavailable: {exc}")
        _placeholder_figure(out_dir, "fig_07b_representative_day_load_trace", cfg, "Representative-day compression", f"Representative-day diagnostic unavailable: {exc}")
        return
    day_ids = [indices[start] // 24 + 1 for start in starts]
    day_weights = [sum(weights[start:start + 24]) / 24.0 for start in starts]
    hours = list(range(min(168, len(ts["load_N1_mw"]))))
    load = [ts["load_N1_mw"][i] + ts["load_N2_mw"][i] + ts["load_N3_mw"][i] for i in hours]
    ren = [ts["ren_avail_N1_mw"][i] + ts["ren_avail_N2_mw"][i] + ts["ren_avail_N3_mw"][i] for i in hours]
    takeaway = f"Selected {len(starts)} representative days; day weights sum to {sum(day_weights):.0f} over a {total_days:.0f}-day year."

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(day_ids, day_weights, width=8.0, color=CASE_COLORS["baseline"], alpha=0.85)
    ax.set_title("Representative days compress 8,760 hours into weighted operating days", fontsize=_fs(14.5), weight="bold")
    ax.set_ylabel("day weight")
    ax.set_xlabel("calendar day")
    _style_axis(ax)
    _add_takeaway(fig, takeaway)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "fig_07a_representative_day_weights", cfg)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(hours, load, label="network load", color=CASE_COLORS["baseline"], lw=2.2)
    ax.plot(hours, ren, label="renewable availability", color=CASE_COLORS["robust"], lw=2.2)
    ax.set_title("First-week load and renewable availability", fontsize=_fs(14.5), weight="bold")
    ax.set_ylabel("MW")
    ax.set_xlabel("first-week hour")
    ax.legend(frameon=False)
    _style_axis(ax)
    _save_figure(fig, out_dir, "fig_07b_representative_day_load_trace", cfg)


def _fig_08_network_and_carbon_context(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    fig, ax_net = plt.subplots(figsize=(6.0, 5.6))
    ax_net.axis("off")
    coords = {"N1": (0.50, 0.82), "N2": (0.18, 0.20), "N3": (0.82, 0.20)}
    for a, b in (("N1", "N2"), ("N2", "N3"), ("N1", "N3")):
        ax_net.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color="#94A3B8", lw=3.0)
    for node, (xpos, ypos) in coords.items():
        ax_net.scatter([xpos], [ypos], s=2200, color="#DBEAFE", edgecolor="#334155", linewidth=1.8, zorder=3)
        ax_net.text(xpos, ypos, node, ha="center", va="center", fontsize=_fs(14), weight="bold")
    ax_net.set_title("Three-node DC-OPF network topology", fontsize=_fs(14.5), weight="bold")
    ax_net.text(0.5, 0.02, "Data-center load is injected by site", ha="center", fontsize=_fs(10.5), color="#334155", weight="medium")
    _save_figure(fig, out_dir, "fig_08a_network_topology", cfg)

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
    ax_load.set_title("Network load and renewable availability", fontsize=_fs(15.5), weight="bold")
    ax_load.set_ylabel("MW")
    ax_load.set_xlabel("hour")
    ax_load.legend(frameon=False)
    _style_axis(ax_load)
    _save_figure(fig, out_dir, "fig_08b_load_and_renewables", cfg)

    archetypes = _all_archetypes(rows)
    caps = [_mean_numeric([row for row in rows if _value(row, "Archetype") == a], "carbon_cap_tco2") / 1e6 for a in archetypes]
    fig, ax_carbon = plt.subplots(figsize=(7.4, 5.0))
    ax_carbon.bar(np.arange(len(archetypes)), caps, color=LEDGER_COLORS["carbon"], alpha=0.88)
    ax_carbon.set_xticks(np.arange(len(archetypes)), archetypes, rotation=18, ha="right")
    ax_carbon.set_ylabel("carbon cap (M tCO2)")
    ax_carbon.set_title("Carbon cap by archetype", fontsize=_fs(15.5), weight="bold")
    _style_axis(ax_carbon)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "fig_08c_carbon_cap_by_archetype", cfg)


def _fig_03_pue_sipue_tradeoff_compass(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.1))
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
    ax.set_xlabel("PUE change vs services-disabled comparator")
    ax.set_ylabel("siPUE change vs services-disabled comparator")
    ax.set_title("Robust design pushes the PUE/siPUE trade-off beyond baseline", fontsize=_fs(14.5), weight="bold")
    all_x = tails_x + heads_x
    all_y = tails_y + heads_y
    if all_x:
        pad_x = 0.1 * ((max(all_x) - min(all_x)) or 1.0)
        pad_y = 0.1 * ((max(all_y) - min(all_y)) or 1.0)
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1], ylim[0], "PUE higher\nsiPUE lower", ha="right", va="bottom", fontsize=_fs(9), color=CASE_COLORS["robust"])
    ax.text(xlim[1], ylim[1], "PUE higher\nsiPUE higher", ha="right", va="top", fontsize=_fs(9), color="#991B1B")
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
    _add_takeaway(fig, "Each arrow moves one scenario from its baseline PUE/siPUE change to its robust-design change; both are measured against the constrained services-disabled comparator.")
    fig.subplots_adjust(right=0.74, bottom=0.22)
    _save_figure(fig, out_dir, "fig_03_pue_sipue_tradeoff_compass", cfg)


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


KPI_PANEL_LABELS = {
    "pue": "PUE",
    "sipue": "siPUE",
    "steady_sipue": "Post-construction siPUE",
    "cue": "CUE",
    "modeled_erf": "Modeled ERF",
    "system_benefit_per_mwh_it": "System benefit",
    "emissions_tco2": "Emissions",
    "carbon_damage_eur": "Carbon damage",
}


def _fig_04_kpi_benchmark_scoreboard(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.canonical_output_dir, "kpi_benchmark_summary.csv")
    kpi_rows = _read_csv_rows(path) or _kpi_rows_from_results(rows)
    metrics = ["pue", "sipue", "cue", "modeled_erf", "system_benefit_per_mwh_it", "emissions_tco2", "carbon_damage_eur"]
    archetypes = _all_archetypes(rows)
    designs = ("no_services", "fair_services_disabled", "baseline", "robust")
    x = np.arange(len(designs))

    def _series(subset: Sequence[Dict], archetype: str) -> List[float]:
        # For a single archetype, IT-energy weighting is a no-op (that
        # archetype's IT energy is constant across its own rows), so the
        # per-archetype lines are unaffected. For "all", which pools all
        # four archetypes, average_value_it_weighted matches the paper's
        # headline numbers; average_value (equal-weighted by archetype) does
        # not, so it is used only as a fallback for older CSVs that predate
        # the IT-weighted column.
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

    for metric in metrics:
        subset = [row for row in kpi_rows if _value(row, "Metric") == metric]
        if not subset:
            continue
        unit = _value(subset[0], "Unit", "")
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        drawn = False
        endpoint_labels = []
        for archetype in archetypes:
            values = _series(subset, archetype)
            if not values:
                continue
            color = ARCHETYPE_COLORS.get(archetype, "#64748B")
            ax.plot(x, values, color=color, lw=2.6, marker="o", markersize=7.5, markeredgecolor="white", markeredgewidth=1.0, alpha=0.92, zorder=3)
            endpoint_labels.append((float(values[-1]), archetype, color))
            drawn = True
        overall = _series(subset, "all")
        if overall:
            ax.plot(x, overall, color="#111827", lw=2.0, linestyle="--", marker="s", markersize=6, zorder=2, label="All archetypes (IT-weighted mean)")
            drawn = True
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xticks(x, [DESIGN_LABELS.get(d, d) for d in designs])
        ax.set_xlim(-0.15, len(designs) - 1 + 0.65)
        ax.set_ylabel(unit)
        ax.set_title(KPI_PANEL_LABELS.get(metric, metric), fontsize=_fs(15.5), weight="bold")
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ymin, ymax = ax.get_ylim()
        yrange = max(ymax - ymin, 1e-9)
        min_sep = 0.055 * yrange
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
        if overall:
            ax.legend(frameon=False, loc="best", fontsize=_fs(8.5))
        _style_axis(ax)
        fig.subplots_adjust(right=0.76, bottom=0.17)
        _save_figure(fig, out_dir, f"fig_04_kpi_{metric}", cfg)


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
    ax.set_ylabel("normalized value (0-1 per axis)")
    ax.set_title("Design portfolio shift across capacity dimensions", fontsize=_fs(15), weight="bold")
    handles = [Line2D([0], [0], color=ARCHETYPE_COLORS.get(a, "#64748B"), lw=2.6, label=a) for a in archetypes] + [
        Line2D([0], [0], color="#334155", lw=2.0, marker="o", linestyle="-", markersize=7, label="Baseline"),
        Line2D([0], [0], color="#334155", lw=2.0, marker="D", linestyle="--", markersize=6.5, label="Robust"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=_fs(8.5))
    _style_axis(ax, "y")
    ax.spines["bottom"].set_visible(False)
    _add_takeaway(fig, "Each axis is normalized independently, so portfolio shape across heat, flexibility, and storage capacity carries the comparison.")
    fig.subplots_adjust(right=0.78, bottom=0.16)
    _save_figure(fig, out_dir, "fig_09_design_portfolio_parallel", cfg)


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
    components = [
        ("heating revenue", "heat"),
        ("flexibility revenue", "flex"),
        ("resource electricity", "electricity"),
        ("cooling electricity", "electricity"),
        ("CO2 market", "carbon"),
        ("capex", "capex"),
        ("land", "land"),
        ("construction delay", "delay"),
    ]
    # Pooled across all four archetypes: weight each row by its own IT energy
    # rather than averaging all 40 rows equally, which would implicitly give
    # every archetype the same one-quarter weight regardless of its IT energy
    # scale (matches the IT-weighting used for the paper's headline numbers).
    values: Dict[str, float] = {}
    for component, _ in components:
        found = [
            (_num(row, "value_eur") / 1e6, _num(row, "e_it_mwh", 1.0))
            for row in decomposition
            if _value(row, "Design") == "robust" and _value(row, "Ledger") == "market" and _value(row, "Component") == component
        ]
        total_weight = sum(w for _, w in found)
        values[component] = (sum(v * w for v, w in found) / total_weight) if total_weight > 0 else 0.0

    inflows = [(c, k, values[c]) for c, k in components if values[c] > 0]
    outflows = [(c, k, values[c]) for c, k in components if values[c] < 0]
    if not inflows:
        _placeholder_figure(out_dir, "fig_10_system_value_sankey", cfg, "System value flow", "No positive value components found in the decomposition ledger.")
        return
    total_in = sum(value for _, _, value in inflows)
    total_out = -sum(value for _, _, value in outflows)
    net_value = total_in - total_out

    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("System value: gross revenue nets out to system value after costs and adjustments", fontsize=_fs(14.5), weight="bold", pad=14)

    max_w = 0.16
    gross_xy = (0.40, 0.62)
    net_xy = (0.95, 0.62)
    costs_xy = (0.62, 0.20)

    def _width(value: float) -> float:
        return max(0.016, max_w * abs(value) / total_in)

    n_in = len(inflows)
    in_ys = np.linspace(0.90, 0.32, n_in) if n_in > 1 else [0.62]
    for (component, key, value), y in zip(inflows, in_ys):
        color = LEDGER_COLORS.get(key, "#64748B")
        _ribbon(ax, (0.08, y), _width(value), gross_xy, _width(value), color, zorder=1)
        _text_box(ax, (0.08, y), f"{_wrap(component.title(), 14)}\n{value:+.1f} M EUR", "#F8FAFC", color, 8.0, zorder=3)

    _ribbon(ax, gross_xy, _width(net_value), net_xy, _width(net_value), CASE_COLORS["robust"], zorder=2)
    if outflows:
        _ribbon(ax, gross_xy, _width(total_out), costs_xy, _width(total_out), LEDGER_COLORS["capex"], zorder=2)

    _text_box(ax, gross_xy, f"Gross value\n{total_in:+.1f} M EUR", "#DCFCE7", CASE_COLORS["robust"], 9.5, zorder=4)
    _text_box(ax, net_xy, f"Net system\nvalue\n{net_value:+.1f} M EUR", "#DBEAFE", CASE_COLORS["baseline"], 9.5, zorder=4)

    if outflows:
        _text_box(ax, costs_xy, f"Costs &\nadjustments\n{-total_out:+.1f} M EUR", "#FEE2E2", LEDGER_COLORS["capex"], 9.0, zorder=4)
        n_out = len(outflows)
        out_ys = np.linspace(0.30, 0.02, n_out) if n_out > 1 else [0.12]
        for (component, key, value), y in zip(outflows, out_ys):
            color = LEDGER_COLORS.get(key, "#64748B")
            _ribbon(ax, costs_xy, _width(value), (0.92, y), _width(value), color, zorder=1)
            _text_box(ax, (0.92, y), f"{_wrap(component.title(), 16)}\n{value:+.1f} M EUR", "#FFF7ED", color, 8.0, zorder=3)

    _add_takeaway(fig, "Business value and policy costs are separated: heat/flex value can be offset by electricity, carbon, capex, land, and delay before net system value is realized.")
    _save_figure(fig, out_dir, "fig_10_system_value_sankey", cfg)


def _fig_11_carbon_ledger_separation(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    archetypes = _all_archetypes(rows)
    metrics = [
        ("power_emissions", "Power emissions", "dc_power_emissions_tco2_robust", "tCO2", False),
        ("avoided_heat_emissions", "Avoided heat emissions", "avoided_heat_emissions_tco2_robust", "tCO2", False),
        ("co2_cost_change", "CO2 cost change vs services-disabled", "co2_cost_reduction_pct_vs_fair_services_disabled_robust", "% vs services-disabled", True),
        ("carbon_damage_change", "Carbon damage change vs services-disabled", "carbon_damage_reduction_pct_vs_fair_services_disabled_robust", "% vs services-disabled", True),
    ]
    y = np.arange(len(archetypes))
    for slug, label, key, unit, is_pct in metrics:
        values = []
        for archetype in archetypes:
            subset = [row for row in rows if _value(row, "Archetype") == archetype]
            values.append(_mean_numeric(subset, key))
        color = LEDGER_COLORS["policy"] if is_pct else LEDGER_COLORS["carbon"]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.barh(y, values, color=color, alpha=0.88)
        if is_pct:
            ax.axvline(0.0, color="#222222", linewidth=1.2)
        ax.set_yticks(y, archetypes)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(label, fontsize=_fs(15), weight="bold")
        _style_axis(ax, "x")
        _add_takeaway(fig, "Emissions, market payments, avoided heat emissions, and carbon damage are tracked as separate ledgers.")
        fig.subplots_adjust(bottom=0.16, left=0.24)
        _save_figure(fig, out_dir, f"fig_11_carbon_ledger_{slug}", cfg)


def _fig_12_delay_credit_bridge(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    archetypes = _all_archetypes(rows)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    y = np.arange(len(archetypes))
    steady = []
    delay = []
    loss = []
    loss_weights = []
    for archetype in archetypes:
        subset = [row for row in rows if _value(row, "Archetype") == archetype]
        steady.append(_mean_numeric(subset, "steady_sipue_robust"))
        delay.append(_mean_numeric(subset, "sipue_robust"))
        loss.append(_mean_numeric(subset, "sipue_credit_loss_pct_robust"))
        loss_weights.append(_num(subset[0], "e_it_mwh_robust", 1.0) if subset else 1.0)
    for yi, a, b in zip(y, steady, delay):
        ax.plot([a, b], [yi, yi], color="#CBD5E1", lw=3.4, zorder=1)
    ax.scatter(steady, y, color="#22C55E", s=105, label="post-construction siPUE", zorder=2, edgecolor="white", linewidth=0.9)
    ax.scatter(delay, y, color=LEDGER_COLORS["delay"], marker="D", s=110, label="delay-adjusted siPUE", zorder=3, edgecolor="white", linewidth=0.9)
    ax.set_yticks(y, archetypes)
    ax.invert_yaxis()
    ax.margins(y=0.2)
    ax.set_xlabel("siPUE")
    ax.set_title("Construction delay weakens useful system credit", fontsize=_fs(15.5), weight="bold")
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax, "x")
    # Weight each archetype's mean loss by its IT energy rather than treating
    # all four archetypes as equally sized (matches the paper's IT-weighting).
    total_loss_weight = sum(loss_weights)
    mean_loss = (sum(v * w for v, w in zip(loss, loss_weights)) / total_loss_weight) if total_loss_weight > 0 else 0.0
    _add_takeaway(fig, f"Construction delay reduces the useful siPUE credit by about {mean_loss:.1f}% for every archetype -- the effect is uniform because delay is applied as a single planning assumption, not a per-archetype outcome.")
    fig.subplots_adjust(bottom=0.14)
    _save_figure(fig, out_dir, "fig_12_delay_credit_bridge", cfg)


def _fig_13_property_check_matrix(out_dir: str, cfg: Config) -> None:
    rows = _read_csv_rows(os.path.join(cfg.canonical_output_dir, "metric_property_tests.csv"))
    if not rows:
        _placeholder_figure(out_dir, "fig_13_property_check_matrix", cfg, "Metric property checks", "No metric_property_tests.csv found in the canonical output directory.")
        return
    seen_labels: Dict[str, int] = {}
    labels = []
    for row in rows:
        base = str(_value(row, "Test", "")).replace("_", " ")
        case = str(_value(row, "Case", "")).replace("_", " ")
        label = f"{base} ({case})" if case and case != "all" and base in seen_labels else base
        seen_labels[base] = seen_labels.get(base, 0) + 1
        labels.append(label)
    passed = [str(_value(row, "Passed", "")).lower() == "true" for row in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.6, max(3.6, 0.52 * n + 1.2)))
    y = np.arange(n)
    for yi, ok, label in zip(y, passed, labels):
        marker_color = "#16A34A" if ok else "#DC2626"
        ax.scatter([0.02], [yi], s=260, color=marker_color, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(0.02, yi, "OK" if ok else "X", ha="center", va="center", fontsize=_fs(9), weight="bold", color="white", zorder=4)
        ax.text(0.08, yi, label, ha="left", va="center", fontsize=_fs(11.5), weight="medium", color="#0F172A")
        status_text = "supported" if ok else "check failed"
        ax.text(0.98, yi, status_text, ha="right", va="center", fontsize=_fs(10.5), weight="bold", color=marker_color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Internal metric property checks", fontsize=_fs(15.5), weight="bold", pad=14)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.03, right=0.97)
    _add_takeaway(fig, "Use this as an internal correctness panel: no-services identity, nonnegativity, delay behavior, benefit monotonicity, and ledger separation.")
    _save_figure(fig, out_dir, "fig_13_property_check_matrix", cfg)


def _feasibility_checklist_figure(out_dir: str, cfg: Config, name: str, title: str, rows: List[Dict], metrics: List[Tuple[str, str]], case_noun: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(metrics) + 1.6))
    y = np.arange(len(metrics))
    for yi, (label, key) in zip(y, metrics):
        pct = _pct_true(rows, key)
        ok = pct >= 99.95
        color = "#16A34A" if ok else "#DC2626"
        ax.scatter([0.02], [yi], s=260, color=color, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(0.02, yi, "OK" if ok else "!", ha="center", va="center", fontsize=_fs(9), weight="bold", color="white", zorder=4)
        ax.text(0.09, yi, label, ha="left", va="center", fontsize=_fs(11.5), weight="medium", color="#0F172A")
        ax.text(0.98, yi, f"{pct:.0f}% of {case_noun}", ha="right", va="center", fontsize=_fs(10.5), weight="bold", color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(metrics) - 0.3)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=_fs(14.5), weight="bold", pad=12)
    fig.subplots_adjust(top=0.82, bottom=0.06, left=0.03, right=0.97)
    _save_figure(fig, out_dir, name, cfg)


def _fig_14_train_test_validation(out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.validation_output_dir, "train-test", f"train_test_1_{cfg.train_test_max_train_id}_vs_rest.csv")
    rows = _read_csv_rows(path)
    if not rows:
        _placeholder_figure(out_dir, "fig_14a_train_test_feasibility", cfg, "Train/test validation", "Train/test validation output is not present. Run run_train_test_scenario_validation.py before generating this diagnostic.")
        _placeholder_figure(out_dir, "fig_14b_train_test_regret", cfg, "Train/test validation", "Train/test validation output is not present. Run run_train_test_scenario_validation.py before generating this diagnostic.")
        return
    archetypes = _all_archetypes(rows)
    feasibility_metrics = [
        ("design feasible", "held_out_design_feasible"),
        ("land feasible", "held_out_land_feasible"),
        ("technical feasible", "held_out_technical_feasible"),
        ("carbon cap satisfied", "carbon_cap_satisfied"),
    ]
    _feasibility_checklist_figure(out_dir, cfg, "fig_14a_train_test_feasibility", "Held-out feasibility (train/test)", rows, feasibility_metrics, "held-out cases")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    regret_key = "regret_share_of_lifecycle" if any(_value(row, "regret_share_of_lifecycle") not in (None, "") for row in rows) else "RegretEURPerMWhIT"
    data = [[_num(row, regret_key) for row in rows if _value(row, "Archetype") == archetype and _value(row, regret_key) not in (None, "")] for archetype in archetypes]
    ax.boxplot(data, labels=archetypes, patch_artist=True, boxprops={"facecolor": "#DBEAFE", "color": CASE_COLORS["baseline"]}, medianprops={"color": "#111827", "linewidth": 2.0})
    ax.set_ylabel("% of lifecycle cost" if regret_key == "regret_share_of_lifecycle" else "EUR/MWh IT")
    ax.set_title("Held-out regret distribution (train/test)", fontsize=_fs(14.5), weight="bold")
    ax.tick_params(axis="x", rotation=18)
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.2)
    _save_figure(fig, out_dir, "fig_14b_train_test_regret", cfg)


def _fig_15_out_of_sample_validation(out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.validation_output_dir, "out-of-sample", "out_of_sample_validation.csv")
    rows = _read_csv_rows(path)
    if not rows:
        _placeholder_figure(out_dir, "fig_15a_out_of_sample_feasibility", cfg, "Out-of-sample validation", "Out-of-sample validation output is not present. Run run_out_of_sample_validation.py before generating this diagnostic.")
        _placeholder_figure(out_dir, "fig_15b_out_of_sample_regret", cfg, "Out-of-sample validation", "Out-of-sample validation output is not present. Run run_out_of_sample_validation.py before generating this diagnostic.")
        return
    archetypes = _all_archetypes(rows)
    metrics = [
        ("operational feasible", "operational_feasible"),
        ("perfect feasible", "carbon_constrained_perfect_feasible"),
        ("land feasible", "land_feasible"),
        ("carbon cap satisfied", "carbon_cap_satisfied"),
    ]
    _feasibility_checklist_figure(out_dir, cfg, "fig_15a_out_of_sample_feasibility", "Policy and technical feasibility (out-of-sample)", rows, metrics, "OOS cases")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    regret_key = "regret_share_of_lifecycle" if any(_value(row, "regret_share_of_lifecycle") not in (None, "") for row in rows) else "RegretEURPerMWhIT"
    data = [[_num(row, regret_key) for row in rows if _value(row, "Archetype") == archetype and _value(row, regret_key) not in (None, "")] for archetype in archetypes]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(CASE_COLORS["robust"])
        body.set_alpha(0.4)
        body.set_edgecolor(CASE_COLORS["robust"])
    ax.set_xticks(np.arange(1, len(archetypes) + 1), archetypes, rotation=18)
    ax.set_ylabel("% of lifecycle cost" if regret_key == "regret_share_of_lifecycle" else "EUR/MWh IT")
    ax.set_title("Out-of-sample regret distribution", fontsize=_fs(14.5), weight="bold")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.2)
    _save_figure(fig, out_dir, "fig_15b_out_of_sample_regret", cfg)


def _fig_16_provenance_map(out_dir: str, cfg: Config) -> None:
    path = os.path.join(cfg.canonical_output_dir, "manifest.json")
    if not os.path.isfile(path):
        _placeholder_figure(out_dir, "fig_16_provenance_map", cfg, "Provenance map", "No manifest.json found in the canonical output directory.")
        return
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    counts = {
        "Inputs": len(manifest.get("input_sha256", {})),
        "Source files": len(manifest.get("source_sha256", {})),
        "Outputs": len(manifest.get("output_sha256", {})),
        "Scenarios": int(manifest.get("scenario_count", 0)),
    }
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axis("off")
    ax.set_title("Evidence bundle provenance", fontsize=_fs(15), weight="bold")
    xs = [0.14, 0.38, 0.62, 0.86]
    colors = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#F3E8FF"]
    for idx, ((label, count), xpos, color) in enumerate(zip(counts.items(), xs, colors)):
        ax.add_patch(patches.FancyBboxPatch((xpos - 0.09, 0.43), 0.18, 0.22, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=color, edgecolor="#94A3B8"))
        ax.text(xpos, 0.57, label, ha="center", va="center", fontsize=_fs(10), weight="bold")
        ax.text(xpos, 0.47, str(count), ha="center", va="center", fontsize=_fs(14), color="#111827")
        if idx < len(xs) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.105, 0.54), xytext=(xpos + 0.105, 0.54), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    ax.text(0.5, 0.24, f"RunID: {manifest.get('run_id', 'unavailable')}", ha="center", fontsize=_fs(9), color="#475569")
    ax.text(0.5, 0.15, f"Git revision: {str(manifest.get('git_revision', 'unavailable'))[:16]}", ha="center", fontsize=_fs(9), color="#475569")
    _add_takeaway(fig, "This figure is for reproducibility: source, data, scenarios, and generated outputs are hash-tracked in the manifest.")
    _save_figure(fig, out_dir, "fig_16_provenance_map", cfg)


def _fig_17_sensitivity_tornado(rows: Sequence[Dict], out_dir: str, cfg: Config) -> None:
    base = [row for row in rows if _value(row, "Factor") == "canonical" and _value(row, "Archetype") == "all"]
    if not base:
        return
    # "all" pools all four archetypes, so use the IT-energy-weighted column to
    # match the paper's headline numbers (equal-weighting by archetype would
    # give a small archetype the same weight as a much larger one); fall back
    # to the equal-weighted column for older CSVs that predate it.
    sipue_key = "average_sipue_it_weighted" if "average_sipue_it_weighted" in base[0] else "average_sipue"
    base_sipue = _num(base[0], sipue_key)
    entries = []
    for factor in sorted({str(_value(row, "Factor")) for row in rows if _value(row, "Factor") not in ("canonical", None, "")}):
        if factor not in ALLOWED_SENSITIVITY_FACTORS:
            continue
        subset = [row for row in rows if _value(row, "Factor") == factor and _value(row, "Archetype") == "all"]
        if not subset:
            continue
        low = min(_num(row, sipue_key) for row in subset) - base_sipue
        high = max(_num(row, sipue_key) for row in subset) - base_sipue
        entries.append((str(factor), low, high))
    if not entries:
        return
    entries.sort(key=lambda item: abs(item[2] - item[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(entries))
    for yi, (factor, low, high) in zip(y, entries):
        ax.barh(yi, high - low, left=low, color=CASE_COLORS["robust"], alpha=0.72)
        ax.text(max(low, high), yi, f" {high - low:+.3g}", va="center", fontsize=_fs(8))
    ax.axvline(0.0, color="#111827", lw=1.0)
    ax.set_yticks(y, [label.title() for label, _, _ in entries])
    ax.invert_yaxis()
    ax.set_xlabel("robust siPUE change vs canonical sensitivity case")
    ax.set_title("Five-factor sensitivity tornado", fontsize=_fs(15), weight="bold")
    _style_axis(ax, "x")
    _save_figure(fig, out_dir, "fig_17_sensitivity_tornado", cfg)


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
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(np.array(matrix), cmap="PuBuGn", aspect="auto")
    ax.set_yticks(np.arange(len(archetypes)), archetypes)
    ax.set_xticks(np.arange(len(factors)), [label.title() for label in factors], rotation=25, ha="right")
    ax.set_title("Sensitivity response by archetype", fontsize=_fs(15), weight="bold")
    fig.colorbar(im, ax=ax, label="siPUE response range")
    _save_figure(fig, out_dir, "fig_18_sensitivity_archetype_response", cfg)


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
    _fig_01_sipue_metric_story(out_dir, cfg)
    _fig_01b_why_sipue_matters(out_dir, cfg)
    _fig_02_model_pipeline_subway(out_dir, cfg)
    _fig_02b_full_study_roadmap(out_dir, cfg)
    _fig_03_pue_sipue_tradeoff_compass(rows, out_dir, cfg)
    _fig_04_kpi_benchmark_scoreboard(rows, out_dir, cfg)
    _fig_05_optimization_formulation_map(out_dir, cfg)
    _fig_06_input_scenario_fingerprint(rows, out_dir, cfg)
    _fig_07_representative_day_compression(out_dir, cfg)
    _fig_08_network_and_carbon_context(rows, out_dir, cfg)
    _fig_09_design_portfolio_by_archetype(rows, out_dir, cfg)
    _fig_10_system_value_sankey(decomposition, out_dir, cfg)
    _fig_11_carbon_ledger_separation(rows, out_dir, cfg)
    _fig_12_delay_credit_bridge(rows, out_dir, cfg)
    _fig_13_property_check_matrix(out_dir, cfg)
    _fig_14_train_test_validation(out_dir, cfg)
    _fig_15_out_of_sample_validation(out_dir, cfg)
    _fig_16_provenance_map(out_dir, cfg)
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


def export_plots(rows: List[Dict], cfg: Config) -> None:
    """Export descriptive plots only; no constructed-target correlations."""
    if not cfg.export_plots or not rows:
        return
    out_dir = _ensure_plot_dir(cfg)
    labels = _labels(rows)
    x = np.arange(len(rows))
    case = "robust"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    pue_delta = [_num(row, f"pue_{case}") - _num(row, "pue_fair_services_disabled") for row in rows]
    sipue_delta = [_num(row, f"sipue_{case}") - _num(row, "sipue_fair_services_disabled") for row in rows]
    axes[0].bar(x, pue_delta, color="#4C78A8")
    axes[1].bar(x, sipue_delta, color="#59A14F")
    for ax, ylabel in zip(axes, ("PUE change", "delay-aware siPUE change")):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "pue_sipue_system_direction", cfg)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    credit_loss = [_num(row, f"sipue_credit_loss_mwh_{case}") for row in rows]
    ax.bar(x, credit_loss, color="#E15759")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("siPUE credit loss (MWh)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "delay_credit_loss", cfg)

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    width = 0.36
    market = [_num(row, f"carbon_market_cost_eur_{case}") / 1e6 for row in rows]
    damage = [_num(row, f"carbon_damage_eur_{case}") / 1e6 for row in rows]
    ax.bar(x - width / 2, market, width, label="CO2 market cost", color="#F28E2B")
    ax.bar(x + width / 2, damage, width, label="carbon damage", color="#B07AA1")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Separate carbon ledgers (million EUR)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "carbon_ledgers", cfg)

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    cooling = [_num(row, f"cooling_mwh_{case}") for row in rows]
    heat_it = [_num(row, f"recovered_heat_it_mwhth_{case}") for row in rows]
    heat_cooling = [_num(row, f"recovered_heat_cooling_mwhth_{case}") for row in rows]
    ax.bar(x, cooling, label="cooling electricity", color="#4C78A8")
    ax.bar(x, heat_it, bottom=cooling, label="IT-source recovered heat", color="#59A14F")
    ax.bar(x, heat_cooling, bottom=np.array(cooling) + np.array(heat_it), label="cooling-source recovered heat", color="#9C755F")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Energy (MWh)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_dir, "cooling_heat_sources", cfg)


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
    colors = ["#2F6B8A", "#3A7D44", "#C17C2C", "#8E5C7A"]
    x = np.arange(len(archetypes))

    def values(archetype: str, key: str) -> List[float]:
        return [_num(row, key) for row in rows if _value(row, "Archetype") == archetype]

    def mean_range(archetype: str, key: str):
        data = values(archetype, key)
        mean = float(np.mean(data)) if data else 0.0
        return mean, mean - min(data, default=mean), max(data, default=mean) - mean

    def _dot_range_figure(name: str, key: str, ylabel: str, title: str) -> None:
        stats = [mean_range(a, key) for a in archetypes]
        means = [item[0] for item in stats]
        lower = [item[1] for item in stats]
        upper = [item[2] for item in stats]
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#94A3B8", elinewidth=2.2, capsize=5, capthick=2.2, zorder=1)
        ax.scatter(x, means, s=140, color=colors, zorder=3, edgecolor="white", linewidth=1.2)
        ax.axhline(0.0, color="#222222", linewidth=1.1)
        ax.set_xticks(x, archetypes, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=_fs(14.5), weight="bold")
        _style_axis(ax)
        fig.subplots_adjust(bottom=0.22)
        _save_figure(fig, out_dir, name, cfg)

    _dot_range_figure("metric_change_pue", "pue_reduction_pct_vs_fair_services_disabled_robust", "PUE reduction vs services-disabled (%)", "PUE change by archetype")
    _dot_range_figure("metric_change_sipue", "sipue_reduction_pct_vs_fair_services_disabled_robust", "siPUE reduction vs services-disabled (%)", "siPUE change by archetype")
    _dot_range_figure("cost_change_lifecycle", "lifecycle_cost_reduction_pct_vs_fair_services_disabled_robust", "Lifecycle cost reduction (%)", "Lifecycle cost change by archetype")
    _dot_range_figure("emissions_change_dc_attributed", "dc_attributed_emissions_reduction_pct_vs_fair_services_disabled_robust", "DC-attributable emissions reduction (%)", "Emissions change by archetype")

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
    ax.set_title("Robust design regret by archetype", fontsize=_fs(14.5), weight="bold")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "regret_comparison_by_archetype", cfg)

    delay_loss = [float(np.mean(values(name, "delay_loss_pct_of_zero_delay_gain_robust"))) for name in archetypes]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(x, delay_loss, s=160, color=colors, zorder=3, edgecolor="white", linewidth=1.2)
    ax.vlines(x, 0.0, delay_loss, color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.axhline(0.0, color="#222222", linewidth=1.1)
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Delay loss (% of zero-delay gain)")
    ax.set_title("Construction-delay value loss by archetype", fontsize=_fs(14.5), weight="bold")
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "construction_delay_effects", cfg)

    market = [float(np.mean(values(name, "carbon_market_cost_eur_robust"))) / 1e6 for name in archetypes]
    damage = [float(np.mean(values(name, "carbon_damage_eur_robust"))) / 1e6 for name in archetypes]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(x - 0.18, market, 0.36, label="ETS payment", color="#C17C2C")
    ax.bar(x + 0.18, damage, 0.36, label="carbon damage", color="#8E5C7A")
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Annual value (million EUR)")
    ax.set_title("Carbon market cost vs social damage", fontsize=_fs(14.5), weight="bold")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.22)
    _save_figure(fig, out_dir, "carbon_and_network_outcomes_cost", cfg)

    congestion = [float(np.mean(values(name, "congestion_hours_robust"))) for name in archetypes]
    curtailment = [float(np.mean(values(name, "renewable_curtailment_mwh_robust"))) for name in archetypes]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(x, congestion, s=160, color=CASE_COLORS["baseline"], zorder=3, edgecolor="white", linewidth=1.2)
    ax.vlines(x, 0.0, congestion, color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Congestion hours per year")
    ax.set_title("Network congestion by archetype", fontsize=_fs(14.5), weight="bold")
    _style_axis(ax)
    max_curtailment = max(curtailment, default=0.0)
    _add_takeaway(fig, f"Renewable curtailment is negligible for every archetype (at most {max_curtailment:.1f} MWh/year) -- the network is congested but not curtailing renewables.")
    fig.subplots_adjust(bottom=0.24)
    _save_figure(fig, out_dir, "carbon_and_network_outcomes_congestion", cfg)

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
                _num(row, "value_eur") / 1e6
                for row in decomposition
                if _value(row, "Archetype") == archetype
                and _value(row, "Design") == "robust"
                and _value(row, "Ledger") == "market"
                and _value(row, "Component") == component
            ]
            data.append(float(np.mean(found)) if found else 0.0)
        data_array = np.array(data)
        bottoms = np.where(data_array >= 0.0, positive, negative)
        ax.bar(x, data_array, bottom=bottoms, label=component, color=color)
        positive += np.where(data_array >= 0.0, data_array, 0.0)
        negative += np.where(data_array < 0.0, data_array, 0.0)
    ax.axhline(0.0, color="#222222", linewidth=1.1)
    ax.set_xticks(x, archetypes, rotation=18, ha="right")
    ax.set_ylabel("Average lifecycle value contribution (million EUR)")
    ax.set_title("Market value decomposition by archetype", fontsize=_fs(14.5), weight="bold")
    ax.legend(ncol=2, frameon=False, fontsize=_fs(9.5))
    _style_axis(ax)
    fig.subplots_adjust(bottom=0.2)
    _save_figure(fig, out_dir, "market_value_decomposition", cfg)

    export_creative_figure_package(rows, comparison, decomposition, cfg)

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
    _save_figure(fig, out_dir, "regret_surface_hx_flex_contour", cfg)
