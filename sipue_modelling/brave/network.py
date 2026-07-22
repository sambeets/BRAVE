from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence, Tuple

import pulp

from brave.config import Config
from brave.solver import make_solver_lp


def _ensure_network_defaults(cfg: Config) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    if cfg.node_assignment is None:
        cfg.node_assignment = {
            "Norway": "N1",
            "Sweden": "N1",
            "Denmark": "N1",
            "Finland": "N1",
            "Iceland": "N1",
            "Germany": "N2",
            "Belgium": "N2",
            "Netherlands": "N2",
            "France": "N2",
            "United Kingdom": "N3",
        }

    if cfg.node_cost_base is None:
        cfg.node_cost_base = {"N1": 48.0, "N2": 65.0, "N3": 72.0}
    if cfg.node_cost_amp is None:
        cfg.node_cost_amp = {"N1": 10.0, "N2": 14.0, "N3": 16.0}
    if cfg.node_load_base is None:
        cfg.node_load_base = {"N1": 35.0, "N2": 45.0, "N3": 30.0}
    if cfg.node_load_amp is None:
        cfg.node_load_amp = {"N1": 10.0, "N2": 14.0, "N3": 9.0}

    return cfg.node_assignment, cfg.node_cost_base, cfg.node_cost_amp, cfg.node_load_base, cfg.node_load_amp


_NETWORK_TS_CACHE: Dict[str, Dict[str, List[float]]] = {}


def _resolve_timeseries_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.normpath(path),  # relative to CWD
        os.path.normpath(os.path.join(here, path)),  # relative to src/
        os.path.normpath(os.path.join(here, "..", path)),  # relative to sipue-modelling/
        os.path.normpath(os.path.join(here, "..", "..", path)),  # relative to repo root
    ]

    prefix = "sipue-modelling" + os.sep
    if path.startswith(prefix) or path.startswith("sipue-modelling/"):
        stripped = path.split("/", 1)[1] if "/" in path else path.split(os.sep, 1)[1]
        candidates.extend([
            os.path.normpath(stripped),
            os.path.normpath(os.path.join(here, "..", stripped)),
        ])

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    tried = "', '".join(candidates)
    raise FileNotFoundError(f"Network time series CSV not found. Tried: '{tried}'")


def load_network_timeseries(cfg: Config, T: int) -> Dict[str, List[float]]:
    """
    Load calibrated network time series.

    Required columns:
      - t
      - load_N1_mw, load_N2_mw, load_N3_mw
      - ren_avail_N1_mw, ren_avail_N2_mw, ren_avail_N3_mw
      - fuel_eur_per_mwh_fuel
    Optional columns:
      - co2_eur_per_t
      - amb_temp_c
      - dh_supply_temp_c
    """
    path = _resolve_timeseries_path(cfg.network_timeseries_csv)
    cache_key = f"{path}::T{T}"
    if cache_key in _NETWORK_TS_CACHE:
        return _NETWORK_TS_CACHE[cache_key]

    series: Dict[str, List[float]] = {
        "load_N1_mw": [],
        "load_N2_mw": [],
        "load_N3_mw": [],
        "ren_avail_N1_mw": [],
        "ren_avail_N2_mw": [],
        "ren_avail_N3_mw": [],
        "fuel_eur_per_mwh_fuel": [],
        "co2_eur_per_t": [],
        "amb_temp_c": [],
        "dh_supply_temp_c": [],
    }

    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"Network time series CSV has no header: {path}")
        cols = set(rdr.fieldnames)
        required = [
            "t",
            "load_N1_mw", "load_N2_mw", "load_N3_mw",
            "ren_avail_N1_mw", "ren_avail_N2_mw", "ren_avail_N3_mw",
            "fuel_eur_per_mwh_fuel",
        ]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"Network time series CSV missing columns {missing}: {path}")

        has_co2 = "co2_eur_per_t" in cols
        has_amb = "amb_temp_c" in cols
        has_dh = "dh_supply_temp_c" in cols

        for row in rdr:
            t_s = row.get("t", "")
            if t_s == "":
                continue
            try:
                t_i = int(float(t_s))
            except Exception as exc:
                raise ValueError(f"Invalid t value '{t_s}' in {path}") from exc
            if t_i != len(series["fuel_eur_per_mwh_fuel"]):
                raise ValueError(f"Non-sequential t in {path}: expected {len(series['fuel_eur_per_mwh_fuel'])}, got {t_i}")

            for k in [
                "load_N1_mw", "load_N2_mw", "load_N3_mw",
                "ren_avail_N1_mw", "ren_avail_N2_mw", "ren_avail_N3_mw",
                "fuel_eur_per_mwh_fuel",
            ]:
                try:
                    series[k].append(float(row[k]))
                except Exception as exc:
                    raise ValueError(f"Invalid float in {path} column {k} at t={t_i}: {row.get(k)!r}") from exc

            if has_co2:
                series["co2_eur_per_t"].append(float(row["co2_eur_per_t"]))
            if has_amb:
                series["amb_temp_c"].append(float(row["amb_temp_c"]))
            if has_dh:
                series["dh_supply_temp_c"].append(float(row["dh_supply_temp_c"]))

            if len(series["fuel_eur_per_mwh_fuel"]) >= T:
                break

    if len(series["fuel_eur_per_mwh_fuel"]) < T:
        raise ValueError(f"Network time series CSV too short for T={T}: got {len(series['fuel_eur_per_mwh_fuel'])} rows. Path: {path}")

    if not series["co2_eur_per_t"]:
        del series["co2_eur_per_t"]
    if not series["amb_temp_c"]:
        del series["amb_temp_c"]
    if not series["dh_supply_temp_c"]:
        del series["dh_supply_temp_c"]

    _NETWORK_TS_CACHE[cache_key] = series
    return series


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = min(1.0, max(0.0, quantile)) * sum(max(0.0, weight) for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0]) if ordered else 0.0


def solve_no_dc_network(
    profile: Dict[str, List[float]],
    cfg: Config,
    line_caps: Tuple[float, float, float],
    load_perturbation_mw: float = 0.0,
) -> Dict[str, object]:
    """Resource-only no-DC dispatch used to freeze network and siPUE inputs."""
    node_loads = profile["node_loads"]
    ren_avail = profile["ren_avail"]
    fuel_price = profile["fuel_price"]
    weights = profile.get("w", [1.0] * len(fuel_price))
    T = len(fuel_price)
    n1, n2, n3 = cfg.network_nodes
    cap12, cap23, cap13 = line_caps
    b12 = cfg.line_susceptance_12_mw_per_rad
    b23 = cfg.line_susceptance_23_mw_per_rad
    b13 = cfg.line_susceptance_13_mw_per_rad
    th_eff = {"N1": cfg.th_eff_N1, "N2": cfg.th_eff_N2, "N3": cfg.th_eff_N3}
    th_vom = {"N1": cfg.th_vom_N1, "N2": cfg.th_vom_N2, "N3": cfg.th_vom_N3}
    th_cap = {"N1": cfg.th_cap_N1_mw, "N2": cfg.th_cap_N2_mw, "N3": cfg.th_cap_N3_mw}
    th_ef = {"N1": cfg.th_ef_N1, "N2": cfg.th_ef_N2, "N3": cfg.th_ef_N3}

    model = pulp.LpProblem("NoDCNetworkCalibration", pulp.LpMinimize)
    g_ren = {n: pulp.LpVariable.dicts(f"cal_G_REN_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
    g_th = {n: pulp.LpVariable.dicts(f"cal_G_TH_{n}", range(T), lowBound=0) for n in cfg.network_nodes}
    theta = {n: pulp.LpVariable.dicts(f"cal_THETA_{n}", range(T), lowBound=None) for n in cfg.network_nodes}
    f12 = pulp.LpVariable.dicts("cal_F12", range(T), lowBound=None)
    f23 = pulp.LpVariable.dicts("cal_F23", range(T), lowBound=None)
    f13 = pulp.LpVariable.dicts("cal_F13", range(T), lowBound=None)

    for t in range(T):
        total_load = sum(node_loads[n][t] for n in cfg.network_nodes)
        perturb = {
            n: load_perturbation_mw * node_loads[n][t] / max(1e-9, total_load)
            for n in cfg.network_nodes
        }
        model += theta[n1][t] == 0.0
        model += f12[t] == b12 * (theta[n1][t] - theta[n2][t])
        model += f23[t] == b23 * (theta[n2][t] - theta[n3][t])
        model += f13[t] == b13 * (theta[n1][t] - theta[n3][t])
        model += f12[t] <= cap12
        model += -f12[t] <= cap12
        model += f23[t] <= cap23
        model += -f23[t] <= cap23
        model += f13[t] <= cap13
        model += -f13[t] <= cap13
        for n in cfg.network_nodes:
            model += g_ren[n][t] <= ren_avail[n][t]
            model += g_th[n][t] <= th_cap[n]
        model += g_ren[n1][t] + g_th[n1][t] - f12[t] - f13[t] == node_loads[n1][t] + perturb[n1]
        model += g_ren[n2][t] + g_th[n2][t] + f12[t] - f23[t] == node_loads[n2][t] + perturb[n2]
        model += g_ren[n3][t] + g_th[n3][t] + f13[t] + f23[t] == node_loads[n3][t] + perturb[n3]

    model += pulp.lpSum(
        (fuel_price[t] / max(1e-9, th_eff[n]) + th_vom[n]) * g_th[n][t] * weights[t]
        for n in cfg.network_nodes for t in range(T)
    )
    model.solve(make_solver_lp(cfg))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"No-DC network calibration failed with status {status}.")
    flows = {
        "12": [float(pulp.value(f12[t])) for t in range(T)],
        "23": [float(pulp.value(f23[t])) for t in range(T)],
        "13": [float(pulp.value(f13[t])) for t in range(T)],
    }
    util = [
        max(
            abs(flows["12"][t]) / max(1e-9, cap12),
            abs(flows["23"][t]) / max(1e-9, cap23),
            abs(flows["13"][t]) / max(1e-9, cap13),
        )
        for t in range(T)
    ]
    total_hours = sum(weights)
    binding_hours = sum(
        weights[t] for t in range(T)
        if util[t] >= 1.0 - cfg.line_binding_tolerance
    )
    renewable_available = sum(
        ren_avail[n][t] * weights[t] for n in cfg.network_nodes for t in range(T)
    )
    renewable_used = sum(
        float(pulp.value(g_ren[n][t])) * weights[t]
        for n in cfg.network_nodes for t in range(T)
    )
    emissions_tco2 = sum(
        th_ef[n] * float(pulp.value(g_th[n][t])) * weights[t]
        for n in cfg.network_nodes for t in range(T)
    )
    return {
        "status": status,
        "resource_cost_eur": float(pulp.value(model.objective)),
        "flows": flows,
        "congestion_fraction": binding_hours / max(1e-9, total_hours),
        "congestion_hours": binding_hours,
        "renewable_curtailment_mwh": max(0.0, renewable_available - renewable_used),
        "emissions_tco2": emissions_tco2,
        "load_shedding_mwh": 0.0,
    }


def calibrate_no_dc_network(profile: Dict[str, List[float]], cfg: Config) -> Dict[str, float]:
    """Freeze line limits at the no-DC target and derive resource-only p_ref."""
    weights = profile.get("w", [1.0] * len(profile["fuel_price"]))
    unconstrained = solve_no_dc_network(profile, cfg, (1e6, 1e6, 1e6))
    target = min(0.95, max(0.05, cfg.congestion_target_fraction))
    quantile = 1.0 - target
    raw_caps = tuple(
        max(1.0, _weighted_quantile(
            [abs(value) for value in unconstrained["flows"][line]],
            weights,
            quantile,
        ))
        for line in ("12", "23", "13")
    )
    candidates = []
    for scale in (0.70 + 0.05 * index for index in range(27)):
        caps = tuple(max(1.0, value * scale) for value in raw_caps)
        dispatch = solve_no_dc_network(profile, cfg, caps)
        candidates.append((abs(float(dispatch["congestion_fraction"]) - target), caps, dispatch))
    _, caps, dispatch = min(candidates, key=lambda item: item[0])
    perturb = max(1e-6, cfg.sipue_reference_price_perturbation_mw)
    perturbed = solve_no_dc_network(profile, cfg, caps, load_perturbation_mw=perturb)
    added_energy = perturb * sum(weights)
    reference_price = (
        float(perturbed["resource_cost_eur"]) - float(dispatch["resource_cost_eur"])
    ) / max(1e-9, added_energy)
    if reference_price <= 0.0:
        raise RuntimeError("No-DC resource marginal cost must be positive.")
    return {
        "line_cap_12_mw": caps[0],
        "line_cap_23_mw": caps[1],
        "line_cap_13_mw": caps[2],
        "congestion_fraction": float(dispatch["congestion_fraction"]),
        "congestion_hours": float(dispatch["congestion_hours"]),
        "load_shedding_mwh": float(dispatch["load_shedding_mwh"]),
        "renewable_curtailment_mwh": float(dispatch["renewable_curtailment_mwh"]),
        "reference_price_eur_per_mwh": reference_price,
    }
