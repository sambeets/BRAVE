from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from brave.config import Config


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
