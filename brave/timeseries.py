from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from brave.config import Config
from brave.scenario import Scenario
from brave.network import _ensure_network_defaults, load_network_timeseries


def _daily_mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _squared_distance(a: List[float], b: List[float]) -> float:
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _standardize_feature_rows(rows: List[List[float]]) -> List[List[float]]:
    if not rows:
        return []

    cols = list(zip(*rows))
    means = [_daily_mean(list(col)) for col in cols]
    scales = []
    for idx, col in enumerate(cols):
        variance = _daily_mean([(val - means[idx]) ** 2 for val in col])
        scales.append(max(variance ** 0.5, 1e-9))

    return [
        [(row[idx] - means[idx]) / scales[idx] for idx in range(len(row))]
        for row in rows
    ]


def _allocate_quotas(stratum_sizes: List[int], target_days: int) -> List[int]:
    n = len(stratum_sizes)
    if target_days <= 0 or n == 0:
        return [0] * n

    total = max(1, sum(stratum_sizes))
    quotas = [0] * n
    remaining = target_days

    if target_days >= n:
        quotas = [1] * n
        remaining -= n

    if remaining <= 0:
        return quotas

    raw = [remaining * size / total for size in stratum_sizes]
    add = [int(val) for val in raw]
    quotas = [quotas[idx] + add[idx] for idx in range(n)]
    assigned = sum(add)
    remainders = sorted(
        range(n),
        key=lambda idx: (raw[idx] - add[idx], stratum_sizes[idx]),
        reverse=True,
    )
    for idx in remainders[: remaining - assigned]:
        quotas[idx] += 1

    return quotas


def _fill_missing_medoids(
    selected: List[int],
    dist: List[List[float]],
    total_points: int,
    target_k: int,
) -> List[int]:
    medoids = selected[:]
    while len(medoids) < target_k and len(medoids) < total_points:
        best_idx = None
        best_dist = -1.0
        for cand in range(total_points):
            if cand in medoids:
                continue
            if medoids:
                min_dist = min(dist[cand][mid] for mid in medoids)
            else:
                min_dist = 0.0
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = cand
        if best_idx is None:
            break
        medoids.append(best_idx)
    return medoids


def _select_stratum_medoids(feature_rows: List[List[float]], target_k: int) -> Tuple[List[int], List[int]]:
    n = len(feature_rows)
    if target_k >= n:
        return list(range(n)), [1] * n

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = _squared_distance(feature_rows[i], feature_rows[j])
            dist[i][j] = dij
            dist[j][i] = dij

    centroid = [
        _daily_mean([row[col_idx] for row in feature_rows])
        for col_idx in range(len(feature_rows[0]))
    ]
    first = min(range(n), key=lambda idx: _squared_distance(feature_rows[idx], centroid))
    medoids = [first]
    medoids = _fill_missing_medoids(medoids, dist, n, target_k)

    for _ in range(5):
        clusters = {mid: [] for mid in medoids}
        for idx in range(n):
            mid = min(medoids, key=lambda cand: dist[idx][cand])
            clusters[mid].append(idx)

        updated = []
        for mid in medoids:
            members = clusters[mid]
            if not members:
                continue
            best = min(
                members,
                key=lambda cand: sum(dist[cand][other] for other in members),
            )
            updated.append(best)
        updated = list(dict.fromkeys(updated))
        updated = _fill_missing_medoids(updated, dist, n, target_k)
        if sorted(updated) == sorted(medoids):
            medoids = updated
            break
        medoids = updated

    clusters = {mid: [] for mid in medoids}
    for idx in range(n):
        mid = min(medoids, key=lambda cand: dist[idx][cand])
        clusters[mid].append(idx)

    ordered = sorted((mid, len(clusters[mid])) for mid in medoids)
    return [mid for mid, _ in ordered], [weight for _, weight in ordered]


def _daily_feature_rows(ts: Dict[str, List[float]], total_days: int) -> List[List[float]]:
    rows: List[List[float]] = []
    amb_series = ts.get("amb_temp_c")
    dh_series = ts.get("dh_supply_temp_c")

    for day_idx in range(total_days):
        start = day_idx * 24
        end = start + 24
        fuel = ts["fuel_eur_per_mwh_fuel"][start:end]
        load_n1 = ts["load_N1_mw"][start:end]
        load_n2 = ts["load_N2_mw"][start:end]
        load_n3 = ts["load_N3_mw"][start:end]
        ren_n1 = ts["ren_avail_N1_mw"][start:end]
        ren_n2 = ts["ren_avail_N2_mw"][start:end]
        ren_n3 = ts["ren_avail_N3_mw"][start:end]

        load_total = [load_n1[h] + load_n2[h] + load_n3[h] for h in range(24)]
        ren_total = [ren_n1[h] + ren_n2[h] + ren_n3[h] for h in range(24)]
        net_load = [max(0.0, load_total[h] - ren_total[h]) for h in range(24)]

        angle = 2.0 * math.pi * (day_idx + 0.5) / max(1, total_days)
        features = [
            _daily_mean(fuel),
            max(fuel),
            _daily_mean(load_total),
            max(load_total),
            _daily_mean(load_n1),
            _daily_mean(load_n2),
            _daily_mean(load_n3),
            _daily_mean(ren_total),
            max(ren_total),
            _daily_mean(ren_n1),
            _daily_mean(ren_n2),
            _daily_mean(ren_n3),
            _daily_mean(net_load),
            max(net_load),
            math.sin(angle),
            math.cos(angle),
        ]

        if amb_series is not None:
            amb = [float(val) for val in amb_series[start:end]]
            features.append(_daily_mean(amb))
        if dh_series is not None:
            dh = [float(val) for val in dh_series[start:end]]
            features.append(_daily_mean(dh))

        rows.append(features)

    return _standardize_feature_rows(rows)


def build_representative_days(ts: Dict[str, List[float]], cfg: Config) -> Tuple[List[int], List[float], List[int], float]:
    total_hours = len(ts["load_N1_mw"])
    total_days = total_hours // 24
    if total_days < 365:
        raise RuntimeError("Time series too short for representative-day selection (need >=365 days).")

    target_days = max(1, min(int(cfg.representative_day_count), total_days))
    if target_days >= total_days:
        rep_day_indices = list(range(total_days * 24))
        rep_day_weights = [1.0] * len(rep_day_indices)
        rep_day_starts = [day_idx * 24 for day_idx in range(total_days)]
        return rep_day_indices, rep_day_weights, rep_day_starts, float(total_days)

    feature_rows = _daily_feature_rows(ts, total_days)
    stratum_count = min(max(1, target_days), 6)
    stratum_boundaries = [
        round(idx * total_days / stratum_count)
        for idx in range(stratum_count + 1)
    ]
    strata = [
        list(range(stratum_boundaries[idx], stratum_boundaries[idx + 1]))
        for idx in range(stratum_count)
        if stratum_boundaries[idx] < stratum_boundaries[idx + 1]
    ]
    quotas = _allocate_quotas([len(stratum) for stratum in strata], target_days)

    selected_days: List[Tuple[int, int]] = []
    for stratum, quota in zip(strata, quotas):
        if quota <= 0:
            continue
        local_rows = [feature_rows[day_idx] for day_idx in stratum]
        medoid_indices, cluster_weights = _select_stratum_medoids(local_rows, quota)
        for local_idx, cluster_weight in zip(medoid_indices, cluster_weights):
            selected_days.append((stratum[local_idx], cluster_weight))

    if len(selected_days) != target_days:
        chosen = {day_idx for day_idx, _ in selected_days}
        remaining_days = [day_idx for day_idx in range(total_days) if day_idx not in chosen]
        feature_dist = [
            (day_idx, min(_squared_distance(feature_rows[day_idx], feature_rows[sel_day]) for sel_day in chosen) if chosen else 0.0)
            for day_idx in remaining_days
        ]
        feature_dist.sort(key=lambda item: item[1], reverse=True)
        for day_idx, _ in feature_dist[: target_days - len(selected_days)]:
            selected_days.append((day_idx, 1))

    selected_days.sort(key=lambda item: item[0])
    rep_day_indices: List[int] = []
    rep_day_weights: List[float] = []
    rep_day_starts: List[int] = []
    for day_idx, cluster_weight in selected_days:
        start_hour = day_idx * 24
        rep_day_starts.append(len(rep_day_indices))
        for h in range(start_hour, start_hour + 24):
            rep_day_indices.append(h)
            rep_day_weights.append(float(cluster_weight))
    return rep_day_indices, rep_day_weights, rep_day_starts, float(total_days)


def make_hourly_profiles_from_indices(
    sc: Scenario,
    cfg: Config,
    indices: List[int],
    weights: List[float],
    rep_day_starts: List[int],
) -> Dict[str, List[float]]:
    T = len(indices)
    it, it_demand, pheat, dhcap, pflex, p_it = [], [], [], [], [], []
    t_amb_c, t_dh_supply_c, cop_t = [], [], []
    cf = [sc.carbon_factor] * T
    pco2 = [sc.carbon_price] * T

    node_assignment, _, _, _, _ = _ensure_network_defaults(cfg)
    dc_node = node_assignment.get(sc.location, cfg.network_nodes[0])

    ts = load_network_timeseries(cfg, T=max(indices) + 1)
    node_loads = {
        "N1": [cfg.network_load_scale * ts["load_N1_mw"][i] for i in indices],
        "N2": [cfg.network_load_scale * ts["load_N2_mw"][i] for i in indices],
        "N3": [cfg.network_load_scale * ts["load_N3_mw"][i] for i in indices],
    }
    ren_avail = {
        "N1": [cfg.network_ren_scale * ts["ren_avail_N1_mw"][i] for i in indices],
        "N2": [cfg.network_ren_scale * ts["ren_avail_N2_mw"][i] for i in indices],
        "N3": [cfg.network_ren_scale * ts["ren_avail_N3_mw"][i] for i in indices],
    }
    load_total = [node_loads["N1"][i] + node_loads["N2"][i] + node_loads["N3"][i] for i in range(T)]
    max_load_total = max(load_total) if load_total else 1.0
    fuel_price = [ts["fuel_eur_per_mwh_fuel"][i] for i in indices]
    co2_price = ts.get("co2_eur_per_t", None)
    if co2_price is not None:
        co2_price = [float(v) * float(cfg.co2_timeseries_scale) for v in co2_price]
    amb_series = ts.get("amb_temp_c", None)
    dh_series = ts.get("dh_supply_temp_c", None)

    def dow_from_hour(h: int) -> int:
        return (h // 24) % 7

    for pos, t_abs in enumerate(indices):
        hour = t_abs % 24
        dow = dow_from_hour(t_abs)
        weekend = (dow in [5, 6])

        if sc.dc_type == "hyperscale":
            it_t = sc.base_it_mw * (0.98 + 0.04 * math.sin(2 * math.pi * hour / 24))
        elif sc.dc_type == "colocation":
            it_t = sc.base_it_mw * (0.88 + 0.22 * math.sin(2 * math.pi * (hour - 6) / 24))
        elif sc.dc_type == "ai":
            it_t = sc.base_it_mw * (0.97 + 0.07 * (1 if hour in [10, 11, 12, 13, 14] else 0))
        elif sc.dc_type == "streaming":
            it_t = sc.base_it_mw * (0.80 + 0.35 * math.sin(2 * math.pi * (hour - 16) / 24) + 0.10)
        else:
            it_t = sc.base_it_mw

        if weekend and sc.dc_type in ["colocation", "streaming"]:
            it_t *= 0.95
        it_t = max(0.5 * sc.base_it_mw, it_t)
        it.append(it_t)

        if amb_series is not None:
            t_amb = float(amb_series[t_abs])
        else:
            if sc.location in ["Norway", "Sweden", "Denmark", "Finland", "Iceland"]:
                t_mean, t_amp = 4.0, 6.0
            elif sc.location in ["United Kingdom"]:
                t_mean, t_amp = 9.0, 5.0
            else:
                t_mean, t_amp = 10.0, 6.0
            t_amb = t_mean + t_amp * math.sin(2 * math.pi * (hour - 15) / 24) + random.uniform(-1.0, 1.0)
        t_amb_c.append(t_amb)

        if dh_series is not None:
            t_dh = float(dh_series[t_abs])
        else:
            t_dh = cfg.t_dh_supply_ref_c + 2.0 * math.sin(2 * math.pi * (hour - 10) / 24)
        t_dh_supply_c.append(t_dh)

        lift = t_dh - t_amb
        bins = cfg.lift_bins_c
        cops = cfg.cop_by_bin
        if len(cops) != len(bins) - 1:
            raise ValueError("cop_by_bin must have length len(lift_bins_c)-1")
        idx = None
        for i in range(len(cops)):
            if bins[i] <= lift < bins[i + 1]:
                idx = i
                break
        if idx is None:
            idx = 0 if lift < bins[0] else (len(cops) - 1)
        cop = max(0.1, float(cops[idx]))
        cop_t.append(cop)

        peak = 1.0 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0.6
        if weekend:
            peak *= 0.92
        cold_factor = max(0.0, min(1.0, (15.0 - t_amb) / 25.0))
        dh = sc.heat_demand_level * (0.30 + 0.70 * cold_factor) * peak
        dhcap.append(max(0.0, min(sc.heat_demand_level, dh)))

        pheat_val = sc.heat_price_level * (0.70 + 0.60 * cold_factor) * (0.85 + 0.30 * peak)
        pheat.append(max(0.0, pheat_val))

        load_norm = load_total[pos] / max(1e-9, max_load_total)
        flex_tight = 0.7 + 0.6 * min(1.0, max(0.0, load_norm))
        base_f = sc.flex_price_level * flex_tight * (0.95 + 0.25 * (1 if hour in [7, 8, 9, 18, 19] else 0))
        spike_f = random.uniform(0.0, sc.flex_price_level) if random.random() < 0.10 else 0.0
        pflex.append(max(0.0, base_f + spike_f))

        dem_base = it_t * cfg.it_demand_scale
        dem_shape = 1.0 + 0.05 * math.sin(2 * math.pi * (hour - 8) / 24)
        dem_noise = random.uniform(-0.05, 0.05)
        it_demand_t = max(0.0, dem_base * (dem_shape + dem_noise))
        it_demand.append(it_demand_t)

        it_price = cfg.it_price_multiplier * sc.el_price_level + cfg.it_price_daily_amp * math.sin(2 * math.pi * (hour - 12) / 24)
        it_price += random.uniform(-cfg.it_price_vol, cfg.it_price_vol)
        p_it.append(max(0.0, it_price))

    p_el = [0.0] * T
    if cfg.sipue_credit_normalization == "ref_price":
        p_el = [float(cfg.sipue_ref_price_eur_per_mwh)] * T
    cong = [0.0] * T

    return {
        "it_base": it,
        "it_demand": it_demand,
        "p_el": p_el,
        "p_heat": pheat,
        "dh_cap": dhcap,
        "p_flex": pflex,
        "p_it": p_it,
        "t_amb_c": t_amb_c,
        "t_dh_supply_c": t_dh_supply_c,
        "cop_t": cop_t,
        "node_loads": node_loads,
        "dc_node": dc_node,
        "cong": cong,
        "grid_cap": None,
        "carbon_factor": cf,
        "p_co2": ([co2_price[i] for i in indices] if co2_price is not None else pco2),
        "ren_avail": ren_avail,
        "fuel_price": fuel_price,
        "w": weights,
        "rep_day_starts": rep_day_starts,
        "source_hours": indices[:],
    }


def make_hourly_profiles(sc: Scenario, T: int, cfg: Config) -> Dict[str, List[float]]:
    it, it_demand, pheat, dhcap, pflex, p_it = [], [], [], [], [], []
    t_amb_c, t_dh_supply_c, cop_t = [], [], []
    cf = [sc.carbon_factor] * T
    pco2 = [sc.carbon_price] * T

    node_assignment, _, _, _, _ = _ensure_network_defaults(cfg)
    dc_node = node_assignment.get(sc.location, cfg.network_nodes[0])

    ts = load_network_timeseries(cfg, T=T)
    node_loads = {
        "N1": [cfg.network_load_scale * v for v in ts["load_N1_mw"][:T]],
        "N2": [cfg.network_load_scale * v for v in ts["load_N2_mw"][:T]],
        "N3": [cfg.network_load_scale * v for v in ts["load_N3_mw"][:T]],
    }
    ren_avail = {
        "N1": [cfg.network_ren_scale * v for v in ts["ren_avail_N1_mw"][:T]],
        "N2": [cfg.network_ren_scale * v for v in ts["ren_avail_N2_mw"][:T]],
        "N3": [cfg.network_ren_scale * v for v in ts["ren_avail_N3_mw"][:T]],
    }
    load_total = [node_loads["N1"][i] + node_loads["N2"][i] + node_loads["N3"][i] for i in range(T)]
    max_load_total = max(load_total) if load_total else 1.0
    fuel_price = ts["fuel_eur_per_mwh_fuel"][:T]
    co2_price = ts.get("co2_eur_per_t", None)
    if co2_price is not None:
        co2_price = [float(v) * float(cfg.co2_timeseries_scale) for v in co2_price]
    amb_series = ts.get("amb_temp_c", None)
    dh_series = ts.get("dh_supply_temp_c", None)

    for t in range(T):
        hour = t % 24
        dow = (t // 24) % 7
        weekend = (dow in [5, 6])

        if sc.dc_type == "hyperscale":
            it_t = sc.base_it_mw * (0.98 + 0.04 * math.sin(2 * math.pi * hour / 24))
        elif sc.dc_type == "colocation":
            it_t = sc.base_it_mw * (0.88 + 0.22 * math.sin(2 * math.pi * (hour - 6) / 24))
        elif sc.dc_type == "ai":
            it_t = sc.base_it_mw * (0.97 + 0.07 * (1 if hour in [10, 11, 12, 13, 14] else 0))
        elif sc.dc_type == "streaming":
            it_t = sc.base_it_mw * (0.80 + 0.35 * math.sin(2 * math.pi * (hour - 16) / 24) + 0.10)
        else:
            it_t = sc.base_it_mw

        if weekend and sc.dc_type in ["colocation", "streaming"]:
            it_t *= 0.95
        it_t = max(0.5 * sc.base_it_mw, it_t)
        it.append(it_t)

        if amb_series is not None:
            t_amb = float(amb_series[t])
        else:
            if sc.location in ["Norway", "Sweden", "Denmark", "Finland", "Iceland"]:
                t_mean, t_amp = 4.0, 6.0
            elif sc.location in ["United Kingdom"]:
                t_mean, t_amp = 9.0, 5.0
            else:
                t_mean, t_amp = 10.0, 6.0
            t_amb = t_mean + t_amp * math.sin(2 * math.pi * (hour - 15) / 24) + random.uniform(-1.0, 1.0)
        t_amb_c.append(t_amb)

        if dh_series is not None:
            t_dh = float(dh_series[t])
        else:
            t_dh = cfg.t_dh_supply_ref_c + 2.0 * math.sin(2 * math.pi * (hour - 10) / 24)
        t_dh_supply_c.append(t_dh)

        lift = t_dh - t_amb
        bins = cfg.lift_bins_c
        cops = cfg.cop_by_bin
        if len(cops) != len(bins) - 1:
            raise ValueError("cop_by_bin must have length len(lift_bins_c)-1")
        idx = None
        for i in range(len(cops)):
            if bins[i] <= lift < bins[i + 1]:
                idx = i
                break
        if idx is None:
            idx = 0 if lift < bins[0] else (len(cops) - 1)
        cop = max(0.1, float(cops[idx]))
        cop_t.append(cop)

        peak = 1.0 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0.6
        if weekend:
            peak *= 0.92
        cold_factor = max(0.0, min(1.0, (15.0 - t_amb) / 25.0))  # ~0 in warm, ~1 in cold
        dh = sc.heat_demand_level * (0.30 + 0.70 * cold_factor) * peak
        dhcap.append(max(0.0, min(sc.heat_demand_level, dh)))

        pheat_val = sc.heat_price_level * (0.70 + 0.60 * cold_factor) * (0.85 + 0.30 * peak)
        pheat.append(max(0.0, pheat_val))

        load_norm = load_total[t] / max(1e-9, max_load_total)
        flex_tight = 0.7 + 0.6 * min(1.0, max(0.0, load_norm))
        base_f = sc.flex_price_level * flex_tight * (0.95 + 0.25 * (1 if hour in [7, 8, 9, 18, 19] else 0))
        spike_f = random.uniform(0.0, sc.flex_price_level) if random.random() < 0.10 else 0.0
        pflex.append(max(0.0, base_f + spike_f))

        dem_base = it_t * cfg.it_demand_scale
        dem_shape = 1.0 + 0.05 * math.sin(2 * math.pi * (hour - 8) / 24)
        dem_noise = random.uniform(-0.05, 0.05)
        it_demand_t = max(0.0, dem_base * (dem_shape + dem_noise))
        it_demand.append(it_demand_t)

        it_price = cfg.it_price_multiplier * sc.el_price_level + cfg.it_price_daily_amp * math.sin(2 * math.pi * (hour - 12) / 24)
        it_price += random.uniform(-cfg.it_price_vol, cfg.it_price_vol)
        p_it.append(max(0.0, it_price))

    p_el = [0.0] * T
    if cfg.sipue_credit_normalization == "ref_price":
        p_el = [float(cfg.sipue_ref_price_eur_per_mwh)] * T
    cong = [0.0] * T

    return {
        "it_base": it,
        "it_demand": it_demand,
        "p_el": p_el,
        "p_heat": pheat,
        "dh_cap": dhcap,  # interpreted as DH market size (max accepted quantity) in the endogenous value model
        "p_flex": pflex,
        "p_it": p_it,
        "t_amb_c": t_amb_c,
        "t_dh_supply_c": t_dh_supply_c,
        "cop_t": cop_t,
        "node_loads": node_loads,
        "dc_node": dc_node,
        "cong": cong,
        "grid_cap": None,
        "carbon_factor": cf,
        "p_co2": (co2_price[:T] if co2_price is not None else pco2),
        "ren_avail": ren_avail,
        "fuel_price": fuel_price,
        "source_hours": list(range(T)),
    }


def make_overhead_alpha_base(sc: Scenario, T: int, cfg: Config, night_drop_override: Optional[float] = None) -> List[float]:
    """
    Base overhead alpha_base[t] such that COOL[t] = alpha_eff[t] * IT[t],
    where alpha_eff[t] = alpha_base[t] * (1 - eff_reduction).

    Here alpha_base[t] captures climate/time effects as free cooling proxy.
    """
    base_alpha = max(0.0, sc.base_pue - 1.0)
    alpha = [base_alpha] * T
    if not cfg.enable_time_varying_overhead:
        return alpha

    nordic = sc.location in ["Norway", "Sweden", "Finland", "Denmark", "Iceland"]
    if not nordic:
        return alpha

    night_drop = cfg.nordic_night_overhead_drop if night_drop_override is None else night_drop_override
    night_drop = max(0.0, min(0.90, night_drop))

    for t in range(T):
        h = t % 24
        if h in [0, 1, 2, 3, 4, 5, 22, 23]:
            alpha[t] = base_alpha * (1.0 - night_drop)
        elif h in [12, 13, 14, 15, 16, 17]:
            alpha[t] = base_alpha * (1.0 + cfg.day_overhead_bump)

    return alpha


def apply_services_mode(sc: Scenario, prof: Dict[str, List[float]], cfg: Config) -> Dict[str, List[float]]:
    prof2 = dict(prof)
    T = len(prof["it_base"])
    source_hours = prof.get("source_hours", list(range(T)))

    def _smooth_multipliers(mult: List[float], ramp_hours: int) -> List[float]:
        if ramp_hours <= 0:
            return mult[:]
        n = len(mult)
        out = mult[:]
        for i in range(1, n):
            if mult[i] != mult[i - 1]:
                a = mult[i - 1]
                b = mult[i]
                for k in range(-ramp_hours, ramp_hours + 1):
                    idx = i + k
                    if 0 <= idx < n:
                        frac = (k + ramp_hours) / max(1, 2 * ramp_hours)
                        out[idx] = a + (b - a) * frac
        return out

    prof2["allow_heat"] = True
    prof2["allow_flex"] = True
    prof2["allow_battery"] = True
    prof2["force_it_flex"] = (sc.dc_type != "streaming")

    p_heat = prof2["p_heat"][:]
    dh_cap = prof2["dh_cap"][:]
    p_flex = prof2["p_flex"][:]

    if len(source_hours) == T:
        week_states = {}
        for abs_hour in source_hours:
            week_idx = abs_hour // 168
            if week_idx in week_states:
                continue
            rng = random.Random(((cfg.seed + 1) * 1_000_003) + (sc.sid * 10_009) + week_idx)
            low_heat = rng.random() < cfg.p_low_heat_week
            week_states[week_idx] = {
                "dh_off": rng.random() < cfg.p_dh_off_week,
                "low_flex": rng.random() < cfg.p_low_flex_week,
                "low_heat": low_heat,
                "heat_start": rng.randint(0, 120) if low_heat else 0,
            }

        for idx, abs_hour in enumerate(source_hours):
            state = week_states[abs_hour // 168]
            hour_of_day = abs_hour % 24
            hour_of_week = abs_hour % 168

            if state["dh_off"]:
                dh_cap[idx] *= 0.60 if hour_of_day in [7, 8, 9, 18, 19] else 0.30

            if state["low_heat"] and state["heat_start"] <= hour_of_week < state["heat_start"] + 48:
                p_heat[idx] *= 0.10

            if state["low_flex"] and ((hour_of_week // 12) % 2) == 0:
                p_flex[idx] *= 0.0
    else:
        hours_per_week = 168
        n_weeks = max(1, (T + hours_per_week - 1) // hours_per_week)

        for w in range(n_weeks):
            w_start = w * hours_per_week
            w_end = min(T, w_start + hours_per_week)
            w_len = w_end - w_start

            dh_off = random.random() < cfg.p_dh_off_week
            low_flex = random.random() < cfg.p_low_flex_week
            low_heat = random.random() < cfg.p_low_heat_week

            if dh_off:
                mult = [1.0] * w_len
                for i in range(w_len):
                    h = (w_start + i) % 24
                    mult[i] = 0.60 if h in [7, 8, 9, 18, 19] else 0.30
                ramped = _smooth_multipliers(mult, ramp_hours=4)
                for i, m in enumerate(ramped):
                    dh_cap[w_start + i] *= m

            if low_heat and w_len >= 48:
                start = random.randint(w_start, w_end - 48)
                mult = [1.0] * w_len
                for i in range(start - w_start, start - w_start + 48):
                    mult[i] *= 0.10
                ramped = _smooth_multipliers(mult, ramp_hours=4)
                for i, m in enumerate(ramped):
                    p_heat[w_start + i] *= m

            if low_flex:
                mult = [1.0] * w_len
                for i in range(w_len):
                    if ((i // 12) % 2) == 0:
                        mult[i] = 0.0
                ramped = _smooth_multipliers(mult, ramp_hours=2)
                for i, m in enumerate(ramped):
                    p_flex[w_start + i] *= m

    prof2["p_heat"] = p_heat
    prof2["dh_cap"] = dh_cap
    prof2["p_flex"] = p_flex
    prof2["alpha_base"] = make_overhead_alpha_base(sc, T, cfg)

    return prof2
