from __future__ import annotations

import csv
import math
import os
from typing import Dict, Iterable, List, Sequence

from brave.metrics import compute_sipue


_TOL = 1e-8


def compute_ere(E_FAC: float, E_IT: float, heat_reuse_mwhth: float) -> float:
    return max(0.0, E_FAC - max(0.0, heat_reuse_mwhth)) / max(1e-9, E_IT)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_validation_summary(rows: Sequence[Dict], *args, **kwargs) -> List[Dict]:
    """Property validation for a constructed system-burden metric.

    Correlation with system benefit per MWh IT is intentionally omitted: that
    target is part of the siPUE definition and is not independent evidence.
    """
    checks: List[Dict] = []
    suffixes = (
        "no_services",
        "fair_services_disabled",
        "baseline",
        "robust",
    )
    for suffix in suffixes:
        available = [row for row in rows if _finite(row.get(f"sipue_{suffix}"))]
        if not available:
            continue
        checks.append({
            "test": "finite_nonnegative",
            "case": suffix,
            "n": len(available),
            "passed": all(float(row[f"sipue_{suffix}"]) >= -_TOL for row in available),
            "note": "siPUE is finite and nonnegative",
        })

    identity_rows = [
        row for row in rows
        if _finite(row.get("pue_no_services")) and _finite(row.get("sipue_no_services"))
    ]
    checks.append({
        "test": "no_services_identity",
        "case": "no_services",
        "n": len(identity_rows),
        "passed": all(
            abs(float(row["pue_no_services"]) - float(row["sipue_no_services"])) <= _TOL
            for row in identity_rows
        ),
        "note": "siPUE equals PUE when system benefit is zero",
    })

    delay_rows = [
        row for row in rows
        if _finite(row.get("sipue_robust")) and _finite(row.get("steady_sipue_robust"))
    ]
    checks.append({
        "test": "delay_monotonicity",
        "case": "robust",
        "n": len(delay_rows),
        "passed": all(
            float(row["sipue_robust"]) + _TOL >= float(row["steady_sipue_robust"])
            if float(row.get("system_benefit_eur_robust", 0.0)) >= 0.0 else True
            for row in delay_rows
        ),
        "note": "construction delay cannot strengthen a positive system credit",
    })

    carbon_rows = [
        row for row in rows
        if _finite(row.get("carbon_market_cost_eur_robust"))
        and _finite(row.get("carbon_damage_eur_robust"))
    ]
    checks.append({
        "test": "carbon_ledger_separation",
        "case": "all",
        "n": len(carbon_rows),
        "passed": len(carbon_rows) == len(rows),
        "note": "ETS cost and carbon damage are exported as separate ledger terms",
    })

    scenario_ids = {row.get("scenario_id") for row in rows}
    archetypes = {row.get("archetype_key") for row in rows if row.get("archetype_key") is not None}
    counts = {
        archetype: sum(row.get("archetype_key") == archetype for row in rows)
        for archetype in archetypes
    }
    checks.append({
        "test": "canonical_scenario_count",
        "case": "all",
        "n": len(rows),
        "passed": len(scenario_ids) == 10 and (not archetypes or (len(archetypes) == 4 and all(value == 10 for value in counts.values()))),
        "note": "canonical analysis contains ten shared states for each of four archetypes",
    })
    base, _ = compute_sipue(120.0, 100.0, 1000.0, 100.0)
    scaled, _ = compute_sipue(240.0, 200.0, 2000.0, 100.0)
    zero_benefit, _ = compute_sipue(120.0, 100.0, 0.0, 100.0)
    delayed, _ = compute_sipue(120.0, 100.0, 1000.0, 100.0, benefit_availability_factor=0.5)
    low_ref, _ = compute_sipue(120.0, 100.0, 1000.0, 50.0)
    high_ref, _ = compute_sipue(120.0, 100.0, 1000.0, 150.0)
    checks.extend([
        {
            "test": "scale_normalization",
            "case": "synthetic",
            "n": 2,
            "passed": abs(base - scaled) <= _TOL,
            "note": "proportional facility scaling leaves siPUE unchanged",
        },
        {
            "test": "benefit_monotonicity",
            "case": "synthetic",
            "n": 2,
            "passed": base <= zero_benefit + _TOL,
            "note": "positive independent benefit lowers siPUE relative to zero benefit",
        },
        {
            "test": "delay_behavior",
            "case": "synthetic",
            "n": 2,
            "passed": delayed + _TOL >= base,
            "note": "lower lifecycle availability weakens a positive credit",
        },
        {
            "test": "reference_price_monotonicity",
            "case": "synthetic",
            "n": 3,
            "passed": low_ref <= base + _TOL and base <= high_ref + _TOL,
            "note": "reference-price monotonicity has the expected direction",
        },
    ])
    return checks


def build_ablation_summary(rows: Sequence[Dict], *args, **kwargs) -> List[Dict]:
    return []


def add_derived_validation_metrics(rows: List[Dict], *args, **kwargs) -> None:
    return None


def _write_csv(rows: Iterable[Dict], path: str) -> None:
    data = list(rows)
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(data[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def export_validation_tables(rows: List[Dict], cfg) -> None:
    _write_csv(build_validation_summary(rows), cfg.validation_summary_csv)


def export_out_of_sample_tables(rows: List[Dict], cfg) -> None:
    _write_csv(rows, cfg.validation_oos_csv)
