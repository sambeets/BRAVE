from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


_BANNED_OUTPUT_TOKENS = ("operator", "settlement", "unitprofit", "profitpct")


def _write_csv(rows: Iterable[Dict], path: str) -> None:
    data = list(rows)
    if not data:
        return
    fields = sorted({key for row in data for key in row})
    banned = [field for field in fields if any(token in field.lower() for token in _BANNED_OUTPUT_TOKENS)]
    if banned:
        raise ValueError(f"Operator-facing fields are not permitted in canonical output: {banned}")
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def export_scenarios_csv(scenarios, path: str) -> None:
    _write_csv((dict(sc.__dict__) for sc in scenarios), path)


def export_results_csv(rows: List[Dict], path: str) -> None:
    _write_csv(rows, path)


def summarize_interpretation(rows: List[Dict], cfg=None) -> None:
    if not rows:
        print("No results available.")
        return
    avg_value = sum(float(row.get("MarketValueNPVEUR_robust", 0.0)) for row in rows) / len(rows)
    avg_benefit = sum(float(row.get("SystemBenefitPerMWhIT_robust", 0.0)) for row in rows) / len(rows)
    carbon_ok = sum(
        float(row.get("EmissionsTCO2_robust", float("inf")))
        <= float(row.get("CarbonCapTCO2", -float("inf"))) + 1e-6
        for row in rows
    )
    print(f"Scenarios: {len(rows)}")
    print(f"Average robust lifecycle market value: EUR {avg_value:,.0f}")
    print(f"Average delay-adjusted system benefit: EUR {avg_benefit:,.2f}/MWh IT")
    print(f"Physical carbon cap satisfied: {carbon_ok}/{len(rows)}")
