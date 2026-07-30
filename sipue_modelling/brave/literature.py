from __future__ import annotations

import csv
import os
from typing import Dict, List


def literature_matrix() -> List[Dict]:
    """Comparison matrix used to delimit the siPUE contribution."""
    return [
        _row(
            "pue", "ISO/IEC 30134-2:2026", "facility energy / IT energy",
            facility=True, reuse=False, carbon=False, grid=False, economics=False, delay=False,
            source="https://www.iso.org/standard/30134-2",
            distinction="Standard facility-efficiency benchmark; no external system credit.",
        ),
        _row(
            "ERF", "ISO/IEC 30134-6:2021", "reused energy / data-centre energy",
            facility=True, reuse=True, carbon=False, grid=False, economics=False, delay=False,
            source="https://www.iso.org/standard/71717.html",
            distinction="Measures reused energy but not net congestion, peak, reserve or carbon-damage effects.",
        ),
        _row(
            "cue", "The Green Grid WP#32", "carbon emissions / IT energy",
            facility=True, reuse=False, carbon=True, grid=False, economics=False, delay=False,
            source="https://www.thegreengrid.org/en/resources/library-and-tools/241-WP%2332---Carbon-Usage-Effectiveness-%28CUE%29%3A-A-Green-Grid-Data-Center-Sustainability-Metric",
            distinction="Carbon-intensity indicator; does not aggregate broader local-system burden.",
        ),
        _row(
            "Energy-system integration study", "Energy 2023 Danish case study",
            "system cost and emissions from flexible cooling and waste heat",
            facility=False, reuse=True, carbon=True, grid=True, economics=True, delay=False,
            source="https://doi.org/10.1016/j.energy.2023.128112",
            distinction="Quantifies system outcomes but does not define a PUE-compatible KPI.",
        ),
        _row(
            "Carbon-aware workload scheduling", "CarbonScaler 2023",
            "workload elasticity optimized against grid carbon",
            facility=False, reuse=False, carbon=True, grid=True, economics=False, delay=False,
            source="https://doi.org/10.1145/3626788",
            distinction="Operational carbon scheduling rather than a multi-service lifecycle metric.",
        ),
        _row(
            "AI data-centre flexibility planning", "ACM e-Energy/Sustainability Week 2026",
            "deferrable and geographically shiftable AI workload in grid planning",
            facility=False, reuse=False, carbon=True, grid=True, economics=True, delay=False,
            source="https://doi.org/10.1145/3765611.3815593",
            distinction="Grid-planning formulation; no delay-aware PUE extension.",
        ),
        _row(
            "sipue", "This study", "delay-adjusted independent system benefit as energy-equivalent facility burden",
            facility=True, reuse=True, carbon=True, grid=True, economics=True, delay=True,
            source="",
            distinction="Proposed model-based indicator; novelty claim remains conditional on structured review.",
        ),
    ]


def _row(
    metric: str,
    reference: str,
    scope: str,
    *,
    facility: bool,
    reuse: bool,
    carbon: bool,
    grid: bool,
    economics: bool,
    delay: bool,
    source: str,
    distinction: str,
) -> Dict:
    return {
        "metric_or_approach": metric,
        "reference": reference,
        "primary_scope": scope,
        "facility_energy": facility,
        "energy_reuse": reuse,
        "carbon": carbon,
        "grid_congestion_or_flexibility": grid,
        "economic_system_effects": economics,
        "construction_delay": delay,
        "source": source,
        "distinction_from_sipue": distinction,
    }


def export_literature_matrix(path: str, run_id: str = "") -> None:
    rows = literature_matrix()
    if run_id:
        for row in rows:
            row["run_id"] = run_id
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
