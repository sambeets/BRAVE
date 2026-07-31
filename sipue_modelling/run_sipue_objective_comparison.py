from __future__ import annotations

"""In-sample cost-regret vs siPUE-regret Stage-2 objective comparison.

Trains and evaluates both Stage-2 objectives on the same ten canonical
uncertainty states used throughout the paper (see paper.tex, siPUE-objective
comparison discussion, and run_sipue_objective_comparison_out_of_sample.py
for the out-of-sample counterpart).

Writes output/congested3/sipue_objective_comparison.csv
"""

from brave.config import Config
from brave.runners import run_sipue_objective_comparison


if __name__ == "__main__":
    cfg = Config()
    cfg.export_csv = True
    cfg.export_plots = False
    run_sipue_objective_comparison(cfg)
