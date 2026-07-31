from __future__ import annotations

"""Out-of-sample check for the cost-regret vs siPUE-regret Stage-2 comparison.

Repeats brave.runners.run_sipue_objective_comparison end to end (both Stage-1
benchmarks, both Stage-2 objective modes) on freshly drawn uncertainty
scenarios instead of the ten canonical states, to test whether the paper's
"3 of 4 archetypes select an identical design" result survives on draws that
played no part in calibrating the model.

Writes output/validation/out-of-sample/sipue_objective_comparison_out_of_sample.csv
"""

from brave.config import Config
from brave.runners import run_sipue_objective_comparison_out_of_sample


if __name__ == "__main__":
    cfg = Config()
    cfg.export_csv = True
    cfg.export_plots = False
    run_sipue_objective_comparison_out_of_sample(cfg)
