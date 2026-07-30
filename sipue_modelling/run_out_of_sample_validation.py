from __future__ import annotations

"""Design on canonical scenarios and evaluate synthetic out-of-sample draws.

Writes output/validation/out-of-sample/out_of_sample_validation.csv and summary.
"""

from brave.config import Config
from brave.runners import run_out_of_sample_validation


if __name__ == "__main__":
    cfg = Config()
    cfg.export_csv = True
    cfg.export_plots = False
    cfg.export_regret_landscape = False
    cfg.out_of_sample_case_count = 40
    run_out_of_sample_validation(cfg)
