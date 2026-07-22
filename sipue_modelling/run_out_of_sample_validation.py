from __future__ import annotations

from brave.config import Config
from brave.runners import run_out_of_sample_validation


if __name__ == "__main__":
    cfg = Config()
    cfg.export_csv = True
    cfg.export_plots = False
    cfg.export_regret_landscape = False
    run_out_of_sample_validation(cfg)
