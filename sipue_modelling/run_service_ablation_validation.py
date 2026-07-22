from __future__ import annotations

import argparse

from brave.config import Config
from brave.runners import run_service_ablation_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "smoke" if args.smoke else "annual"
    cfg = Config(
        T=24 if args.smoke else 8760,
        use_representative_days_annual=not args.smoke,
        enforce_locked_calibration=False,
        export_plots=False,
        canonical_output_dir=f"output/congested3/service-ablations-{tag}",
    )
    rows = run_service_ablation_validation(cfg)
    print(f"Exported {len(rows)} service-ablation rows.")


if __name__ == "__main__":
    main()
