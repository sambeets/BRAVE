"""Generate the creative SIPUE figure package from existing CSV outputs."""

from __future__ import annotations

import argparse
from dataclasses import replace

from brave.config import Config
from brave.plotting import export_creative_figure_package_from_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default=None, help="Primary plot format, e.g. pdf or png.")
    parser.add_argument(
        "--extra-format",
        action="append",
        default=[],
        help="Additional plot format. Repeat for multiple formats.",
    )
    args = parser.parse_args()
    cfg = Config(enforce_locked_calibration=False)
    if args.format:
        cfg = replace(cfg, plot_format=args.format)
    if args.extra_format:
        cfg = replace(cfg, plot_extra_formats=tuple(args.extra_format))
    export_creative_figure_package_from_csv(cfg)


if __name__ == "__main__":
    main()
