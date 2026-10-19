#!/usr/bin/env python3
"""
Paper-style inertial pipeline wrapper.

Runs `pipeline_inertial.py` with settings that are closer to WEAR paper
comparison: include NULL class and report macro-F1.
"""
from __future__ import annotations

import sys

from pipeline_inertial import main as base_main


def main() -> None:
    extra = [
        "--include-null-class",
        "--postprocess-majority-sec", "10",
        "--compute-map",
        "--map-tiou", "0.3", "0.4", "0.5", "0.6", "0.7",
        "--exclude-null-in-map",
    ]
    base_main(sys.argv[1:] + extra)


if __name__ == "__main__":
    main()
