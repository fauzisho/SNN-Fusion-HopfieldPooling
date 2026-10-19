#!/usr/bin/env python3
"""Merge BF1/BF2 Hopfield duty-cycle summaries once both jobs finish."""
from __future__ import annotations

import json
from pathlib import Path
import nwear_paths as P

ROOT = P.OUTPUT_ROOT / "sima"
OUT = ROOT / "duty_cycle_bf1_vs_bf2_hopfield_loso54.json"


def load(pairing: str) -> dict:
    p = ROOT / f"duty_cycle_{pairing}_hopfield_loso54.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def main() -> None:
    payload = {
        "protocol": "same trigger k=4 entropy>0.05 or mean maxp<0.90; 18-class activity-frame; 54 LOSO; dual=Hopfield",
        "bf1": load("bf1"),
        "bf2": load("bf2"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT)
    for pairing in ("bf1", "bf2"):
        for head in ("single", "double"):
            m = payload[pairing]["heads"][head]["means_54"]
            print(
                f"{pairing} {head:6s}  d {m['d_windows']['mean']:.1f}%  "
                f"IMU {m['imu_F1']['mean']:.2f}  dyn {m['dynamic_F1']['mean']:.2f}  "
                f"AO {m['always_on_F1']['mean']:.2f}  lift {m['lift_F1']['mean']:+.2f}  "
                f"ret {m['retention_F1']['mean']:.3f}"
            )


if __name__ == "__main__":
    main()
