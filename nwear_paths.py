"""Central path configuration for the N-WEAR code release.

Every script reads its default locations from here, so a new machine only needs
four environment variables (see ``setup_env.sh``):

    NWEAR_DATA        root of the WEAR data and the converted event clips
    NWEAR_OUTPUT      root for checkpoints, caches and result files
    WEAR_REPO         checkout of https://github.com/mariusbock/wear with
                      ``sima/wear_patches`` applied (SIM A baselines only)
    NEUROBENCH_ROOT   optional NeuroBench 2.3.0 clone (else the pip package)

Expected layout under ``NWEAR_DATA`` (names kept from the original experiments):

    wear_dataset_annotations/60fps/wear_split_{1..18}.json
    wear_dataset/processed/inertial_features/50hz/120_frames_60_stride/sbj_{0..17}.npy
    wear_dataset/processed/i3d_features/60fps/120_frames_60_stride/sbj_{0..17}.npy
    wear_dataset/raw/inertial/50hz/sbj_{0..17}.csv
    sbj_{0..17}_events_dvs128[_new]/<activity>_part_<k>_from_<t>/<...>_NNN_dvs128.h5
    SuperSloMo39.ckpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_ROOT = _env_path("NWEAR_DATA", REPO_ROOT / "data")
OUTPUT_ROOT = _env_path("NWEAR_OUTPUT", REPO_ROOT / "outputs")
WEAR_REPO = _env_path("WEAR_REPO", REPO_ROOT / "third_party" / "wear")
NEUROBENCH_ROOT = _env_path("NEUROBENCH_ROOT", REPO_ROOT / "third_party" / "neurobench")

# WEAR inputs
ANN_DIR = DATA_ROOT / "wear_dataset_annotations" / "60fps"
IMU_FEAT_DIR = DATA_ROOT / "wear_dataset" / "processed" / "inertial_features" / "50hz" / "120_frames_60_stride"
I3D_FEAT_DIR = DATA_ROOT / "wear_dataset" / "processed" / "i3d_features" / "60fps" / "120_frames_60_stride"
RAW_INERTIAL_DIR = DATA_ROOT / "wear_dataset" / "raw" / "inertial" / "50hz"

# Code folders (scripts import sibling modules by name)
EVENT_SNN_DIR = REPO_ROOT / "event_snn"
IMU_SNN_DIR = REPO_ROOT / "imu_snn"
SIMA_DIR = REPO_ROOT / "sima"
SIMB_DIR = REPO_ROOT / "simb"

# Outputs of each stage
EVENT_RUNS = OUTPUT_ROOT / "event_snn" / "loso54"
EVENT_TENSOR_CACHE = OUTPUT_ROOT / "event_snn" / "tensor_cache"
IMU_RUNS = OUTPUT_ROOT / "imu_snn" / "loso54"
SIMA_BF1 = OUTPUT_ROOT / "sima" / "bf1_ad_camera"
SIMA_BF2 = OUTPUT_ROOT / "sima" / "bf2_imu_camera"
SIMA_ENERGY = OUTPUT_ROOT / "sima" / "energy"
SIMB_HEADS = OUTPUT_ROOT / "simb" / "heads"
SIMB_CACHE = OUTPUT_ROOT / "simb" / "feature_cache"
RESULTS_DIR = OUTPUT_ROOT / "results"


def add_to_path(*dirs: Path) -> None:
    """Prepend code folders to ``sys.path`` so sibling modules import by name."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
