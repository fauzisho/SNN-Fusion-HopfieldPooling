"""WEAR fine-grained activity labels for DVS240 v2e H5 folders."""

from __future__ import annotations

import re

FINE_ACTIVITY_CLASSES = [
    "jogging",
    "jogging (rotating arms)",
    "jogging (skipping)",
    "jogging (sidesteps)",
    "jogging (butt-kicks)",
    "stretching (triceps)",
    "stretching (lunging)",
    "stretching (shoulders)",
    "stretching (hamstrings)",
    "stretching (lumbar rotation)",
    "push-ups",
    "push-ups (complex)",
    "sit-ups",
    "sit-ups (complex)",
    "burpees",
    "lunges",
    "lunges (complex)",
    "bench-dips",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(FINE_ACTIVITY_CLASSES)}
NUM_CLASSES = len(FINE_ACTIVITY_CLASSES)

FINE_TO_COARSE: dict[str, str] = {
    "jogging": "jogging",
    "jogging (rotating arms)": "jogging",
    "jogging (skipping)": "jogging",
    "jogging (sidesteps)": "jogging",
    "jogging (butt-kicks)": "jogging",
    "stretching (triceps)": "stretching",
    "stretching (lunging)": "stretching",
    "stretching (shoulders)": "stretching",
    "stretching (hamstrings)": "stretching",
    "stretching (lumbar rotation)": "stretching",
    "push-ups": "push-ups",
    "push-ups (complex)": "push-ups",
    "sit-ups": "sit-ups",
    "sit-ups (complex)": "sit-ups",
    "burpees": "burpees",
    "lunges": "lunges",
    "lunges (complex)": "lunges",
    "bench-dips": "bench-dips",
}

COARSE_ACTIVITY_CLASSES = [
    "jogging",
    "stretching",
    "push-ups",
    "sit-ups",
    "burpees",
    "lunges",
    "bench-dips",
]

COARSE_CLASS_TO_IDX = {name: idx for idx, name in enumerate(COARSE_ACTIVITY_CLASSES)}

FINE_IDX_TO_COARSE_IDX = tuple(
    COARSE_CLASS_TO_IDX[FINE_TO_COARSE[name]] for name in FINE_ACTIVITY_CLASSES
)

FOLDER_PREFIX_TO_CLASS: dict[str, str] = {
    "jogging": "jogging",
    "jogging_rotating_arms": "jogging (rotating arms)",
    "jogging_skipping": "jogging (skipping)",
    "jogging_sidesteps": "jogging (sidesteps)",
    "jogging_butt-kicks": "jogging (butt-kicks)",
    "stretching_triceps": "stretching (triceps)",
    "stretching_lunging": "stretching (lunging)",
    "stretching_shoulders": "stretching (shoulders)",
    "stretching_hamstrings": "stretching (hamstrings)",
    "stretching_lumbar_rotation": "stretching (lumbar rotation)",
    "push-ups": "push-ups",
    "push-ups_complex": "push-ups (complex)",
    "sit-ups": "sit-ups",
    "sit-ups_complex": "sit-ups (complex)",
    "burpees": "burpees",
    "lunges": "lunges",
    "lunges_complex": "lunges (complex)",
    "bench-dips": "bench-dips",
}

_ACTIVITY_PAREN_PREFIX = re.compile(r"^(.+)_\(([^)]+)\)$")


def normalize_activity_folder_key(activity_key: str) -> str:
    match = _ACTIVITY_PAREN_PREFIX.match(activity_key)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return activity_key


def label_from_h5_path(h5_path: str) -> int | None:
    folder_name = h5_path.split("/")[-2] if "/" in h5_path else h5_path
    if "_part_" not in folder_name:
        return None
    activity_key = normalize_activity_folder_key(folder_name.split("_part_")[0])
    class_name = FOLDER_PREFIX_TO_CLASS.get(activity_key)
    if class_name is None:
        return None
    return CLASS_TO_IDX.get(class_name)
