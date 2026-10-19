#!/usr/bin/env python3
"""Step 1: cut the WEAR egocentric video of one subject into 2 s activity clips.

REIMPLEMENTATION. The script that produced the clips used in the paper was not
recovered; this one reproduces its output layout, which the rest of the pipeline
relies on (the activity label is parsed from the folder name):

    <out>/sbj_<k>_clips/<activity>_part_<n>_from_<start>/<activity>_part_<n>_from_<start>_<iii>.mp4

* one folder per annotated activity segment; spaces in the label become ``_``,
  e.g. ``jogging (butt-kicks)`` -> ``jogging_(butt-kicks)``
* ``<n>`` counts the segments of that activity for the subject in time order (1-based)
* ``<start>`` is the segment start in seconds with two decimals
* the segment is cut into consecutive 2 s chunks ``_000, _001, ...``; the last chunk
  keeps the remainder (it can be shorter than 2 s)

The paper's clips were cut with the 30 fps WEAR annotations (start times such as
``281.04``); the re-converted subjects 16/17 (``*_new``) used the 60 fps annotations.
Clip boundaries can therefore differ by a few frames from the published event data.

    python data_preparation/cut_wear_clips.py --subject 3 \
        --video  "$NWEAR_DATA/wear_dataset/raw/videos/sbj_3.mp4" \
        --ann-json "$NWEAR_DATA/wear_dataset_annotations/60fps/wear_split_1.json" \
        --out-root "$NWEAR_DATA"

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

CHUNK_SEC = 2.0


def folder_name(label: str, part: int, start: float) -> str:
    return f"{label.replace(' ', '_')}_part_{part}_from_{start:.2f}"


def segments(ann_json: Path, subject: int) -> list[tuple[str, float, float]]:
    db = json.loads(ann_json.read_text())["database"]
    anns = db[f"sbj_{subject}"]["annotations"]
    return sorted(((a["label"], float(a["segment"][0]), float(a["segment"][1])) for a in anns), key=lambda s: s[1])


def cut(video: Path, start: float, duration: float, out: Path, overwrite: bool) -> None:
    if out.is_file() and not overwrite:
        return
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--video", type=Path, required=True, help="WEAR egocentric video of this subject")
    ap.add_argument("--ann-json", type=Path, required=True, help="any wear_split_*.json containing the subject")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.out_root / f"sbj_{args.subject}_clips"
    parts: dict[str, int] = defaultdict(int)
    n_clips = 0
    for label, t0, t1 in segments(args.ann_json, args.subject):
        parts[label] += 1
        name = folder_name(label, parts[label], t0)
        seg_dir = root / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        n_chunks = max(1, math.ceil((t1 - t0) / CHUNK_SEC - 1e-6))
        for i in range(n_chunks):
            start = t0 + i * CHUNK_SEC
            dur = min(CHUNK_SEC, t1 - start)
            if dur <= 0:
                break
            cut(args.video, start, dur, seg_dir / f"{name}_{i:03d}.mp4", args.overwrite)
            n_clips += 1
    print(f"sbj_{args.subject}: {sum(parts.values())} segments, {n_clips} clips -> {root}")


if __name__ == "__main__":
    main()
