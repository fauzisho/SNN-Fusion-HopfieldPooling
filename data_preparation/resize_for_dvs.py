#!/usr/bin/env python3
"""Resize video(s) to DVS sensor resolutions supported by v2e.

Creates one output file per model in the output directory, named:
  {stem}_{model}.mp4  (e.g. burpees_part_1_from_545.04_000_dvs640.mp4)

Requires ffmpeg on PATH, or: pip install imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Supported v2e DVS models and native resolutions (width x height).
DVS_MODELS: dict[str, tuple[int, int]] = {
    "dvs128": (128, 128),
    "dvs240": (240, 180),
    "dvs346": (346, 260),
    "dvs640": (640, 480),
    "dvs1024": (1024, 768),
}


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit(
            "ffmpeg not found. Install system ffmpeg or: pip install imageio-ffmpeg"
        ) from exc


def build_scale_filter(width: int, height: int, stretch: bool) -> str:
    if stretch:
        return f"scale={width}:{height}"
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def resize_video(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    *,
    stretch: bool,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = build_scale_filter(width, height, stretch)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "burpees_part_1_from_545.04_000.mp4"

    parser = argparse.ArgumentParser(
        description="Create DVS-resolution copies of a video file."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input video(s). Default: burpees_part_1_from_545.04_000.mp4 in this folder.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for resized outputs (default: same folder as this script).",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=sorted(DVS_MODELS),
        default=sorted(DVS_MODELS),
        metavar="MODEL",
        help="DVS models to generate (default: all).",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Stretch to exact WxH (default: letterbox to fit, preserve aspect ratio).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "burpees_part_1_from_545.04_000.mp4"
    inputs = args.inputs if args.inputs else [default_input]

    ffmpeg = find_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg}")

    for input_path in inputs:
        input_path = input_path.resolve()
        if not input_path.is_file():
            print(f"Skip (not found): {input_path}", file=sys.stderr)
            continue

        stem = input_path.stem
        print(f"\nInput: {input_path}")

        for model in args.models:
            width, height = DVS_MODELS[model]
            output_path = args.output_dir / f"{stem}_{model}.mp4"
            print(f"  -> {model} ({width}x{height}): {output_path}")
            resize_video(
                ffmpeg,
                input_path,
                output_path,
                width,
                height,
                stretch=args.stretch,
                overwrite=args.overwrite,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
