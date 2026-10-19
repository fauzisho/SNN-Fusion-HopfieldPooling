#!/usr/bin/env python3
"""Export per-frame TriDet predictions/scores for all subjects in a LOSO split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

WEAR_REPO = P.WEAR_REPO
if str(WEAR_REPO) not in sys.path:
    sys.path.insert(0, str(WEAR_REPO))

from camera_baseline.tridet.libs.core.config import _update_config  # noqa: E402
from camera_baseline.tridet.libs.datasets import make_dataset, make_data_loader  # noqa: E402
from camera_baseline.tridet.libs.modeling import make_meta_arch  # noqa: E402
from camera_baseline.tridet.libs.utils.train_utils import valid_one_epoch  # noqa: E402
from utils.data_utils import convert_segments_to_samples  # noqa: E402


def load_model(
    cfg: dict,
    checkpoint: Path,
    device: torch.device,
    *,
    use_ema: bool = False,
) -> nn.Module:
    cfg = _update_config(cfg)
    model = make_meta_arch(cfg["model"]["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=[0])
    ckpt = torch.load(checkpoint, map_location=device)
    if use_ema:
        state = ckpt.get("state_dict_ema") or ckpt["state_dict"]
        weight_tag = "state_dict_ema"
    else:
        state = ckpt["state_dict"]
        weight_tag = "state_dict"
    model.load_state_dict(state)
    print(f"Loaded checkpoint weights: {weight_tag}")
    model.to(device)
    model.eval()
    return model


def segments_to_dataframe(segments) -> pd.DataFrame:
    if isinstance(segments, pd.DataFrame):
        return segments
    try:
        return pd.DataFrame(
            {
                "video_id": segments["video-id"],
                "t_start": segments["t-start"].tolist(),
                "t_end": segments["t-end"].tolist(),
                "label": segments["label"].tolist(),
                "score": segments["score"].tolist(),
            }
        )
    except Exception:
        return pd.DataFrame(columns=["video_id", "t_start", "t_end", "label", "score"])


def convert_segments_to_class_scores(
    segments,
    sens: np.ndarray,
    sampling_rate: int,
    num_classes: int = 18,
    threshold: float = 0.0,
) -> np.ndarray:
    """Per-frame max TriDet score for each activity class (N, num_classes)."""
    segments_df = segments_to_dataframe(segments)
    chunks: list[np.ndarray] = []
    for sbj in np.unique(sens[:, 0]):
        sbj_len = len(sens[sens[:, 0] == sbj])
        frame_scores = np.zeros((sbj_len, num_classes), dtype=np.float32)
        sbj_segments = segments_df[segments_df.video_id == f"sbj_{int(sbj)}"]
        if threshold > 0.0:
            sbj_segments = sbj_segments[sbj_segments.score > threshold]
        for _, seg in sbj_segments.iterrows():
            label_idx = int(seg["label"])
            if not (0 <= label_idx < num_classes):
                continue
            start = int(np.floor(seg["t_start"] * sampling_rate))
            end = int(np.ceil(seg["t_end"] * sampling_rate))
            frame_scores[start:end, label_idx] = np.maximum(
                frame_scores[start:end, label_idx],
                float(seg["score"]),
            )
        chunks.append(frame_scores)
    return np.concatenate(chunks, axis=0)


def load_sens_for_subjects(sens_folder: Path, subjects: list[str], label_dict: dict) -> np.ndarray:
    chunks = []
    for sbj in subjects:
        csv_path = sens_folder / f"{sbj}.csv"
        df = pd.read_csv(csv_path, index_col=False).replace({"label": label_dict}).fillna(0)
        chunks.append(df.to_numpy())
    return np.concatenate(chunks, axis=0)


def export_frames(
    cfg_path: Path,
    checkpoint: Path,
    ann_json: Path,
    out_npz: Path,
    score_threshold: float,
    use_ema: bool,
    device: torch.device,
) -> None:
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.load(os.path.expandvars(handle.read()), Loader=yaml.FullLoader)  # ${NWEAR_DATA} in configs

    with open(ann_json, encoding="utf-8") as handle:
        ann = json.load(handle)

    labels = ["null"] + list(ann["label_dict"])
    label_dict = dict(zip(labels, range(len(labels))))
    cfg["labels"] = labels
    cfg["label_dict"] = label_dict
    cfg["dataset"]["json_anno"] = str(ann_json)
    cfg["devices"] = ["cuda:0"]

    all_subjects = [k for k, v in ann["database"].items() if os.path.isfile(
        Path(cfg["dataset"]["feat_folder"]) / f"{k}.npy"
    )]
    train_subjects = [k for k in all_subjects if ann["database"][k]["subset"] == "Training"]
    val_subjects = [k for k in all_subjects if ann["database"][k]["subset"] == "Validation"]

    infer_cfg = dict(cfg)
    infer_cfg["train_split"] = ["training", "validation"]
    infer_cfg["val_split"] = ["training", "validation"]
    infer_cfg["dataset"] = dict(cfg["dataset"])
    infer_cfg["dataset"]["json_anno"] = str(ann_json)

    dataset = make_dataset(
        infer_cfg["dataset_name"],
        False,
        infer_cfg["val_split"],
        **infer_cfg["dataset"],
    )
    loader = make_data_loader(dataset, False, None, 1, infer_cfg["loader"]["num_workers"])

    model = load_model(infer_cfg, checkpoint, device, use_ema=use_ema)
    _, segments = valid_one_epoch(loader, model)

    sens = load_sens_for_subjects(
        Path(cfg["dataset"]["sens_folder"]),
        sorted(all_subjects, key=lambda s: int(s.split("_")[1])),
        label_dict,
    )
    sampling_rate = cfg["dataset"]["sampling_rate"]
    num_classes = cfg["dataset"]["num_classes"]
    preds, gt, scores = convert_segments_to_samples(
        segments,
        sens,
        sampling_rate,
        threshold_type="score",
        threshold=score_threshold,
    )
    class_scores = convert_segments_to_class_scores(
        segments,
        sens,
        sampling_rate,
        num_classes=num_classes,
        threshold=score_threshold,
    )

    subject_ids = sens[:, 0].astype(np.int64)
    is_val = np.isin(subject_ids, [int(s.split("_")[1]) for s in val_subjects])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        preds=preds.astype(np.int64),
        gt=gt.astype(np.int64),
        scores=scores.astype(np.float32),
        class_scores=class_scores.astype(np.float32),
        subject_ids=subject_ids,
        is_val=is_val.astype(np.bool_),
        train_subjects=np.array([int(s.split("_")[1]) for s in train_subjects], dtype=np.int64),
        val_subjects=np.array([int(s.split("_")[1]) for s in val_subjects], dtype=np.int64),
    )
    print(f"Exported {len(preds)} frames -> {out_npz}")
    print(f"  train subjects: {train_subjects}")
    print(f"  val subjects:   {val_subjects}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ann-json", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help=(
            "Drop low-scoring segments before rasterising. TriDet emits up to "
            "2000 segments per video, so at 0.0 every frame is covered and the "
            "NULL class is never predicted; WEAR uses 0.1 (default: 0.1)."
        ),
    )
    ap.add_argument(
        "--use-ema",
        action="store_true",
        help="Load EMA weights; default loads state_dict like TriDet validation.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_frames(
        cfg_path=args.config,
        checkpoint=args.checkpoint,
        ann_json=args.ann_json,
        out_npz=args.out_npz,
        score_threshold=args.score_threshold,
        use_ema=args.use_ema,
        device=device,
    )


if __name__ == "__main__":
    main()
