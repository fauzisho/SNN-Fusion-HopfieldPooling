#!/usr/bin/env python3
"""Export per-frame Attend-and-Discriminate predictions/scores for a LOSO split.

Output schema matches infer_tridet_frames.py so score-pick / oracle fusion can
reuse the same npz keys (preds, gt, scores, class_scores, subject_ids, is_val).
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

WEAR_REPO = P.WEAR_REPO
if str(WEAR_REPO) not in sys.path:
    sys.path.insert(0, str(WEAR_REPO))

from inertial_baseline.AttendAndDiscriminate import AttendAndDiscriminate  # noqa: E402
from utils.data_utils import unwindow_inertial_data  # noqa: E402
from utils.torch_utils import InertialDataset, worker_init_reset_seed  # noqa: E402


def load_sens_for_subjects(sens_folder: Path, subjects: list[str], label_dict: dict) -> np.ndarray:
    chunks = []
    for sbj in subjects:
        csv_path = sens_folder / f"{sbj}.csv"
        df = pd.read_csv(csv_path, index_col=False).replace({"label": label_dict}).fillna(0)
        chunks.append(df.to_numpy())
    return np.concatenate(chunks, axis=0)


def predict_windows(
    net: AttendAndDiscriminate,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, scores, class_scores = [], [], []
    net.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device)
            logits = net(inputs)
            probs = F.softmax(logits, dim=-1)
            batch_preds = probs.argmax(dim=-1).cpu().numpy()
            batch_scores = probs.max(dim=-1).values.cpu().numpy()
            batch_class = probs.cpu().numpy()
            # Drop null column (index 0) for class_scores aligned with TriDet (18 activities).
            if batch_class.shape[1] == num_classes + 1:
                batch_class = batch_class[:, 1:]
            preds.append(batch_preds)
            scores.append(batch_scores)
            class_scores.append(batch_class)
    return (
        np.concatenate(preds),
        np.concatenate(scores),
        np.concatenate(class_scores),
    )


def export_frames(
    cfg_path: Path,
    checkpoint: Path,
    ann_json: Path,
    out_npz: Path,
    device: torch.device,
) -> None:
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.load(os.path.expandvars(handle.read()), Loader=yaml.FullLoader)  # ${NWEAR_DATA} in configs

    with open(ann_json, encoding="utf-8") as handle:
        ann = json.load(handle)

    labels = ["null"] + list(ann["label_dict"])
    label_dict = dict(zip(labels, range(len(labels))))
    sens_folder = Path(cfg["dataset"]["sens_folder"])

    all_subjects = sorted(
        [k for k in ann["database"] if (sens_folder / f"{k}.csv").is_file()],
        key=lambda s: int(s.split("_")[1]),
    )
    train_subjects = [k for k in all_subjects if ann["database"][k]["subset"] == "Training"]
    val_subjects = [k for k in all_subjects if ann["database"][k]["subset"] == "Validation"]

    sens = load_sens_for_subjects(sens_folder, all_subjects, label_dict)
    dataset = InertialDataset(
        sens,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_reset_seed,
    )

    num_classes = int(cfg["dataset"]["num_classes"])
    net = AttendAndDiscriminate(
        dataset.channels,
        dataset.classes,
        cfg["model"]["hidden_dim"],
        cfg["model"]["conv_kernels"],
        cfg["model"]["conv_kernel_size"],
        cfg["model"]["enc_layers"],
        cfg["model"]["enc_is_bidirectional"],
        cfg["model"]["dropout"],
        cfg["model"]["dropout_rnn"],
        cfg["model"]["dropout_cls"],
        cfg["model"]["activation"],
        cfg["model"]["sa_div"],
        gpu=str(device),
    )
    ckpt = torch.load(checkpoint, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)

    win_preds, win_scores, win_class = predict_windows(net, loader, device, num_classes)
    frame_preds, frame_gt = unwindow_inertial_data(
        sens,
        dataset.ids,
        win_preds,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
    )
    frame_scores, _ = unwindow_inertial_data(
        sens,
        dataset.ids,
        win_scores,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
    )

    # Unwindow each activity class probability independently.
    class_cols = []
    for c in range(win_class.shape[1]):
        col, _ = unwindow_inertial_data(
            sens,
            dataset.ids,
            win_class[:, c],
            cfg["dataset"]["window_size"],
            cfg["dataset"]["window_overlap"],
        )
        class_cols.append(col.astype(np.float32))
    frame_class = np.stack(class_cols, axis=1)

    subject_ids = sens[:, 0].astype(np.int64)
    is_val = np.isin(subject_ids, [int(s.split("_")[1]) for s in val_subjects])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        preds=frame_preds.astype(np.int64),
        gt=frame_gt.astype(np.int64),
        scores=frame_scores.astype(np.float32),
        class_scores=frame_class.astype(np.float32),
        subject_ids=subject_ids,
        is_val=is_val.astype(np.bool_),
        train_subjects=np.array([int(s.split("_")[1]) for s in train_subjects], dtype=np.int64),
        val_subjects=np.array([int(s.split("_")[1]) for s in val_subjects], dtype=np.int64),
    )
    print(f"Exported {len(frame_preds)} frames -> {out_npz}")
    print(f"  train subjects: {train_subjects}")
    print(f"  val subjects:   {val_subjects}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ann-json", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_frames(
        cfg_path=args.config,
        checkpoint=args.checkpoint,
        ann_json=args.ann_json,
        out_npz=args.out_npz,
        device=device,
    )


if __name__ == "__main__":
    main()
