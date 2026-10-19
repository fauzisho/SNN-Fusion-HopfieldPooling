#!/usr/bin/env python3
"""
Hopfield late fusion replacing oracle: fuse TriDet IMU + TriDet I3D frame scores.

Train fusion on Training subjects, evaluate on held-out Validation subject (LOSO).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import nwear_paths as P

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from hflayers import HopfieldPooling
except ImportError as exc:
    raise ImportError("pip install git+https://github.com/ml-jku/hopfield-layers") from exc

WEAR_REPO = P.WEAR_REPO


def _load_wear_postprocessing_utils():
    """Load wear repo postprocessing utils without TriDet postprocessing.py name clash."""
    path = WEAR_REPO / "postprocessing" / "utils.py"
    spec = importlib.util.spec_from_file_location("wear_postprocessing_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combine_predictions = _load_wear_postprocessing_utils().combine_predictions

NUM_CLASSES = 18
ACTIVITY_CLASSES = [
    "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
    "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)",
    "stretching (shoulders)", "stretching (hamstrings)", "stretching (lumbar rotation)",
    "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)",
    "burpees", "lunges", "lunges (complex)", "bench-dips",
]


def preds_to_soft(preds: np.ndarray, scores: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    soft = np.zeros((len(preds), num_classes), dtype=np.float32)
    for i, (p, s) in enumerate(zip(preds, scores)):
        cls = int(p)
        if 1 <= cls <= num_classes:
            soft[i, cls - 1] = max(float(s), 1e-6)
        else:
            soft[i, :] = 1.0 / num_classes
    row_sum = soft.sum(axis=1, keepdims=True)
    soft /= np.maximum(row_sum, 1e-8)
    return soft


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true > 0
    y_true = y_true[mask] - 1
    y_pred = y_pred[mask] - 1
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
        "n": int(len(y_true)),
    }


class HopfieldLateFusion(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 64):
        super().__init__()
        self.proj_imu = nn.Linear(num_classes, hidden)
        self.proj_i3d = nn.Linear(num_classes, hidden)
        self.pool = HopfieldPooling(
            input_size=hidden,
            output_size=hidden,
            hidden_size=max(hidden // 2, 16),
            num_heads=1,
            scaling=0.1,
            update_steps_max=3,
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, imu_soft: torch.Tensor, i3d_soft: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((self.proj_imu(imu_soft), self.proj_i3d(i3d_soft)), dim=1)
        fused = self.pool(tokens)
        return self.classifier(fused)


def plot_confusion(y_true, y_pred, out_path: Path, title: str) -> None:
    mask = y_true > 0
    y_true = y_true[mask] - 1
    y_pred = y_pred[mask] - 1
    k = NUM_CLASSES
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(ACTIVITY_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ACTIVITY_CLASSES, fontsize=7)
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label="Recall per class")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_fusion(
    imu_soft: np.ndarray,
    i3d_soft: np.ndarray,
    gt: np.ndarray,
    is_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> tuple[HopfieldLateFusion, dict]:
    tr = ~is_val
    va = is_val

    x_imu_tr = torch.tensor(imu_soft[tr], dtype=torch.float32)
    x_i3d_tr = torch.tensor(i3d_soft[tr], dtype=torch.float32)
    y_tr = torch.tensor(gt[tr] - 1, dtype=torch.long)
    mask_tr = y_tr >= 0

    model = HopfieldLateFusion().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(mask_tr.sum().item())
        idx = torch.where(mask_tr)[0][perm]
        total = 0.0
        for start in range(0, len(idx), batch_size):
            batch = idx[start : start + batch_size]
            xb_i, xb_v, yb = x_imu_tr[batch].to(device), x_i3d_tr[batch].to(device), y_tr[batch].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_i, xb_v), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  fusion epoch {ep+1}/{epochs} | train loss {total / max(len(idx) // batch_size, 1):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(imu_soft[va], dtype=torch.float32).to(device),
            torch.tensor(i3d_soft[va], dtype=torch.float32).to(device),
        )
        pred = logits.argmax(dim=1).cpu().numpy() + 1

    val_gt = gt[va]
    val_metrics = metrics(val_gt, pred)
    return model, val_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Hopfield late fusion on TriDet IMU + I3D exports.")
    ap.add_argument("--imu-npz", type=Path, required=True)
    ap.add_argument("--i3d-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--imu-label",
        default="TriDet IMU",
        help="Name for stream A in summary (e.g. 'A&D (IMU)' when fusing Attend-and-Discriminate).",
    )
    ap.add_argument(
        "--i3d-label",
        default="TriDet I3D",
        help="Name for stream B in summary.",
    )
    args = ap.parse_args()

    imu = np.load(args.imu_npz)
    i3d = np.load(args.i3d_npz)
    for key in ("preds", "gt", "scores", "is_val"):
        if len(imu[key]) != len(i3d[key]):
            raise SystemExit(f"IMU/I3D length mismatch on {key}")

    imu_soft = preds_to_soft(imu["preds"], imu["scores"])
    i3d_soft = preds_to_soft(i3d["preds"], i3d["scores"])
    gt = imu["gt"]
    is_val = imu["is_val"]

    val_mask = is_val & (gt > 0)
    imu_m = metrics(gt[is_val], imu["preds"][is_val])
    i3d_m = metrics(gt[is_val], i3d["preds"][is_val])

    oracle_val = combine_predictions(
        imu["preds"][val_mask].astype(int),
        i3d["preds"][val_mask].astype(int),
        gt[val_mask].astype(int),
    )
    oracle_val_m = {
        "acc": float(accuracy_score(gt[val_mask] - 1, oracle_val - 1)),
        "f1": float(f1_score(gt[val_mask] - 1, oracle_val - 1, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Hopfield late fusion on Training subjects...")
    model, fusion_m = train_fusion(
        imu_soft, i3d_soft, gt, is_val, args.epochs, args.lr, args.batch_size, device
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        logits = model(
            torch.tensor(imu_soft[is_val], dtype=torch.float32).to(device),
            torch.tensor(i3d_soft[is_val], dtype=torch.float32).to(device),
        )
        fusion_pred = logits.argmax(dim=1).cpu().numpy() + 1

    plot_confusion(
        gt[is_val], fusion_pred,
        args.out_dir / "02_confusion_hopfield_val.png",
        title=f"Single Hopfield ({args.imu_label}+{args.i3d_label}) | acc={fusion_m['acc']*100:.1f}%",
    )
    torch.save({"model_state_dict": model.state_dict(), "metrics": fusion_m}, args.out_dir / "model.pt")

    summary = (
        f"Single Hopfield late fusion — {args.imu_label} + {args.i3d_label}\n"
        f"Val subjects: {imu['val_subjects'].tolist()}\n"
        f"Train frames: {(~is_val).sum()} | Val frames: {is_val.sum()}\n\n"
        "--- Per-frame accuracy (validation subject, activity labels only) ---\n"
        f"{args.imu_label} only:     acc={imu_m['acc']*100:.2f}%  F1={imu_m['f1']*100:.2f}%  (n={imu_m['n']})\n"
        f"{args.i3d_label} only:     acc={i3d_m['acc']*100:.2f}%  F1={i3d_m['f1']*100:.2f}%  (n={i3d_m['n']})\n"
        f"Single Hopfield fusion: acc={fusion_m['acc']*100:.2f}%  F1={fusion_m['f1']*100:.2f}%  (n={fusion_m['n']})\n"
        f"Oracle (GT ceiling): acc={oracle_val_m['acc']*100:.2f}%  F1={oracle_val_m['f1']*100:.2f}%  [not deployable]\n"
    )
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
