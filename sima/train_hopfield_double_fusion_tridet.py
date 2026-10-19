#!/usr/bin/env python3
"""
TriDet + double Hopfield late fusion (no BiGRU).

Per modality: project TriDet frame soft scores → HP-time over a temporal window.
Cross-modal: HP-fusion over IMU vs I3D summaries → classifier.

Train on Training-subject windows; evaluate per-frame on held-out Validation subject (LOSO).
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
import nwear_paths as P

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from hflayers import HopfieldPooling
except ImportError as exc:
    raise ImportError("pip install git+https://github.com/ml-jku/hopfield-layers") from exc

WEAR_REPO = P.WEAR_REPO


def _load_wear_postprocessing_utils():
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


def enrich_soft(soft: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Append TriDet confidence and max-probability per frame."""
    conf = scores.astype(np.float32).reshape(-1, 1)
    max_prob = soft.max(axis=1, keepdims=True)
    return np.concatenate([soft, conf, max_prob], axis=1)


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


def score_pick_pred(
    imu_preds: np.ndarray,
    i3d_preds: np.ndarray,
    imu_scores: np.ndarray,
    i3d_scores: np.ndarray,
) -> np.ndarray:
    pick_imu = imu_scores >= i3d_scores
    return np.where(pick_imu, imu_preds, i3d_preds)


def class_weights(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    return w.to(device)


def _hp_kwargs(hidden: int, heads: int, dropout: float, scaling: float, steps: int) -> dict:
    return dict(
        input_size=hidden,
        output_size=hidden,
        hidden_size=max(hidden // 2, 32),
        num_heads=heads,
        dropout=dropout,
        scaling=scaling,
        update_steps_max=steps,
    )


class TriDetDoubleHopfieldFusion(nn.Module):
    """HP-time per TriDet modality, then HP-fusion across modalities."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_classes: int = NUM_CLASSES,
        hopfield_heads: int = 1,
        hopfield_dropout: float = 0.1,
        hopfield_scaling: float = 0.1,
        hopfield_update_steps: int = 3,
    ):
        super().__init__()
        hp_kw = _hp_kwargs(hidden, hopfield_heads, hopfield_dropout, hopfield_scaling, hopfield_update_steps)
        self.proj_imu = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.proj_i3d = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.hp_time_imu = HopfieldPooling(**hp_kw)
        self.hp_time_i3d = HopfieldPooling(**hp_kw)
        self.hp_fusion = HopfieldPooling(**hp_kw)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, imu_seq: torch.Tensor, i3d_seq: torch.Tensor) -> torch.Tensor:
        h_imu = self.proj_imu(imu_seq)
        h_i3d = self.proj_i3d(i3d_seq)
        s_imu = self.hp_time_imu(h_imu)
        s_i3d = self.hp_time_i3d(h_i3d)
        fused = self.hp_fusion(torch.stack((s_imu, s_i3d), dim=1))
        return self.classifier(fused)


def build_subject_windows(
    imu_feat: np.ndarray,
    i3d_feat: np.ndarray,
    gt: np.ndarray,
    subject_ids: np.ndarray,
    subject_filter: np.ndarray | None,
    seq_len: int,
    stride: int,
    label_mode: str = "center",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (N, T, F) windows; label at center frame by default."""
    imu_windows, i3d_windows, labels, frame_indices = [], [], [], []
    half = seq_len // 2
    subjects = np.unique(subject_ids)
    if subject_filter is not None:
        subjects = subjects[subject_filter[subjects]]

    for sid in subjects:
        idx = np.where(subject_ids == sid)[0]
        if len(idx) < seq_len:
            continue
        imu_s = imu_feat[idx]
        i3d_s = i3d_feat[idx]
        gt_s = gt[idx]
        global_idx = idx

        for start in range(0, len(idx) - seq_len + 1, stride):
            end = start + seq_len
            center = start + half
            if label_mode == "center":
                label = int(gt_s[center])
                if label <= 0:
                    continue
                target_frame = int(global_idx[center])
            else:
                active = gt_s[start:end]
                active = active[active > 0]
                if len(active) == 0:
                    continue
                label, count = Counter(active.tolist()).most_common(1)[0]
                if count / seq_len < 0.6:
                    continue
                target_frame = int(global_idx[center])

            imu_windows.append(imu_s[start:end])
            i3d_windows.append(i3d_s[start:end])
            labels.append(label - 1)
            frame_indices.append(target_frame)

    if not imu_windows:
        empty = np.empty((0, seq_len, imu_feat.shape[-1]), dtype=np.float32)
        return empty, empty.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return (
        np.stack(imu_windows).astype(np.float32),
        np.stack(i3d_windows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(frame_indices, dtype=np.int64),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight: torch.Tensor | None,
) -> float:
    model.train()
    total = 0.0
    for imu_x, i3d_x, y in loader:
        imu_x, i3d_x, y = imu_x.to(device), i3d_x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(imu_x, i3d_x), y, weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict_frames(
    model: nn.Module,
    imu_feat: np.ndarray,
    i3d_feat: np.ndarray,
    gt: np.ndarray,
    subject_ids: np.ndarray,
    val_subject_ids: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Sliding-window inference; assign each val frame the latest window prediction."""
    half = seq_len // 2
    n_frames = len(gt)
    preds = np.zeros(n_frames, dtype=np.int64)
    filled = np.zeros(n_frames, dtype=bool)

    for sid in val_subject_ids:
        idx = np.where(subject_ids == sid)[0]
        if len(idx) < seq_len:
            continue
        imu_s = imu_feat[idx]
        i3d_s = i3d_feat[idx]
        global_idx = idx

        batch_imu, batch_i3d, batch_centers = [], [], []
        batch_size = 512
        for start in range(0, len(idx) - seq_len + 1):
            end = start + seq_len
            center = start + half
            batch_imu.append(imu_s[start:end])
            batch_i3d.append(i3d_s[start:end])
            batch_centers.append(int(global_idx[center]))

            if len(batch_imu) == batch_size or start == len(idx) - seq_len:
                logits = model(
                    torch.tensor(np.stack(batch_imu), dtype=torch.float32, device=device),
                    torch.tensor(np.stack(batch_i3d), dtype=torch.float32, device=device),
                )
                pred = logits.argmax(dim=1).cpu().numpy() + 1
                for frame_idx, p in zip(batch_centers, pred):
                    preds[frame_idx] = p
                    filled[frame_idx] = True
                batch_imu, batch_i3d, batch_centers = [], [], []

    # Edge frames: copy nearest filled neighbor inside same val subject
    for sid in val_subject_ids:
        idx = np.where(subject_ids == sid)[0]
        if len(idx) == 0:
            continue
        for local_i, frame_idx in enumerate(idx):
            if filled[frame_idx]:
                continue
            left = idx[:local_i][filled[idx[:local_i]]]
            right = idx[local_i + 1 :][filled[idx[local_i + 1 :]]]
            if len(left):
                preds[frame_idx] = preds[left[-1]]
            elif len(right):
                preds[frame_idx] = preds[right[0]]
            else:
                preds[frame_idx] = 0
            filled[frame_idx] = True

    return preds


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


def main() -> None:
    ap = argparse.ArgumentParser(description="TriDet double Hopfield fusion (HP-time + HP-fusion).")
    ap.add_argument("--imu-npz", type=Path, required=True)
    ap.add_argument("--i3d-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=60, help="Temporal window in frames (1 s @ 50 Hz).")
    ap.add_argument("--train-stride", type=int, default=30, help="Training window stride.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--hopfield-heads", type=int, default=1)
    ap.add_argument("--hopfield-dropout", type=float, default=0.1)
    ap.add_argument("--hopfield-scaling", type=float, default=0.1)
    ap.add_argument("--hopfield-update-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-class-weights", action="store_true")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    imu = np.load(args.imu_npz)
    i3d = np.load(args.i3d_npz)
    for key in ("preds", "gt", "scores", "is_val", "subject_ids"):
        if len(imu[key]) != len(i3d[key]):
            raise SystemExit(f"IMU/I3D length mismatch on {key}")

    imu_soft = enrich_soft(preds_to_soft(imu["preds"], imu["scores"]), imu["scores"])
    i3d_soft = enrich_soft(preds_to_soft(i3d["preds"], i3d["scores"]), i3d["scores"])
    gt = imu["gt"]
    is_val = imu["is_val"]
    subject_ids = imu["subject_ids"]
    val_subjects = imu["val_subjects"]
    in_dim = imu_soft.shape[1]

    train_mask = ~is_val
    train_subjects = np.unique(subject_ids[train_mask])
    train_subject_filter = np.zeros(int(subject_ids.max()) + 1, dtype=bool)
    train_subject_filter[train_subjects] = True

    print(f"Building train windows: seq_len={args.seq_len}, stride={args.train_stride}")
    x_imu_tr, x_i3d_tr, y_tr, _ = build_subject_windows(
        imu_soft, i3d_soft, gt, subject_ids, train_subject_filter,
        args.seq_len, args.train_stride, label_mode="center",
    )
    print(f"Train windows: {len(y_tr)}")
    if len(y_tr) == 0:
        raise SystemExit("No training windows produced.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriDetDoubleHopfieldFusion(
        in_dim=in_dim,
        hidden=args.hidden,
        hopfield_heads=args.hopfield_heads,
        hopfield_dropout=args.hopfield_dropout,
        hopfield_scaling=args.hopfield_scaling,
        hopfield_update_steps=args.hopfield_update_steps,
    ).to(device)

    weight = None if args.no_class_weights else class_weights(torch.tensor(y_tr), device)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_imu_tr, dtype=torch.float32),
            torch.tensor(x_i3d_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [max(1, args.epochs // 3), max(2, 2 * args.epochs // 3)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    best_state = None
    best_loss = float("inf")
    print(f"Training double Hopfield fusion on {device}...")
    for ep in range(args.epochs):
        tr_loss = train_epoch(model, loader, optimizer, device, weight)
        scheduler.step()
        if tr_loss < best_loss:
            best_loss = tr_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, args.epochs // 10) == 0 or ep + 1 == args.epochs:
            print(f"  epoch {ep+1:3d}/{args.epochs} | train loss {tr_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Per-frame inference on validation subject (sliding window)...")
    fusion_pred = predict_frames(
        model, imu_soft, i3d_soft, gt, subject_ids, val_subjects,
        args.seq_len, device,
    )

    val_gt = gt[is_val]
    imu_val_pred = imu["preds"][is_val]
    i3d_val_pred = i3d["preds"][is_val]
    fusion_val_pred = fusion_pred[is_val]

    imu_m = metrics(val_gt, imu_val_pred)
    i3d_m = metrics(val_gt, i3d_val_pred)
    fusion_m = metrics(val_gt, fusion_val_pred)

    score_pick = score_pick_pred(
        imu["preds"], i3d["preds"], imu["scores"], i3d["scores"],
    )
    score_pick_m = metrics(gt[is_val], score_pick[is_val])

    val_mask = is_val & (gt > 0)
    oracle_val = combine_predictions(
        imu["preds"][val_mask].astype(int),
        i3d["preds"][val_mask].astype(int),
        gt[val_mask].astype(int),
    )
    oracle_val_m = {
        "acc": float(accuracy_score(gt[val_mask] - 1, oracle_val - 1)),
        "f1": float(f1_score(gt[val_mask] - 1, oracle_val - 1, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        val_gt, fusion_val_pred,
        args.out_dir / "02_confusion_double_hp_val.png",
        title=f"Double Hopfield ({args.imu_label}+{args.i3d_label}) | acc={fusion_m['acc']*100:.1f}%",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": fusion_m,
            "seq_len": args.seq_len,
            "train_stride": args.train_stride,
            "hidden": args.hidden,
        },
        args.out_dir / "model.pt",
    )

    summary = (
        f"Double Hopfield fusion (HP-time + HP-fusion) — {args.imu_label} + {args.i3d_label}\n"
        f"Val subjects: {val_subjects.tolist()}\n"
        f"Train windows: {len(y_tr)} | seq_len={args.seq_len} train_stride={args.train_stride}\n"
        f"hidden={args.hidden} epochs={args.epochs} batch={args.batch_size} lr={args.lr}\n\n"
        "--- Per-frame accuracy (validation subject, activity labels only) ---\n"
        f"{args.imu_label} only:        acc={imu_m['acc']*100:.2f}%  F1={imu_m['f1']*100:.2f}%  (n={imu_m['n']})\n"
        f"{args.i3d_label} only:        acc={i3d_m['acc']*100:.2f}%  F1={i3d_m['f1']*100:.2f}%  (n={i3d_m['n']})\n"
        f"Score-pick baseline:    acc={score_pick_m['acc']*100:.2f}%  F1={score_pick_m['f1']*100:.2f}%  (n={score_pick_m['n']})\n"
        f"Double Hopfield fusion: acc={fusion_m['acc']*100:.2f}%  F1={fusion_m['f1']*100:.2f}%  (n={fusion_m['n']})\n"
        f"Oracle (GT ceiling):    acc={oracle_val_m['acc']*100:.2f}%  F1={oracle_val_m['f1']*100:.2f}%  [not deployable]\n"
    )
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
