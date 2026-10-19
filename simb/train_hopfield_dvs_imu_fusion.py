#!/usr/bin/env python3
"""
Train Hopfield fusion on frozen DVS128 + IMU encoders (LOSO).

Modes:
  single — 1× HP-fusion over per-window DVS/IMU soft features (M10-style)
  double — HP-time per modality + HP-fusion over temporal windows (M11-style)

Encoders stay frozen; only fusion heads are trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import nwear_paths as P

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from hflayers import HopfieldPooling
except ImportError as exc:
    raise ImportError("pip install git+https://github.com/ml-jku/hopfield-layers") from exc

import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)

from dvs_imu_fusion_data import (  # noqa: E402
    ACTIVITY_CLASSES,
    FEAT_DIM,
    NUM_CLASSES,
    AlignedSplit,
    SubjectAligned,
    collect_aligned_split,
)
from pipeline_inertial import macro_f1_score  # noqa: E402



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


class SingleHopfieldDVSIMU(nn.Module):
    """Frozen-encoder soft features → proj → HP-fusion → classifier."""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj_dvs = nn.Linear(in_dim, hidden)
        self.proj_imu = nn.Linear(in_dim, hidden)
        self.pool = HopfieldPooling(
            input_size=hidden,
            output_size=hidden,
            hidden_size=max(hidden // 2, 16),
            num_heads=1,
            scaling=0.1,
            update_steps_max=3,
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, dvs_x: torch.Tensor, imu_x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((self.proj_dvs(dvs_x), self.proj_imu(imu_x)), dim=1)
        return self.classifier(self.pool(tokens))


class DoubleHopfieldDVSIMU(nn.Module):
    """HP-time per modality, then HP-fusion (TriDet double-HP analogue)."""

    def __init__(
        self,
        in_dim: int = FEAT_DIM,
        hidden: int = 128,
        num_classes: int = NUM_CLASSES,
        hopfield_heads: int = 1,
        hopfield_dropout: float = 0.1,
        hopfield_scaling: float = 0.1,
        hopfield_update_steps: int = 3,
    ):
        super().__init__()
        hp_kw = _hp_kwargs(hidden, hopfield_heads, hopfield_dropout, hopfield_scaling, hopfield_update_steps)
        self.proj_dvs = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.proj_imu = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.hp_time_dvs = HopfieldPooling(**hp_kw)
        self.hp_time_imu = HopfieldPooling(**hp_kw)
        self.hp_fusion = HopfieldPooling(**hp_kw)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def forward(self, dvs_seq: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_explain(dvs_seq, imu_seq)
        return logits

    def forward_explain(
        self, dvs_seq: torch.Tensor, imu_seq: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Same as forward, plus association weights for demos / plots.

        Returns
        -------
        logits : [B, C]
        info : dict with detached tensors
            proj_dvs / proj_imu : [B, W, H]
            assoc_time_dvs / assoc_time_imu : [B, heads, 1, W]
            assoc_fusion : [B, heads, 1, 2]  (DVS, IMU)
            s_dvs / s_imu / fused : pooled vectors [B, H]
        """
        h_dvs = self.proj_dvs(dvs_seq)
        h_imu = self.proj_imu(imu_seq)

        assoc_dvs = self.hp_time_dvs.get_association_matrix(h_dvs)
        assoc_imu = self.hp_time_imu.get_association_matrix(h_imu)
        s_dvs = self.hp_time_dvs(h_dvs)
        s_imu = self.hp_time_imu(h_imu)

        tokens = torch.stack((s_dvs, s_imu), dim=1)  # [B, 2, H]
        assoc_fus = self.hp_fusion.get_association_matrix(tokens)
        fused = self.hp_fusion(tokens)
        logits = self.classifier(fused)

        info = {
            "proj_dvs": h_dvs.detach(),
            "proj_imu": h_imu.detach(),
            "assoc_time_dvs": assoc_dvs.detach(),
            "assoc_time_imu": assoc_imu.detach(),
            "assoc_fusion": assoc_fus.detach(),
            "s_dvs": s_dvs.detach(),
            "s_imu": s_imu.detach(),
            "fused": fused.detach(),
        }
        return logits, info


def class_weights(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    counts = torch.clamp(counts, min=1.0)
    return (counts.sum() / (NUM_CLASSES * counts)).to(device)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"acc": 0.0, "f1": 0.0, "n": 0.0}
    return {
        "acc": float((y_true == y_pred).mean()),
        "f1": float(macro_f1_score(y_true, y_pred, num_classes=NUM_CLASSES)),
        "n": float(len(y_true)),
    }


def score_pick(dvs_feat: np.ndarray, imu_feat: np.ndarray) -> np.ndarray:
    dvs_probs = dvs_feat[:, :NUM_CLASSES]
    imu_probs = imu_feat[:, :NUM_CLASSES]
    dvs_conf = dvs_probs.max(axis=1)
    imu_conf = imu_probs.max(axis=1)
    return np.where(dvs_conf >= imu_conf, dvs_probs.argmax(1), imu_probs.argmax(1))


def build_temporal_windows(
    subjects: list[SubjectAligned],
    *,
    train_only: bool,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dvs_w, imu_w, labels = [], [], []
    for subj in subjects:
        if train_only and subj.is_val:
            continue
        n = len(subj.labels)
        if n < seq_len:
            continue
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            center = start + seq_len // 2
            label = int(subj.labels[center])
            dvs_w.append(subj.dvs_feat[start:end])
            imu_w.append(subj.imu_feat[start:end])
            labels.append(label)
    if not dvs_w:
        empty = np.empty((0, seq_len, FEAT_DIM), dtype=np.float32)
        return empty, empty.copy(), np.empty(0, dtype=np.int64)
    return (
        np.stack(dvs_w).astype(np.float32),
        np.stack(imu_w).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


@torch.no_grad()
def predict_double_on_subject(
    model: DoubleHopfieldDVSIMU,
    subj: SubjectAligned,
    *,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    n = len(subj.labels)
    preds = np.full(n, -1, dtype=np.int64)
    half = seq_len // 2
    if n < seq_len:
        return preds

    batch_dvs, batch_imu, batch_centers = [], [], []
    for start in range(0, n - seq_len + 1):
        end = start + seq_len
        center = start + half
        batch_dvs.append(subj.dvs_feat[start:end])
        batch_imu.append(subj.imu_feat[start:end])
        batch_centers.append(center)
        if len(batch_dvs) == 256 or start == n - seq_len:
            logits = model(
                torch.tensor(np.stack(batch_dvs), dtype=torch.float32, device=device),
                torch.tensor(np.stack(batch_imu), dtype=torch.float32, device=device),
            )
            pred = logits.argmax(dim=1).cpu().numpy()
            for c, p in zip(batch_centers, pred):
                preds[c] = int(p)
            batch_dvs, batch_imu, batch_centers = [], [], []

    filled = preds >= 0
    for i in range(n):
        if filled[i]:
            continue
        left = np.where(preds[:i] >= 0)[0]
        right = np.where(preds[i + 1 :] >= 0)[0]
        if len(left):
            preds[i] = preds[left[-1]]
        elif len(right):
            preds[i] = preds[right[0] + i + 1]
    return preds


def train_single(
    model: SingleHopfieldDVSIMU,
    dvs_tr: torch.Tensor,
    imu_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    cls_weight: torch.Tensor | None,
) -> SingleHopfieldDVSIMU:
    loader = DataLoader(TensorDataset(dvs_tr, imu_tr, y_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_loss = None, float("inf")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb_d, xb_i, yb in loader:
            xb_d, xb_i, yb = xb_d.to(device), xb_i.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_d, xb_i), yb, weight=cls_weight)
            loss.backward()
            opt.step()
            total += loss.item()
        ep_loss = total / max(len(loader), 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep + 1 == epochs:
            print(f"  single HP epoch {ep+1:3d}/{epochs} | loss {ep_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_double(
    model: DoubleHopfieldDVSIMU,
    dvs_tr: torch.Tensor,
    imu_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    cls_weight: torch.Tensor | None,
) -> DoubleHopfieldDVSIMU:
    loader = DataLoader(TensorDataset(dvs_tr, imu_tr, y_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_loss = None, float("inf")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb_d, xb_i, yb in loader:
            xb_d, xb_i, yb = xb_d.to(device), xb_i.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_d, xb_i), yb, weight=cls_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        ep_loss = total / max(len(loader), 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep + 1 == epochs:
            print(f"  double HP epoch {ep+1:3d}/{epochs} | loss {ep_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def plot_confusion(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTIVITY_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ACTIVITY_CLASSES, fontsize=7)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label="Recall per class")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fmt(m: dict[str, float]) -> str:
    return f"acc={100*m['acc']:.2f}%  F1={100*m['f1']:.2f}%  (n={int(m['n'])})"


def oracle_pick(y_true: np.ndarray, dvs_pred: np.ndarray, imu_pred: np.ndarray) -> np.ndarray:
    out = dvs_pred.copy()
    dvs_ok = dvs_pred == y_true
    imu_ok = imu_pred == y_true
    out[~dvs_ok & imu_ok] = imu_pred[~dvs_ok & imu_ok]
    return out


def evaluate_baselines(val_subj: SubjectAligned) -> dict[str, dict[str, float]]:
    y = val_subj.labels
    dvs_pred = val_subj.dvs_feat[:, :NUM_CLASSES].argmax(axis=1)
    imu_pred = val_subj.imu_feat[:, :NUM_CLASSES].argmax(axis=1)
    return {
        "dvs_only": metrics(y, dvs_pred),
        "imu_only": metrics(y, imu_pred),
        "score_pick": metrics(y, score_pick(val_subj.dvs_feat, val_subj.imu_feat)),
        "oracle": metrics(y, oracle_pick(y, dvs_pred, imu_pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hopfield fusion on frozen DVS128 + IMU encoders")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-subject", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--dvs-out-root",
        type=Path,
        default=P.EVENT_RUNS,
    )
    ap.add_argument(
        "--imu-out-root",
        type=Path,
        default=P.IMU_RUNS,
    )
    ap.add_argument(
        "--feat-dir",
        type=Path,
        default=P.IMU_FEAT_DIR,
    )
    ap.add_argument("--ann-dir", type=Path, default=P.ANN_DIR)
    ap.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    ap.add_argument(
        "--dvs-cache-dir",
        type=Path,
        default=P.EVENT_TENSOR_CACHE,
    )
    ap.add_argument("--max-time-diff", type=float, default=1.0)
    ap.add_argument("--imu-hidden", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--single-epochs", type=int, default=30)
    ap.add_argument("--double-epochs", type=int, default=50)
    ap.add_argument("--single-lr", type=float, default=1e-3)
    ap.add_argument("--double-lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--single-hidden", type=int, default=64)
    ap.add_argument("--double-hidden", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=5, help="Aligned windows per temporal HP input (~10s @ 2s clips)")
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["single", "double"], choices=["single", "double"])
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA requested but not available.")
    device = torch.device(args.device)

    split = args.val_subject + 1
    ann_path = args.ann_dir / f"wear_split_{split}.json"
    dvs_ckpt = args.dvs_out_root / f"seed_{args.seed}" / f"test_sbj_{args.val_subject}" / "model.pt"
    imu_ckpt = args.imu_out_root / f"seed_{args.seed}" / f"train_split_{split}" / "model.pt"
    if not dvs_ckpt.is_file():
        raise SystemExit(f"Missing DVS checkpoint: {dvs_ckpt}")
    if not imu_ckpt.is_file():
        raise SystemExit(f"Missing IMU checkpoint: {imu_ckpt}")
    out_dir = args.out_dir or (
        P.SIMB_HEADS / f"seed_{args.seed}" / f"sbj_{args.val_subject}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Hopfield DVS+IMU fusion | seed={args.seed} | sbj_{args.val_subject} | device={device}")
    print("Frozen encoders → train Hopfield fusion heads only")
    print("Collecting aligned features from all subjects...")
    aligned = collect_aligned_split(
        seed=args.seed,
        val_subject=args.val_subject,
        dvs_ckpt=dvs_ckpt,
        imu_ckpt=imu_ckpt,
        feat_dir=args.feat_dir,
        ann_path=ann_path,
        dataset_base=args.dataset_base,
        dvs_cache_dir=args.dvs_cache_dir,
        max_time_diff=args.max_time_diff,
        imu_hidden=args.imu_hidden,
        min_overlap=0.6,
        device=device,
        batch_size=args.batch_size,
    )
    val_subj = next(s for s in aligned.subjects if s.is_val)
    train_subjects = [s for s in aligned.subjects if not s.is_val]

    baselines = evaluate_baselines(val_subj)
    results: dict[str, dict[str, float]] = dict(baselines)
    summary = [
        "DVS128 + IMU Hopfield fusion (frozen encoders)",
        f"seed={args.seed} hold_out=sbj_{args.val_subject} split={split} device={device}",
        f"feat_dim={FEAT_DIM} (probs + max_prob + entropy)",
        f"train_subjects={[s.subject_id for s in train_subjects]}",
        f"val_pairs={len(val_subj.labels)}",
        "",
        "--- Baselines (val aligned pairs) ---",
    ]
    for name, m in baselines.items():
        tag = " [not deployable]" if name == "oracle" else ""
        line = f"{name:22s}: {fmt(m)}{tag}"
        print(line)
        summary.append(line)

    if "single" in args.modes:
        dvs_tr = torch.tensor(np.concatenate([s.dvs_feat for s in train_subjects]), dtype=torch.float32)
        imu_tr = torch.tensor(np.concatenate([s.imu_feat for s in train_subjects]), dtype=torch.float32)
        y_tr = torch.tensor(np.concatenate([s.labels for s in train_subjects]), dtype=torch.long)
        print(f"\nSingle HP | train samples={len(y_tr)}")
        single = SingleHopfieldDVSIMU(FEAT_DIM, hidden=args.single_hidden).to(device)
        w = None if args.no_class_weights else class_weights(y_tr, device)
        single = train_single(
            single, dvs_tr, imu_tr, y_tr,
            epochs=args.single_epochs, lr=args.single_lr, batch_size=args.batch_size,
            device=device, cls_weight=w,
        )
        single.eval()
        with torch.no_grad():
            logits = single(
                torch.tensor(val_subj.dvs_feat, dtype=torch.float32, device=device),
                torch.tensor(val_subj.imu_feat, dtype=torch.float32, device=device),
            )
            single_pred = logits.argmax(dim=1).cpu().numpy()
        results["single_hp"] = metrics(val_subj.labels, single_pred)
        torch.save(single.state_dict(), out_dir / "single_hopfield.pt")
        plot_confusion(
            val_subj.labels, single_pred,
            out_dir / "confusion_single_hp.png",
            title=f"Single HP | {fmt(results['single_hp'])}",
        )
        print(f"single_hp              : {fmt(results['single_hp'])}")

    if "double" in args.modes:
        x_dvs_tr, x_imu_tr, y_dw = build_temporal_windows(
            aligned.subjects, train_only=True, seq_len=args.seq_len, stride=args.train_stride,
        )
        print(f"\nDouble HP | train windows={len(y_dw)} seq_len={args.seq_len} stride={args.train_stride}")
        if len(y_dw) == 0:
            raise SystemExit("No double-HP training windows.")
        double = DoubleHopfieldDVSIMU(in_dim=FEAT_DIM, hidden=args.double_hidden).to(device)
        w = None if args.no_class_weights else class_weights(torch.tensor(y_dw), device)
        double = train_double(
            double,
            torch.tensor(x_dvs_tr, dtype=torch.float32),
            torch.tensor(x_imu_tr, dtype=torch.float32),
            torch.tensor(y_dw, dtype=torch.long),
            epochs=args.double_epochs, lr=args.double_lr, batch_size=args.batch_size,
            device=device, cls_weight=w,
        )
        double.eval()
        double_pred = predict_double_on_subject(double, val_subj, seq_len=args.seq_len, device=device)
        results["double_hp"] = metrics(val_subj.labels, double_pred)
        torch.save(double.state_dict(), out_dir / "double_hopfield.pt")
        plot_confusion(
            val_subj.labels, double_pred,
            out_dir / "confusion_double_hp.png",
            title=f"Double HP | {fmt(results['double_hp'])}",
        )
        print(f"double_hp              : {fmt(results['double_hp'])}")

    summary.extend(["", "--- Hopfield fusion ---"])
    for key in ("single_hp", "double_hp"):
        if key in results:
            summary.append(f"{key:22s}: {fmt(results[key])}")

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_subject": args.val_subject,
                "device": str(device),
                "results": results,
                "seq_len": args.seq_len,
                "train_stride": args.train_stride,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
