#!/usr/bin/env python3
"""SIM A Hopfield duty-cycle: 54 LOSO folds, single and double HP.

Same frozen trigger as score-pick BF1/BF2 (not retuned):
  k = 4, 2 s windows (100 @ 50 Hz, stride 50)
  uncertain iff entropy > 0.05 or mean max-prob < 0.90
  ON: last k uncertain, or disagreement while on
  OFF: last k certain and IMU/camera labels agree

Home = IMU always on. Dual stream when camera is on = Hopfield preds
(re-inferred from model.pt), NOT score-pick.

Usage:
  python3 eval_hopfield_duty_cycle_loso54.py --pairing bf1 --device cuda
  python3 eval_hopfield_duty_cycle_loso54.py --pairing bf2 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = P.OUTPUT_ROOT / "sima"
P.add_to_path(P.SIMA_DIR)
from aggregate_ad_fusion_map import (  # noqa: E402
    enrich_soft,
    infer_double_hp,
    infer_single_hp,
    preds_to_soft,
)

NUM_CLASSES = 18
WIN = 100
STRIDE = 50
K = 4
ENT_THR = 0.05
MAXP_THR = 0.90

PAIRINGS = {
    "bf1": {
        "root": P.SIMA_BF1,
        "imu_npz": "ad_frames.npz",
        "label": "BF1 A-and-D IMU + TriDet camera",
    },
    "bf2": {
        "root": P.SIMA_BF2,
        "imu_npz": "imu_frames.npz",
        "label": "BF2 TriDet IMU + TriDet camera",
    },
}

HEADS = ("single", "double")

MEAN_KEYS = [
    "d_windows",
    "d_all",
    "d_activity",
    "uncertain_windows",
    "imu_P",
    "imu_R",
    "imu_F1",
    "cam_P",
    "cam_R",
    "cam_F1",
    "always_on_P",
    "always_on_R",
    "always_on_F1",
    "dynamic_P",
    "dynamic_R",
    "dynamic_F1",
    "retention_F1",
    "lift_F1",
]


def prf(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = gt > 0
    y = gt[mask] - 1
    p = np.clip(pred[mask] - 1, 0, NUM_CLASSES - 1)
    return {
        "acc": float(accuracy_score(y, p) * 100),
        "P": float(precision_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "R": float(recall_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "F1": float(f1_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "n": int(mask.sum()),
    }


def window_flags(prob: np.ndarray, imu_pred: np.ndarray, cam_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(prob)
    n_w = 1 + max(0, (n - WIN) // STRIDE) if n >= WIN else 1
    uncertain = np.zeros(n_w, dtype=bool)
    disagree = np.zeros(n_w, dtype=bool)
    for i in range(n_w):
        start = min(i * STRIDE, max(0, n - WIN))
        end = min(start + WIN, n)
        sl = slice(start, end)
        mx = float(prob[sl].max(axis=1).mean())
        ent = float((-(prob[sl] * np.log(np.maximum(prob[sl], 1e-12))).sum(axis=1)).mean())
        uncertain[i] = (ent > ENT_THR) or (mx < MAXP_THR)
        c = start + (end - start) // 2
        disagree[i] = int(imu_pred[c]) != int(cam_pred[c])
    return uncertain, disagree


def camera_on_windows(uncertain: np.ndarray, disagree: np.ndarray) -> np.ndarray:
    n = len(uncertain)
    on = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        left = max(0, i - K + 1)
        full = (i - left + 1) >= K
        block_u = bool(np.all(uncertain[left : i + 1])) if full else False
        block_certain = bool(np.all(~uncertain[left : i + 1])) if full else False
        block_agree = bool(np.all(~disagree[left : i + 1])) if full else False
        if not state:
            if block_u:
                state = True
        elif block_certain and block_agree:
            state = False
        elif uncertain[i] or disagree[i]:
            state = True
        on[i] = state
    return on


def expand_on(n_frames: int, on_w: np.ndarray) -> np.ndarray:
    out = np.zeros(n_frames, dtype=bool)
    n_w = len(on_w)
    for t in range(n_frames):
        out[t] = bool(on_w[min(t // STRIDE, n_w - 1)])
    return out


def ms(rows: list[dict], key: str) -> dict[str, float]:
    xs = [float(r[key]) for r in rows]
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
    }


def hopfield_preds(
    split_dir: Path,
    head: str,
    imu,
    i3d,
    device: torch.device,
    force: bool,
) -> np.ndarray:
    cache = split_dir / f"hopfield_{'late' if head == 'single' else 'double'}_fusion" / "duty_cycle_preds.npy"
    if cache.exists() and not force:
        return np.load(cache)
    imu_soft = preds_to_soft(imu["preds"], imu["scores"])
    i3d_soft = preds_to_soft(i3d["preds"], i3d["scores"])
    if head == "single":
        preds = infer_single_hp(
            split_dir / "hopfield_late_fusion" / "model.pt",
            imu_soft,
            i3d_soft,
            imu["is_val"],
            device,
        )
    else:
        imu_feat = enrich_soft(imu_soft, imu["scores"])
        i3d_feat = enrich_soft(i3d_soft, i3d["scores"])
        preds = infer_double_hp(
            split_dir / "hopfield_double_fusion" / "model.pt",
            imu_feat,
            i3d_feat,
            imu["subject_ids"],
            imu["val_subjects"],
            device,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, preds)
    return preds


def run_split(
    pairing: str,
    seed: int,
    split: int,
    device: torch.device,
    force: bool,
) -> dict[str, dict]:
    cfg = PAIRINGS[pairing]
    split_dir = cfg["root"] / f"seed_{seed}" / f"split_{split}"
    imu = np.load(split_dir / cfg["imu_npz"])
    i3d = np.load(split_dir / "i3d_frames.npz")
    fus = np.load(split_dir / "score_oracle_fusion" / "fusion_preds.npz")
    idx = np.where(imu["is_val"])[0]
    gt_v = imu["gt"][idx].astype(np.int64)
    imu_pred = fus["ad_preds"][idx].astype(np.int64)
    cam = fus["i3d_preds"][idx].astype(np.int64)
    cs = imu["class_scores"][idx]
    prob = cs.astype(np.float64) / np.maximum(cs.sum(axis=1, keepdims=True), 1e-8)

    uncertain, disagree = window_flags(prob, imu_pred, cam)
    on_w = camera_on_windows(uncertain, disagree)
    on = expand_on(len(gt_v), on_w)

    base = {
        "seed": seed,
        "split": split,
        "sbj": int(imu["val_subjects"][0]),
        "d_all": float(on.mean() * 100),
        "d_activity": float(on[gt_v > 0].mean() * 100) if np.any(gt_v > 0) else 0.0,
        "d_windows": float(on_w.mean() * 100),
        "uncertain_windows": float(uncertain.mean() * 100),
        "n_windows": int(len(on_w)),
        "n_val": int(len(gt_v)),
    }
    out = {}
    for head in HEADS:
        hp = hopfield_preds(split_dir, head, imu, i3d, device, force)
        hp_v = hp[idx].astype(np.int64)
        dyn = np.where(on, hp_v, imu_pred)
        rec = dict(base)
        rec["head"] = head
        for name, pred in (("imu", imu_pred), ("cam", cam), ("always_on", hp_v), ("dynamic", dyn)):
            for k, v in prf(gt_v, pred).items():
                rec[f"{name}_{k}"] = v
        rec["retention_F1"] = rec["dynamic_F1"] / rec["always_on_F1"] if rec["always_on_F1"] else 0.0
        rec["lift_F1"] = rec["dynamic_F1"] - rec["imu_F1"]
        out[head] = rec
    return out


def write_head(out_dir: Path, pairing: str, head: str, rows: list[dict]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_seed = {str(s): {k: ms([r for r in rows if r["seed"] == s], k) for k in MEAN_KEYS} for s in (1, 2, 3)}
    summary = {
        "n_folds": len(rows),
        "pairing": PAIRINGS[pairing]["label"],
        "head": f"{head} Hopfield",
        "k": K,
        "entropy_thr": ENT_THR,
        "max_prob_thr": MAXP_THR,
        "fusion_when_on": f"{head} Hopfield (re-inferred model.pt)",
        "protocol": "same trigger k=4 entropy>0.05 or mean maxp<0.90; 18-class activity-frame P/R/F1; 54 LOSO",
        "means_54": {k: ms(rows, k) for k in MEAN_KEYS},
        "means_by_seed": by_seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", choices=("bf1", "bf2"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force-infer", action="store_true")
    ap.add_argument("--max-folds", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"pairing={args.pairing} device={device}", flush=True)

    rows = {h: [] for h in HEADS}
    n = 0
    for seed in (1, 2, 3):
        for split in range(1, 19):
            n += 1
            if args.max_folds and n > args.max_folds:
                break
            recs = run_split(args.pairing, seed, split, device, args.force_infer)
            for head, rec in recs.items():
                rows[head].append(rec)
                print(
                    f"{head:6s} seed {seed} sbj_{rec['sbj']:02d}  d {rec['d_windows']:5.1f}%  "
                    f"IMU {rec['imu_F1']:5.2f}  dyn {rec['dynamic_F1']:5.2f}  "
                    f"AO {rec['always_on_F1']:5.2f}",
                    flush=True,
                )
        if args.max_folds and n > args.max_folds:
            break

    cfg = PAIRINGS[args.pairing]
    summaries = {}
    for head in HEADS:
        out = cfg["root"] / f"duty_cycle_{args.pairing}_{head}_hp_loso54"
        summaries[head] = write_head(out, args.pairing, head, rows[head])
        print(f"\n=== {args.pairing} {head} HP 54-fold mean ± std ===")
        for k in MEAN_KEYS:
            m = summaries[head]["means_54"][k]
            print(f"{k:20s} {m['mean']:7.2f} ± {m['std']:.2f}")
        print("wrote", out, flush=True)

    side = ROOT / f"duty_cycle_{args.pairing}_hopfield_loso54.json"
    side.write_text(json.dumps({"pairing": args.pairing, "heads": summaries}, indent=2))
    print("wrote", side, flush=True)


if __name__ == "__main__":
    main()
