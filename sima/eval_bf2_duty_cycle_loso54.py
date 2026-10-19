#!/usr/bin/env python3
"""SIM A BF2 duty-cycle: all 54 LOSO folds (seeds 1–3 × 18 subjects).

Same frozen trigger as BF1 (not retuned on BF2):
  k = 4, 2 s windows (100 @ 50 Hz, stride 50)
  uncertain iff entropy > 0.05 or mean max-prob < 0.90
  Home = TriDet IMU (fusion dump key ad_preds)
  Dual-stream = saved score-pick (TriDet IMU + TriDet camera)
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
import nwear_paths as P

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = P.SIMA_BF2
OUT = ROOT / "duty_cycle_bf2_loso54"
NUM_CLASSES = 18
WIN = 100
STRIDE = 50
K = 4
ENT_THR = 0.05
MAXP_THR = 0.90

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


def run_split(seed: int, split: int) -> dict:
    split_dir = ROOT / f"seed_{seed}" / f"split_{split}"
    imu = np.load(split_dir / "imu_frames.npz")
    fus = np.load(split_dir / "score_oracle_fusion" / "fusion_preds.npz")
    idx = np.where(imu["is_val"])[0]
    gt_v = imu["gt"][idx].astype(np.int64)
    imu_pred = fus["ad_preds"][idx].astype(np.int64)
    cam = fus["i3d_preds"][idx].astype(np.int64)
    pick = fus["score_pick"][idx].astype(np.int64)
    cs = imu["class_scores"][idx]
    prob = cs.astype(np.float64) / np.maximum(cs.sum(axis=1, keepdims=True), 1e-8)

    uncertain, disagree = window_flags(prob, imu_pred, cam)
    on_w = camera_on_windows(uncertain, disagree)
    on = expand_on(len(gt_v), on_w)
    dyn = np.where(on, pick, imu_pred)

    rec = {
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
    for name, pred in (("imu", imu_pred), ("cam", cam), ("always_on", pick), ("dynamic", dyn)):
        for k, v in prf(gt_v, pred).items():
            rec[f"{name}_{k}"] = v
    rec["retention_F1"] = rec["dynamic_F1"] / rec["always_on_F1"] if rec["always_on_F1"] else 0.0
    rec["lift_F1"] = rec["dynamic_F1"] - rec["imu_F1"]
    return rec


def ms(rows: list[dict], key: str) -> dict[str, float]:
    xs = [float(r[key]) for r in rows]
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (1, 2, 3):
        for split in range(1, 19):
            rec = run_split(seed, split)
            rows.append(rec)
            print(
                f"seed {seed} sbj_{rec['sbj']:02d}  d {rec['d_windows']:5.1f}%  "
                f"IMU {rec['imu_F1']:5.2f}  dyn {rec['dynamic_F1']:5.2f}  "
                f"AO {rec['always_on_F1']:5.2f}",
                flush=True,
            )

    with (OUT / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_seed = {str(s): {k: ms([r for r in rows if r["seed"] == s], k) for k in MEAN_KEYS} for s in (1, 2, 3)}
    summary = {
        "n_folds": len(rows),
        "pairing": "BF2 TriDet IMU + TriDet camera",
        "k": K,
        "entropy_thr": ENT_THR,
        "max_prob_thr": MAXP_THR,
        "fusion_when_on": "score-pick (saved BF2 dump)",
        "means_54": {k: ms(rows, k) for k in MEAN_KEYS},
        "means_by_seed": by_seed,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== BF2 54-fold mean ± std ===")
    for k in MEAN_KEYS:
        m = summary["means_54"][k]
        print(f"{k:20s} {m['mean']:7.2f} ± {m['std']:.2f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
