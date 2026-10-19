#!/usr/bin/env python3
"""SIM B IMU-gated duty cycle on frozen DVS+IMU fusion, 54 LOSO folds.

Same trigger as SIM A (not retuned):
  k = 4 consecutive aligned 2 s windows
  uncertain iff IMU entropy > 0.05 or IMU max-prob < 0.90
  ON: last k uncertain, or disagreement while on
  OFF: last k certain and IMU / DVS labels agree

Home = IMU SNN always on. Dual stream while the event camera is on is one of:
  score-pick, single Hopfield pooling, double Hopfield pooling.

  python3 simb/eval_simb_duty_cycle_loso54.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)

from dvs_imu_fusion_data import FEAT_DIM, NUM_CLASSES, SubjectAligned  # noqa: E402
from train_hopfield_dvs_imu_fusion import (  # noqa: E402
    DoubleHopfieldDVSIMU,
    SingleHopfieldDVSIMU,
    predict_double_on_subject as predict_double_float,
    score_pick,
)

K = 4
ENT_THR = 0.05
MAXP_THR = 0.90
SEQ_LEN = 5
CACHE = P.SIMB_CACHE
FLOAT_ROOT = P.SIMB_HEADS

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


@dataclass(frozen=True)
class Head:
    key: str
    kind: str  # score_pick | float_single | float_double
    ckpt_root: Path | None = None
    ckpt_name: str | None = None
    out_dir: Path | None = None
    label: str = ""


def all_heads() -> list[Head]:
    heads = [
        Head("score_pick", "score_pick", label="score-pick", out_dir=FLOAT_ROOT / "duty_cycle_simb_score_pick_loso54"),
        Head(
            "single_hp",
            "float_single",
            ckpt_root=FLOAT_ROOT,
            ckpt_name="single_hopfield.pt",
            label="float single Hopfield",
            out_dir=FLOAT_ROOT / "duty_cycle_simb_single_hp_loso54",
        ),
        Head(
            "double_hp",
            "float_double",
            ckpt_root=FLOAT_ROOT,
            ckpt_name="double_hopfield.pt",
            label="float double Hopfield",
            out_dir=FLOAT_ROOT / "duty_cycle_simb_double_hp_loso54",
        ),
    ]
    return heads




def prf(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = gt.astype(np.int64)
    p = np.clip(pred.astype(np.int64), 0, NUM_CLASSES - 1)
    return {
        "acc": float(accuracy_score(y, p) * 100),
        "P": float(precision_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "R": float(recall_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "F1": float(f1_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100),
        "n": int(len(y)),
    }


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


def ms(rows: list[dict], key: str) -> dict[str, float]:
    xs = [float(r[key]) for r in rows]
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
    }


def load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_val(seed: int, sbj: int) -> SubjectAligned:
    path = CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    times = np.asarray(blob["times"])
    order = np.argsort(times)
    return SubjectAligned(
        subject_id=int(blob["subject_id"]),
        is_val=True,
        dvs_feat=np.asarray(blob["dvs_feat"])[order],
        imu_feat=np.asarray(blob["imu_feat"])[order],
        labels=np.asarray(blob["labels"])[order],
        times=times[order],
    )


@torch.no_grad()
def infer_head(head: Head, subj: SubjectAligned, seed: int, sbj: int, device: torch.device) -> np.ndarray:
    dvs, imu = subj.dvs_feat, subj.imu_feat
    if head.kind == "score_pick":
        return score_pick(dvs, imu).astype(np.int64)
    ckpt = head.ckpt_root / f"seed_{seed}" / f"sbj_{sbj}" / head.ckpt_name
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if head.kind == "float_single":
        model = SingleHopfieldDVSIMU(FEAT_DIM, hidden=64).to(device)
        model.load_state_dict(load_state(ckpt, device))
        model.eval()
        logits = model(
            torch.tensor(dvs, dtype=torch.float32, device=device),
            torch.tensor(imu, dtype=torch.float32, device=device),
        )
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    if head.kind == "float_double":
        model = DoubleHopfieldDVSIMU(in_dim=FEAT_DIM, hidden=128).to(device)
        model.load_state_dict(load_state(ckpt, device))
        model.eval()
        preds = predict_double_float(model, subj, seq_len=SEQ_LEN, device=device)
        if np.any(preds < 0):
            raise RuntimeError(f"unfilled double HP {ckpt}")
        return preds.astype(np.int64)
    raise RuntimeError(head.kind)


def run_fold(heads: list[Head], seed: int, sbj: int, device: torch.device) -> dict[str, dict]:
    subj = load_val(seed, sbj)
    imu_prob = subj.imu_feat[:, :NUM_CLASSES]
    maxp = subj.imu_feat[:, NUM_CLASSES]
    ent = subj.imu_feat[:, NUM_CLASSES + 1]
    imu_pred = imu_prob.argmax(axis=1).astype(np.int64)
    cam_pred = subj.dvs_feat[:, :NUM_CLASSES].argmax(axis=1).astype(np.int64)
    uncertain = (ent > ENT_THR) | (maxp < MAXP_THR)
    disagree = imu_pred != cam_pred
    on = camera_on_windows(uncertain, disagree)
    gt = subj.labels.astype(np.int64)
    base = {
        "seed": seed,
        "sbj": sbj,
        "split": sbj + 1,
        "d_windows": float(on.mean() * 100),
        "d_all": float(on.mean() * 100),
        "d_activity": float(on.mean() * 100),
        "uncertain_windows": float(uncertain.mean() * 100),
        "n_windows": int(len(on)),
        "n_val": int(len(gt)),
    }
    out = {}
    for head in heads:
        fuse = infer_head(head, subj, seed, sbj, device)
        dyn = np.where(on, fuse, imu_pred)
        rec = dict(base)
        rec["head"] = head.key
        rec["label"] = head.label
        for name, pred in (("imu", imu_pred), ("cam", cam_pred), ("always_on", fuse), ("dynamic", dyn)):
            for k, v in prf(gt, pred).items():
                rec[f"{name}_{k}"] = v
        rec["retention_F1"] = rec["dynamic_F1"] / rec["always_on_F1"] if rec["always_on_F1"] else 0.0
        rec["lift_F1"] = rec["dynamic_F1"] - rec["imu_F1"]
        out[head.key] = rec
    return out


def write_head(head: Head, rows: list[dict]) -> dict:
    head.out_dir.mkdir(parents=True, exist_ok=True)
    with (head.out_dir / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_seed = {
        str(s): {k: ms([r for r in rows if r["seed"] == s], k) for k in MEAN_KEYS} for s in (1, 2, 3)
    }
    summary = {
        "n_folds": len(rows),
        "pairing": "SIM B IMU SNN + Vision SNN",
        "head": head.label,
        "k": K,
        "entropy_thr": ENT_THR,
        "max_prob_thr": MAXP_THR,
        "fusion_when_on": head.label,
        "protocol": "same trigger k=4 entropy>0.05 or maxp<0.90; aligned 2s windows; 18-class P/R/F1; 54 LOSO",
        "means_54": {k: ms(rows, k) for k in MEAN_KEYS},
        "means_by_seed": by_seed,
    }
    (head.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    heads = all_heads()
    print(f"SIM B duty-cycle n_heads={len(heads)} device={device}")
    rows_by_head: dict[str, list[dict]] = {h.key: [] for h in heads}
    for seed in (1, 2, 3):
        for sbj in range(18):
            print(f"  seed {seed} sbj_{sbj}", flush=True)
            fold = run_fold(heads, seed, sbj, device)
            for key, rec in fold.items():
                rows_by_head[key].append(rec)
    payload = {
        "pairing": "simb",
        "family": "float",
        "heads": {},
    }
    for head in heads:
        payload["heads"][head.key] = write_head(head, rows_by_head[head.key])
        m = payload["heads"][head.key]["means_54"]
        print(
            f"{head.key:22s} d {m['d_windows']['mean']:.1f}%  "
            f"IMU {m['imu_F1']['mean']:.2f}  dyn {m['dynamic_F1']['mean']:.2f}  "
            f"AO {m['always_on_F1']['mean']:.2f}  lift {m['lift_F1']['mean']:+.2f}  "
            f"ret {m['retention_F1']['mean']:.3f}"
        )
    part = P.RESULTS_DIR / "duty_cycle_simb_float_loso54.json"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(payload, indent=2))
    print("wrote", part)


if __name__ == "__main__":
    main()
