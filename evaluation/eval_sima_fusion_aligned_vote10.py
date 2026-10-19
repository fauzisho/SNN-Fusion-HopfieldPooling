#!/usr/bin/env python3
"""Table I, SIM A fusion rows (BF1, BF2) on the aligned windows, raw and 10 s vote.

Reads the frozen 50 Hz frame dumps and trained heads of each SIM A fold and scores
them at the frames of the 45,243 aligned windows (window times from the SIM B
feature cache). Score-pick uses the raw frame predictions (no vote on A&D); single
and double Hopfield pooling use the one-hot "soft" scores; the oracle is WEAR's
``combine_predictions``. A NULL prediction counts as an error. Nothing is retrained.
"""
import csv
import statistics
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch

P.add_to_path(P.SIMA_DIR)
from aggregate_ad_fusion_map import (  # noqa: E402
    HopfieldLateFusion,
    TriDetDoubleHopfieldFusion,
    combine_predictions,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CACHE = P.SIMB_CACHE
BF1 = P.SIMA_BF1
BF2 = P.SIMA_BF2
OUT = P.RESULTS_DIR / "results_sima_fusion_aligned_vote10.csv"
NUM = 18
NULL = 18
LABELS = list(range(NUM))
GAP = 2.5
KERNEL = 5
DEVICE = torch.device("cpu")


def prf(y, p):
    return {
        "Acc": accuracy_score(y, p) * 100,
        "P": precision_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "R": recall_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "F1": f1_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
    }


def vote_10s(preds, times, kernel=KERNEL, gap=GAP):
    preds = np.asarray(preds, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    order = np.argsort(times, kind="mergesort")
    seq = preds[order]
    t = times[order]
    out = seq.copy()
    n = len(seq)
    breaks = np.flatnonzero(np.diff(t) > gap) if n else np.array([], dtype=int)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, n]
    half = kernel // 2
    for a, b in zip(starts, ends):
        span = seq[a:b]
        m = len(span)
        sm = np.empty(m, dtype=np.int64)
        for j in range(m):
            left = max(0, j - half)
            right = min(m, j + half + 1)
            if right - left < kernel:
                if left == 0:
                    right = min(m, kernel)
                else:
                    left = max(0, m - kernel)
            sm[j] = int(np.bincount(span[left:right]).argmax())
        out[a:b] = sm
    voted = np.empty_like(preds)
    voted[order] = out
    return voted


def to_activity(pred):
    pred = np.asarray(pred, dtype=np.int64)
    return np.where(pred == 0, NULL, pred - 1).astype(np.int64)


def subject_arrays(path, sbj):
    z = np.load(path)
    mask = (z["subject_ids"] == sbj) & z["is_val"]
    return (
        np.asarray(z["preds"][mask], dtype=np.int64),
        np.asarray(z["scores"][mask], dtype=np.float32),
        np.asarray(z["gt"][mask], dtype=np.int64),
    )


def soft_from(preds, scores):
    soft = np.full((len(preds), NUM), 1.0 / NUM, dtype=np.float32)
    ok = (preds >= 1) & (preds <= NUM)
    soft[ok] = 0.0
    soft[ok, preds[ok] - 1] = np.maximum(scores[ok], 1e-6)
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), 1e-8)
    return soft


def enrich(soft, scores):
    conf = scores.astype(np.float32).reshape(-1, 1)
    return np.concatenate([soft, conf, soft.max(axis=1, keepdims=True)], axis=1)


def single_at(ckpt, imu_soft, cam_soft):
    model = HopfieldLateFusion().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    out = np.empty(len(imu_soft), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(imu_soft), 2048):
            sl = slice(start, start + 2048)
            logits = model(
                torch.tensor(imu_soft[sl], dtype=torch.float32, device=DEVICE),
                torch.tensor(cam_soft[sl], dtype=torch.float32, device=DEVICE),
            )
            out[sl] = logits.argmax(dim=1).cpu().numpy() + 1
    return out


def double_at(ckpt, imu_feat, cam_feat, centers):
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    seq_len = int(state.get("seq_len", 100))
    hidden = int(state.get("hidden", 128))
    half = seq_len // 2
    model = TriDetDoubleHopfieldFusion(in_dim=imu_feat.shape[1], hidden=hidden).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    n = len(imu_feat)
    pred = np.zeros(len(centers), dtype=np.int64)
    valid = []
    valid_pos = []
    for i, c in enumerate(centers):
        start = int(c) - half
        if 0 <= start and start + seq_len <= n:
            valid.append(start)
            valid_pos.append(i)
    valid = np.asarray(valid, dtype=np.int64)
    valid_pos = np.asarray(valid_pos, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(valid), 128):
            sl = slice(start, start + 128)
            starts = valid[sl]
            imu_b = np.stack([imu_feat[s : s + seq_len] for s in starts])
            cam_b = np.stack([cam_feat[s : s + seq_len] for s in starts])
            logits = model(
                torch.tensor(imu_b, dtype=torch.float32, device=DEVICE),
                torch.tensor(cam_b, dtype=torch.float32, device=DEVICE),
            )
            pred[valid_pos[sl]] = logits.argmax(dim=1).cpu().numpy() + 1
    if len(valid_pos) < len(centers):
        known = np.zeros(len(centers), dtype=bool)
        known[valid_pos] = True
        for i, c in enumerate(centers):
            if known[i]:
                continue
            dist = np.abs(centers[known] - c)
            pred[i] = pred[np.flatnonzero(known)[int(np.argmin(dist))]]
    return pred


def oracle_at(imu_p, cam_p, gt, times):
    order = np.argsort(times, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    picked = combine_predictions(imu_p[order], cam_p[order], gt[order])
    return picked[inv]


def score_pair(imu_p, imu_s, cam_p, cam_s, gt, times, idx, single_ckpt, double_ckpt):
    imu_c, cam_c, gt_c = imu_p[idx], cam_p[idx], gt[idx]
    sp = np.where(imu_s[idx] >= cam_s[idx], imu_c, cam_c)
    ora = oracle_at(imu_c, cam_c, gt_c, times)
    imu_soft = soft_from(imu_p, imu_s)
    cam_soft = soft_from(cam_p, cam_s)
    single = single_at(single_ckpt, imu_soft[idx], cam_soft[idx])
    double = double_at(
        double_ckpt,
        enrich(imu_soft, imu_s),
        enrich(cam_soft, cam_s),
        idx,
    )
    raw = {
        "score_pick": to_activity(sp),
        "single_hp": to_activity(single),
        "double_hp": to_activity(double),
        "oracle": to_activity(ora),
    }
    return raw


rows = []
n_total = 0
for seed in (1, 2, 3):
    for sbj in range(18):
        blob = torch.load(
            CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt",
            map_location="cpu",
            weights_only=False,
        )
        y = np.asarray(blob["labels"], dtype=np.int64)
        t = np.asarray(blob["times"], dtype=np.float64)
        fold = f"seed_{seed}/split_{sbj+1}"
        cam_p, cam_s, cam_gt = subject_arrays(BF2 / fold / "i3d_frames.npz", sbj)
        ad_p, ad_s, ad_gt = subject_arrays(BF1 / fold / "ad_frames.npz", sbj)
        imu_p, imu_s, imu_gt = subject_arrays(BF2 / fold / "imu_frames.npz", sbj)
        idx = np.clip(np.rint(t * 50.0).astype(np.int64), 0, len(cam_p) - 1)
        if not (np.array_equal(cam_gt[idx] - 1, y) and np.array_equal(ad_gt[idx] - 1, y) and np.array_equal(imu_gt[idx] - 1, y)):
            raise SystemExit(f"label mismatch seed {seed} sbj {sbj}")
        bf1 = score_pair(
            ad_p, ad_s, cam_p, cam_s, cam_gt, t, idx,
            BF1 / fold / "hopfield_late_fusion/model.pt",
            BF1 / fold / "hopfield_double_fusion/model.pt",
        )
        bf2 = score_pair(
            imu_p, imu_s, cam_p, cam_s, cam_gt, t, idx,
            BF2 / fold / "hopfield_late_fusion/model.pt",
            BF2 / fold / "hopfield_double_fusion/model.pt",
        )
        row = {"seed": seed, "sbj": sbj, "n": len(y)}
        for prefix, pack in (("bf1", bf1), ("bf2", bf2)):
            for name, pred in pack.items():
                a = prf(y, pred)
                b = prf(y, vote_10s(pred, t))
                for k, v in a.items():
                    row[f"{prefix}_{name}_none_{k}"] = v
                for k, v in b.items():
                    row[f"{prefix}_{name}_vote_{k}"] = v
        rows.append(row)
        n_total += len(y)
        print(
            f"seed {seed} sbj {sbj} n {len(y)} "
            f"bf1 sp {row['bf1_score_pick_none_F1']:.1f}->{row['bf1_score_pick_vote_F1']:.1f} "
            f"bf2 sp {row['bf2_score_pick_none_F1']:.1f}->{row['bf2_score_pick_vote_F1']:.1f}",
            flush=True,
        )

assert n_total == 45243, n_total
fields = list(rows[0].keys())
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


def ms(key):
    vals = [r[key] for r in rows]
    return statistics.mean(vals), statistics.stdev(vals)

print("FOLDS", len(rows), "WINDOWS", n_total)
for prefix, title in (("bf1", "BF1"), ("bf2", "BF2")):
    print(title)
    for name in ("score_pick", "single_hp", "double_hp", "oracle"):
        for cond in ("none", "vote"):
            bits = []
            for k in ("Acc", "P", "R", "F1"):
                mu, sd = ms(f"{prefix}_{name}_{cond}_{k}")
                bits.append(f"{k} {mu:.2f}±{sd:.2f}")
            print(f"  {name:12} {cond:4} " + " ".join(bits))
print("wrote", OUT)
