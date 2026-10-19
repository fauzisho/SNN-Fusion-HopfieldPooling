#!/usr/bin/env python3
"""SIM A single-modality baselines on the native and aligned grids, raw and 10 s vote.

Scores the frozen frame predictions of Attend-and-Discriminate, inertial TriDet and
camera TriDet (``ad_act``: A&D decided over its 18 activity probabilities only, the
rule the IMU SNN uses; the other rows may predict NULL, which counts as an error) with the same window sets and vote rules as
results_sima_fusion_{native,aligned}_vote10.csv. Camera TriDet is included as a
consistency check against those files. Nothing is retrained.
"""

import csv
import statistics
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

P.add_to_path(P.IMU_SNN_DIR)
from pipeline_inertial import labels_from_json  # noqa: E402

FEAT = P.IMU_FEAT_DIR
ANN = P.ANN_DIR
CACHE = P.SIMB_CACHE
BF1, BF2 = P.SIMA_BF1, P.SIMA_BF2
OUT = P.RESULTS_DIR / "results_sima_single_modality_vote10.csv"
NUM, NULL = 18, 18
LABELS = list(range(NUM))
WINDOW, STRIDE, FILTER = 2.0, 1.0, 10.0
GAP, KERNEL = 2.5, 5


def prf(y, p):
    return {
        "Acc": accuracy_score(y, p) * 100,
        "P": precision_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "R": recall_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "F1": f1_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
    }


def vote_native(pred, mask):
    """+-5 s on the 1 s grid, inside contiguous activity runs (same as the native fusion run)."""
    out = pred.copy()
    half = int(round(FILTER / STRIDE)) // 2
    i, n = 0, len(pred)
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i == n:
            break
        j = i + 1
        while j < n and mask[j]:
            j += 1
        seq = pred[i:j]
        sm = np.empty_like(seq)
        for k in range(len(seq)):
            lo, hi = max(0, k - half), min(len(seq), k + half + 1)
            sm[k] = int(np.bincount(seq[lo:hi]).argmax())
        out[i:j] = sm
        i = j
    return out


def vote_aligned(preds, times):
    """Five 2 s windows, restarted at gaps > 2.5 s (same as the aligned fusion run)."""
    order = np.argsort(times, kind="mergesort")
    seq, t = preds[order], times[order]
    out = seq.copy()
    n = len(seq)
    breaks = np.flatnonzero(np.diff(t) > GAP) if n else np.array([], dtype=int)
    half = KERNEL // 2
    for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks + 1, n]):
        span = seq[a:b]
        m = len(span)
        sm = np.empty(m, dtype=np.int64)
        for j in range(m):
            left, right = max(0, j - half), min(m, j + half + 1)
            if right - left < KERNEL:
                if left == 0:
                    right = min(m, KERNEL)
                else:
                    left = max(0, m - KERNEL)
            sm[j] = int(np.bincount(span[left:right]).argmax())
        out[a:b] = sm
    voted = np.empty_like(preds)
    voted[order] = out
    return voted


def to_activity(pred):
    return np.where(pred == 0, NULL, pred - 1).astype(np.int64)


def subject_arrays(path, sbj, activity_argmax=False):
    z = np.load(path)
    mask = (z["subject_ids"] == sbj) & z["is_val"]
    if activity_argmax:
        # argmax over the 18 activity probabilities, no NULL output (same rule as the IMU SNN)
        preds = np.asarray(z["class_scores"][mask]).argmax(axis=1).astype(np.int64) + 1
    else:
        preds = np.asarray(z["preds"][mask], dtype=np.int64)
    return preds, np.asarray(z["gt"][mask], dtype=np.int64)


MODELS = ("ad", "ad_act", "tridet_imu", "camera")
rows = []
n_native = n_aligned = 0
for seed in (1, 2, 3):
    for sbj in range(18):
        fold = f"seed_{seed}/split_{sbj+1}"
        frames = {
            "ad": subject_arrays(BF1 / fold / "ad_frames.npz", sbj),
            "ad_act": subject_arrays(BF1 / fold / "ad_frames.npz", sbj, activity_argmax=True),
            "tridet_imu": subject_arrays(BF2 / fold / "imu_frames.npz", sbj),
            "camera": subject_arrays(BF2 / fold / "i3d_frames.npz", sbj),
        }
        row = {"seed": seed, "sbj": sbj}

        # native grid
        n_win = np.load(FEAT / f"sbj_{sbj}.npy", mmap_mode="r").shape[0]
        y_all = labels_from_json(ANN / f"wear_split_{sbj+1}.json", sbj, n_win, WINDOW, STRIDE, 0.6)
        mask = y_all >= 0
        n_cam = len(frames["camera"][0])
        idx = np.clip(np.rint((np.arange(n_win) * STRIDE + WINDOW / 2.0) * 50.0).astype(np.int64), 0, n_cam - 1)
        row["n_native"] = int(mask.sum())
        n_native += int(mask.sum())
        for m in MODELS:
            pred = to_activity(frames[m][0][idx])
            for tag, pp in (("none", pred[mask]), ("vote", vote_native(pred, mask)[mask])):
                for k, v in prf(y_all[mask], pp).items():
                    row[f"native_{m}_{tag}_{k}"] = v

        # aligned grid
        blob = torch.load(CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt",
                          map_location="cpu", weights_only=False)
        y = np.asarray(blob["labels"], dtype=np.int64)
        t = np.asarray(blob["times"], dtype=np.float64)
        idx = np.clip(np.rint(t * 50.0).astype(np.int64), 0, n_cam - 1)
        for m in MODELS:
            if not np.array_equal(frames[m][1][idx] - 1, y):
                raise SystemExit(f"label mismatch {m} seed {seed} sbj {sbj}")
        row["n_aligned"] = len(y)
        n_aligned += len(y)
        for m in MODELS:
            pred = to_activity(frames[m][0][idx])
            for tag, pp in (("none", pred), ("vote", vote_aligned(pred, t))):
                for k, v in prf(y, pp).items():
                    row[f"aligned_{m}_{tag}_{k}"] = v
        rows.append(row)
        print(f"seed {seed} sbj {sbj} native {row['n_native']} aligned {row['n_aligned']}", flush=True)

assert n_native == 95439 and n_aligned == 45243, (n_native, n_aligned)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("FOLDS", len(rows), "NATIVE", n_native, "ALIGNED", n_aligned)
for metric in ("F1", "Acc"):
    print(metric)
    for m in MODELS:
        cells = []
        for grid in ("native", "aligned"):
            for tag in ("none", "vote"):
                v = [r[f"{grid}_{m}_{tag}_{metric}"] for r in rows]
                cells.append(f"{statistics.mean(v):.2f}±{statistics.stdev(v):.2f}")
        print(f"  {m:11} " + "  ".join(cells))
print("wrote", OUT)
