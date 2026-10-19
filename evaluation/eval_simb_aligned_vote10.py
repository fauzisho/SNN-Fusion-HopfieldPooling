#!/usr/bin/env python3
"""Table I, SIM B rows on the aligned windows: raw decisions and the 10 s vote.

Reads the held-out aligned features (``aligned_val_with_times.pt``, written by
``simb/build_aligned_cache.py``) and the trained heads in ``P.SIMB_HEADS``, and writes
one row per fold (54) with Acc / macro P / R / F1 for score-pick, single and double
Hopfield pooling, the oracle and both encoders, with and without the vote.

Vote: five consecutive aligned windows (2 s apart, about 10 s), restarted at gaps
> 2.5 s. Macro scores are over all 18 classes. Nothing is retrained.
"""
import csv, statistics
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)
from dvs_imu_fusion_data import NUM_CLASSES  # noqa: E402
from train_hopfield_dvs_imu_fusion import (  # noqa: E402
    DoubleHopfieldDVSIMU, SingleHopfieldDVSIMU, oracle_pick, predict_double_on_subject, score_pick,
)

ROOT = P.SIMB_HEADS
CACHE = P.SIMB_CACHE
OUT = P.RESULTS_DIR / "results_simb_fusion_vote10.csv"
GAP = 2.5
KERNEL = 5

def prf(y, p):
    labels = list(range(NUM_CLASSES))
    return {
        "Acc": accuracy_score(y, p) * 100,
        "P": precision_score(y, p, average="macro", zero_division=0, labels=labels) * 100,
        "R": recall_score(y, p, average="macro", zero_division=0, labels=labels) * 100,
        "F1": f1_score(y, p, average="macro", zero_division=0, labels=labels) * 100,
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

def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)
folds = sorted(
    p for p in ROOT.glob("seed_*/sbj_*")
    if (p / "single_hopfield.pt").is_file() and (p / "double_hopfield.pt").is_file()
)
print("folds", len(folds), flush=True)
rows = []
for i, fold in enumerate(folds):
    seed = int(fold.parent.name.split("_")[1])
    sbj = int(fold.name.split("_")[1])
    cache = CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt"
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    from train_hopfield_dvs_imu_fusion import SubjectAligned
    subj = SubjectAligned(
        subject_id=int(blob["subject_id"]),
        is_val=True,
        dvs_feat=np.asarray(blob["dvs_feat"]),
        imu_feat=np.asarray(blob["imu_feat"]),
        labels=np.asarray(blob["labels"]),
        times=np.asarray(blob["times"]),
    )
    y = subj.labels.astype(np.int64)
    dvs = subj.dvs_feat[:, :NUM_CLASSES].argmax(1).astype(np.int64)
    imu = subj.imu_feat[:, :NUM_CLASSES].argmax(1).astype(np.int64)
    pick = score_pick(subj.dvs_feat, subj.imu_feat).astype(np.int64)
    orc = oracle_pick(y, dvs, imu).astype(np.int64)
    single = SingleHopfieldDVSIMU().to(device)
    single.load_state_dict(load_state(fold / "single_hopfield.pt", device))
    single.eval()
    with torch.no_grad():
        sp = single(
            torch.tensor(subj.dvs_feat, dtype=torch.float32, device=device),
            torch.tensor(subj.imu_feat, dtype=torch.float32, device=device),
        ).argmax(1).cpu().numpy().astype(np.int64)
    double = DoubleHopfieldDVSIMU().to(device)
    double.load_state_dict(load_state(fold / "double_hopfield.pt", device))
    double.eval()
    dp = predict_double_on_subject(double, subj, seq_len=5, device=device)
    if np.any(dp < 0):
        raise RuntimeError(f"unfilled double preds {fold}")
    methods = {
        "dvs": dvs, "imu": imu, "score_pick": pick,
        "single_hp": sp, "double_hp": dp, "oracle": orc,
    }
    row = {"seed": seed, "sbj": sbj, "n": int(len(y))}
    for name, pred in methods.items():
        raw = prf(y, pred)
        voted = prf(y, vote_10s(pred, subj.times))
        for k, v in raw.items():
            row[f"{name}_{k}"] = v
        for k, v in voted.items():
            row[f"{name}_vote_{k}"] = v
    rows.append(row)
    print(
        f"[{i+1}/{len(folds)}] seed {seed} sbj {sbj} "
        f"double {row['double_hp_F1']:.2f} -> {row['double_hp_vote_F1']:.2f}",
        flush=True,
    )

rows.sort(key=lambda r: (r["seed"], r["sbj"]))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT, "bytes", OUT.stat().st_size)

def ms(key):
    xs = [r[key] for r in rows]
    return statistics.mean(xs), statistics.stdev(xs)

print("\nSUMMARY n", len(rows))
for name in ["dvs", "imu", "score_pick", "single_hp", "double_hp", "oracle"]:
    for tag, label in (("", "none"), ("_vote", "10s")):
        acc, asd = ms(f"{name}{tag}_Acc")
        p, ps = ms(f"{name}{tag}_P")
        r, rs = ms(f"{name}{tag}_R")
        f1, fs = ms(f"{name}{tag}_F1")
        print(f"{name:12} {label:4} Acc {acc:.2f}±{asd:.2f}  P {p:.2f}±{ps:.2f}  R {r:.2f}±{rs:.2f}  F1 {f1:.2f}±{fs:.2f}")
