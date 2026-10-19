#!/usr/bin/env python3
"""Table I, SIM A fusion rows (BF1, BF2) on the native windows, raw and 10 s vote.

Native windows: every 2 s / 1 s-stride window of the inertial timeline whose label
(>= 60 % overlap) is an activity, 95,439 over the 18 held-out subjects. Each window
is read at its centre frame of the 50 Hz SIM A dumps. Vote: +-5 windows inside each
contiguous activity run. A NULL prediction counts as an error. Nothing is retrained.
"""
import csv, statistics, sys
from pathlib import Path
import numpy as np, torch
import nwear_paths as P
P.add_to_path(P.SIMA_DIR, P.IMU_SNN_DIR)
from aggregate_ad_fusion_map import HopfieldLateFusion, TriDetDoubleHopfieldFusion, combine_predictions
from pipeline_inertial import labels_from_json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FEAT = P.IMU_FEAT_DIR
ANN = P.ANN_DIR
BF1, BF2 = P.SIMA_BF1, P.SIMA_BF2
OUT = P.RESULTS_DIR / "results_sima_fusion_native_vote10.csv"
NUM, NULL = 18, 18
LABELS = list(range(NUM))
WINDOW, STRIDE, FILTER = 2.0, 1.0, 10.0
DEVICE = torch.device("cpu")

def prf(y, p):
    return {
        "Acc": accuracy_score(y, p) * 100,
        "P": precision_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "R": recall_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
        "F1": f1_score(y, p, average="macro", zero_division=0, labels=LABELS) * 100,
    }

def vote_spans(pred, mask):
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

def to_activity(pred):
    pred = np.asarray(pred, dtype=np.int64)
    return np.where(pred == 0, NULL, pred - 1).astype(np.int64)

def subject_arrays(path, sbj):
    z = np.load(path)
    mask = (z["subject_ids"] == sbj) & z["is_val"]
    return (np.asarray(z["preds"][mask], dtype=np.int64), np.asarray(z["scores"][mask], dtype=np.float32), np.asarray(z["gt"][mask], dtype=np.int64))

def soft_from(preds, scores):
    soft = np.full((len(preds), NUM), 1.0 / NUM, dtype=np.float32)
    ok = (preds >= 1) & (preds <= NUM)
    soft[ok] = 0.0
    soft[ok, preds[ok] - 1] = np.maximum(scores[ok], 1e-6)
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), 1e-8)
    return soft

def enrich(soft, scores):
    return np.concatenate([soft, scores.astype(np.float32).reshape(-1, 1), soft.max(axis=1, keepdims=True)], axis=1)

def single_at(ckpt, imu_soft, cam_soft):
    model = HopfieldLateFusion().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    out = np.empty(len(imu_soft), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(imu_soft), 2048):
            sl = slice(start, start + 2048)
            logits = model(torch.tensor(imu_soft[sl]), torch.tensor(cam_soft[sl]))
            out[sl] = logits.argmax(1).cpu().numpy() + 1
    return out

def double_at(ckpt, imu_feat, cam_feat, centers):
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    seq_len, hidden = int(state.get("seq_len", 100)), int(state.get("hidden", 128))
    half = seq_len // 2
    model = TriDetDoubleHopfieldFusion(in_dim=imu_feat.shape[1], hidden=hidden).to(DEVICE)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    n = len(imu_feat)
    starts = centers.astype(np.int64) - half
    valid = np.flatnonzero((starts >= 0) & (starts + seq_len <= n))
    pred = np.zeros(len(centers), dtype=np.int64)
    offs = np.arange(seq_len)
    with torch.no_grad():
        for s in range(0, len(valid), 128):
            sl = valid[s:s+128]
            st = starts[sl]
            logits = model(torch.tensor(imu_feat[st[:, None] + offs]), torch.tensor(cam_feat[st[:, None] + offs]))
            pred[sl] = logits.argmax(1).cpu().numpy() + 1
    if len(valid) < len(centers):
        known = np.zeros(len(centers), dtype=bool); known[valid] = True
        known_i = np.flatnonzero(known)
        for i in np.flatnonzero(~known):
            pred[i] = pred[known_i[np.argmin(np.abs(centers[known_i] - centers[i]))]]
    return pred

def oracle_at(imu_p, cam_p, gt):
    return combine_predictions(imu_p, cam_p, gt)

rows, n_total, agree = [], 0, 0
for seed in (1, 2, 3):
    for sbj in range(18):
        n_win = np.load(FEAT / f"sbj_{sbj}.npy", mmap_mode="r").shape[0]
        y_all = labels_from_json(ANN / f"wear_split_{sbj+1}.json", sbj, n_win, WINDOW, STRIDE, 0.6)
        centers_t = np.arange(n_win) * STRIDE + WINDOW / 2.0
        fold = f"seed_{seed}/split_{sbj+1}"
        cam_p, cam_s, cam_gt = subject_arrays(BF2 / fold / "i3d_frames.npz", sbj)
        ad_p, ad_s, _ = subject_arrays(BF1 / fold / "ad_frames.npz", sbj)
        imu_p, imu_s, _ = subject_arrays(BF2 / fold / "imu_frames.npz", sbj)
        idx = np.clip(np.rint(centers_t * 50.0).astype(np.int64), 0, len(cam_p) - 1)
        mask = y_all >= 0
        agree += int(np.sum((cam_gt[idx][mask] - 1) == y_all[mask]))
        def pack(imu_pp, imu_ss, single_ckpt, double_ckpt):
            sp = np.where(imu_ss[idx] >= cam_s[idx], imu_pp[idx], cam_p[idx])
            ora = oracle_at(imu_pp[idx], cam_p[idx], cam_gt[idx])
            imu_soft, cam_soft = soft_from(imu_pp, imu_ss), soft_from(cam_p, cam_s)
            single = single_at(single_ckpt, imu_soft[idx], cam_soft[idx])
            double = double_at(double_ckpt, enrich(imu_soft, imu_ss), enrich(cam_soft, cam_s), idx)
            return {"score_pick": to_activity(sp), "single_hp": to_activity(single), "double_hp": to_activity(double), "oracle": to_activity(ora)}
        methods = {"camera": to_activity(cam_p[idx])}
        methods.update({f"bf1_{k}": v for k, v in pack(ad_p, ad_s, BF1/fold/"hopfield_late_fusion/model.pt", BF1/fold/"hopfield_double_fusion/model.pt").items()})
        methods.update({f"bf2_{k}": v for k, v in pack(imu_p, imu_s, BF2/fold/"hopfield_late_fusion/model.pt", BF2/fold/"hopfield_double_fusion/model.pt").items()})
        row = {"seed": seed, "sbj": sbj, "n": int(mask.sum())}
        y = y_all[mask]
        for name, pred in methods.items():
            voted = vote_spans(pred, mask)
            for tag, pp in (("none", pred[mask]), ("vote", voted[mask])):
                for k, v in prf(y, pp).items():
                    row[f"{name}_{tag}_{k}"] = v
        rows.append(row); n_total += int(mask.sum())
        print(f"seed {seed} sbj {sbj} n {int(mask.sum())} cam {row['camera_none_Acc']:.1f}->{row['camera_vote_Acc']:.1f} bf1s {row['bf1_single_hp_none_F1']:.1f}->{row['bf1_single_hp_vote_F1']:.1f}", flush=True)

print("WINDOWS", n_total, "GT_AGREE", agree, f"{100*agree/n_total:.2f}%")
fields = list(rows[0]);
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def ms(key):
    vals=[r[key] for r in rows]
    return statistics.mean(vals), statistics.stdev(vals)
names=["camera","bf1_score_pick","bf1_single_hp","bf1_double_hp","bf1_oracle","bf2_score_pick","bf2_single_hp","bf2_double_hp","bf2_oracle"]
for name in names:
    for cond in ("none","vote"):
        bits=[]
        for k in ("Acc","P","R","F1"):
            mu,sd=ms(f"{name}_{cond}_{k}")
            bits.append(f"{k} {mu:.2f}±{sd:.2f}")
        print(f"{name:16} {cond:4} "+" ".join(bits))
print("wrote", OUT)
