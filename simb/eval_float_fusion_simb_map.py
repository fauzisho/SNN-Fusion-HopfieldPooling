#!/usr/bin/env python3
"""CPU eval of saved float DVS+IMU Hopfield heads: window P/R/F1 + WEAR mAP.

Paper: average mAP over tIoU 0.3-0.7 on the aligned windows after the same
five-window vote (double 69.54, single 63.21, score-pick 59.89, oracle 76.16).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)
from dvs_imu_fusion_data import NUM_CLASSES  # noqa: E402
from simb_common import (  # noqa: E402
    load_subj_cache,
    score_map,
)
from train_hopfield_dvs_imu_fusion import (  # noqa: E402
    DoubleHopfieldDVSIMU,
    SingleHopfieldDVSIMU,
    oracle_pick,
    predict_double_on_subject,
    score_pick,
)

ROOT = P.SIMB_HEADS
CACHE = P.SIMB_CACHE
OUT = P.RESULTS_DIR / "results_simb_fusion_pr_map.csv"
WINDOW_SEC = 2.0
SEQ_LEN = 5
MV = 5
TIOU = ("0.3", "0.4", "0.5", "0.6", "0.7")


def prf(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "Acc": accuracy_score(y, p) * 100,
        "P": precision_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100,
        "R": recall_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100,
        "F1": f1_score(y, p, average="macro", zero_division=0, labels=range(NUM_CLASSES)) * 100,
    }


@torch.no_grad()
def predict_single(model, subj, device) -> np.ndarray:
    logits = model(
        torch.tensor(subj.dvs_feat, dtype=torch.float32, device=device),
        torch.tensor(subj.imu_feat, dtype=torch.float32, device=device),
    )
    return logits.argmax(dim=1).cpu().numpy().astype(np.int64)


def main() -> None:
    device = torch.device("cpu")
    rows = []
    folds = sorted(ROOT.glob("seed_*/sbj_*"))
    folds = [f for f in folds if (f / "metrics.json").is_file()]
    print(f"folds {len(folds)}", flush=True)
    for i, fold in enumerate(folds):
        seed = int(fold.parts[-2].split("_")[1])
        sbj = int(fold.name.split("_")[1])
        cache = CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt"
        subj = load_subj_cache(cache)
        y = subj.labels.astype(np.int64)
        dvs = subj.dvs_feat[:, :NUM_CLASSES].argmax(1)
        imu = subj.imu_feat[:, :NUM_CLASSES].argmax(1)
        pick = score_pick(subj.dvs_feat, subj.imu_feat)
        orc = oracle_pick(y, dvs, imu)

        single = SingleHopfieldDVSIMU().to(device)
        single.load_state_dict(torch.load(fold / "single_hopfield.pt", map_location=device))
        single.eval()
        sp = predict_single(single, subj, device)

        double = DoubleHopfieldDVSIMU().to(device)
        double.load_state_dict(torch.load(fold / "double_hopfield.pt", map_location=device))
        double.eval()
        dp = predict_double_on_subject(double, subj, seq_len=SEQ_LEN, device=device)
        if np.any(dp < 0):
            raise RuntimeError(f"{fold}: unfilled double preds")

        video_id = f"sbj_{sbj}"
        methods = {
            "dvs_only": dvs,
            "imu_only": imu,
            "score_pick": pick,
            "single_hp": sp,
            "double_hp": dp,
            "oracle": orc,
        }
        row = {"seed": seed, "sbj": sbj}
        for name, pred in methods.items():
            rec = prf(y, pred)
            mp = score_map(pred, y, subj.times, WINDOW_SEC, video_id, MV)
            for k, v in rec.items():
                row[f"{name}_{k}"] = v
            for t in TIOU:
                row[f"{name}_mAP_{t}"] = mp[f"mAP_{t}"]
            row[f"{name}_mAP_avg"] = mp["mAP_avg"]
        rows.append(row)
        print(
            f"[{i+1}/{len(folds)}] seed {seed} sbj_{sbj} "
            f"double F1 {row['double_hp_F1']:.2f} mAP {row['double_hp_mAP_avg']:.2f}",
            flush=True,
        )

    fields = list(rows[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["seed"], r["sbj"])))
    print("wrote", OUT)

    def ms(key):
        xs = [r[key] for r in rows]
        return f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}"

    print("\nmeans n=", len(rows))
    for name in methods:
        print(
            name,
            "P", ms(f"{name}_P"),
            "R", ms(f"{name}_R"),
            "F1", ms(f"{name}_F1"),
            "mAP", " ".join(ms(f"{name}_mAP_{t}").split(" ± ")[0] for t in TIOU),
            "avg", ms(f"{name}_mAP_avg"),
        )


if __name__ == "__main__":
    main()
