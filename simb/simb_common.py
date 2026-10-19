"""Shared SIM B helpers: aligned-feature cache I/O, majority vote, macro P/R/F1, mAP."""

from __future__ import annotations

import functools
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)
from dvs_imu_fusion_data import ACTIVITY_CLASSES, NUM_CLASSES, SubjectAligned  # noqa: E402

TIOU = [0.3, 0.4, 0.5, 0.6, 0.7]
TIOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
WEAR_PP = P.WEAR_REPO / "postprocessing" / "utils.py"


@functools.lru_cache(maxsize=1)
def _load_anet():
    """ANETdetection from the WEAR repository (temporal action localisation mAP)."""
    spec = importlib.util.spec_from_file_location("wear_pp_utils", WEAR_PP)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.ANETdetection


def majority_vote(preds: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or len(preds) == 0:
        return preds.astype(np.int64, copy=True)
    out = np.empty_like(preds, dtype=np.int64)
    half = kernel // 2
    n = len(preds)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        if right - left < kernel:
            if left == 0:
                right = min(n, kernel)
            else:
                left = max(0, n - kernel)
        out[i] = int(np.bincount(preds[left:right].astype(np.int64)).argmax())
    return out


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    precs, recs, f1s = [], [], []
    for c in range(num_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "acc": acc,
        "P": float(np.mean(precs)),
        "R": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
        "n": float(len(y_true)),
    }


def merge_windows(
    labels: np.ndarray,
    times: np.ndarray,
    window_sec: float,
    video_id: str,
    scores: np.ndarray | None = None,
) -> dict:
    order = np.argsort(times)
    half = window_sec / 2.0
    f_ids, f_lab, f_t0, f_t1, f_sc = [], [], [], [], []
    cur = None
    for i in order:
        lab = int(labels[i])
        if lab < 0:
            continue
        t0 = float(times[i]) - half
        t1 = float(times[i]) + half
        sc = float(scores[i]) if scores is not None else 1.0
        if cur is None:
            cur = [lab, t0, t1, [sc]]
        elif lab == cur[0] and t0 <= cur[2] + 0.51:
            cur[2] = max(cur[2], t1)
            cur[3].append(sc)
        else:
            f_ids.append(video_id)
            f_lab.append(cur[0])
            f_t0.append(cur[1])
            f_t1.append(cur[2])
            f_sc.append(float(np.mean(cur[3])))
            cur = [lab, t0, t1, [sc]]
    if cur is not None:
        f_ids.append(video_id)
        f_lab.append(cur[0])
        f_t0.append(cur[1])
        f_t1.append(cur[2])
        f_sc.append(float(np.mean(cur[3])))
    return {
        "video-id": f_ids,
        "label": np.asarray(f_lab, dtype=np.float64),
        "t-start": np.asarray(f_t0, dtype=np.float64),
        "t-end": np.asarray(f_t1, dtype=np.float64),
        "score": np.asarray(f_sc, dtype=np.float64),
    }


def build_gt_json(video_id: str, labels: np.ndarray, times: np.ndarray, window_sec: float, path: Path) -> None:
    segs = merge_windows(labels, times, window_sec, video_id)
    anns = []
    duration = 0.0
    for lab, t0, t1 in zip(segs["label"], segs["t-start"], segs["t-end"]):
        cid = int(lab)
        anns.append(
            {
                "label": ACTIVITY_CLASSES[cid],
                "segment": [float(t0), float(t1)],
                "label_id": cid,
            }
        )
        duration = max(duration, float(t1))
    payload = {
        "label_dict": {name: i for i, name in enumerate(ACTIVITY_CLASSES)},
        "database": {
            video_id: {
                "subset": "Validation",
                "duration": duration,
                "annotations": anns,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def score_map(
    preds: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    window_sec: float,
    video_id: str,
    mv_kernel: int,
) -> dict[str, float]:
    voted = majority_vote(preds, mv_kernel)
    rec = macro_prf(labels, voted, NUM_CLASSES)
    pred_seg = merge_windows(voted, times, window_sec, video_id)
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        build_gt_json(video_id, labels, times, window_sec, gt_path)
        det = _load_anet()(str(gt_path), "validation", tiou_thresholds=TIOU, num_workers=2)
        v_map, _ = det.evaluate(pred_seg)
    v_map = np.asarray(v_map, dtype=np.float64)
    return {
        **rec,
        "mAP_avg": float(v_map.mean() * 100.0),
        **{f"mAP_{k}": float(v * 100.0) for k, v in zip(TIOU_KEYS, v_map)},
        "n_pred_segs": int(len(pred_seg["video-id"])),
        "mv_kernel": int(mv_kernel),
    }


def load_subj_cache(path: Path) -> SubjectAligned:
    blob = torch.load(path, map_location="cpu")
    return SubjectAligned(
        subject_id=int(blob["subject_id"]),
        is_val=True,
        dvs_feat=np.asarray(blob["dvs_feat"]),
        imu_feat=np.asarray(blob["imu_feat"]),
        labels=np.asarray(blob["labels"]),
        times=np.asarray(blob["times"]),
    )


def save_subj_cache(path: Path, subj: SubjectAligned) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "subject_id": subj.subject_id,
            "dvs_feat": subj.dvs_feat,
            "imu_feat": subj.imu_feat,
            "labels": subj.labels,
            "times": subj.times,
        },
        path,
    )


