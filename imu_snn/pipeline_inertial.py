#!/usr/bin/env python3
"""
EGRU classifier on WEAR dataset inertial feature .npy files with JSON annotation labels.

Follows the WEAR paper (Bock et al., 2023):
  - Features : inertial_feature/{rate}/{W}_frames_{S}_stride/sbj_*.npy
  - Labels   : annotations/{fps}/wear_split_{N}.json

Window parameters (imu_hz, video_fps, window_sec, stride_sec) are auto-detected
from the feature directory name, e.g. "10hz/12_frames_6_stride" → 10 Hz IMU,
12 fps video, 1.0 s window, 0.5 s stride.

Annotation files contain per-subject activity segments (start_sec, end_sec) and
a subset field ("Training"/"Validation") that encodes the cross-validation split.
Each wear_split_N.json corresponds to one leave-one-subject-out fold.

Usage (annotation-split based — paper method):
  python pipeline_inertial.py \
    --feat-dir C:/path/inertial_feature/10hz/12_frames_6_stride \
    --ann-dir  C:/path/annotations/12fps \
    --split 1

Usage (simple subject holdout):
  python pipeline_inertial.py \
    --feat-dir C:/path/inertial_feature/10hz/12_frames_6_stride \
    --ann-dir  C:/path/annotations/12fps \
    --val-subjects 3

Usage (CSV label fallback, no annotation dir):
  python pipeline_inertial.py \
    --feat-dir C:/path/inertial_feature/10hz/12_frames_6_stride \
    --csv-dir  C:/path/2nd-wear-dataset-challenge/train \
    --val-subjects 3
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Activity classes (WEAR dataset, 18 classes) ──────────────────────────────
ACTIVITY_CLASSES = [
    "jogging",
    "jogging (rotating arms)",
    "jogging (skipping)",
    "jogging (sidesteps)",
    "jogging (butt-kicks)",
    "stretching (triceps)",
    "stretching (lunging)",
    "stretching (shoulders)",
    "stretching (hamstrings)",
    "stretching (lumbar rotation)",
    "push-ups",
    "push-ups (complex)",
    "sit-ups",
    "sit-ups (complex)",
    "burpees",
    "lunges",
    "lunges (complex)",
    "bench-dips",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ACTIVITY_CLASSES)}
NUM_CLASSES = len(ACTIVITY_CLASSES)
NULL_CLASS_NAME = "null"

# IMU sampling rate → video fps (WEAR dataset convention)
RATE_TO_FPS: dict[str, int] = {"10hz": 12, "25hz": 30, "50hz": 60}
CSV_HZ = 50  # raw WEAR CSV is always 50 Hz


# ═══════════════════════════════════════════════════════════════════════════════
# Window parameter detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_window_params(feat_dir: Path, imu_hz_override: int | None = None,
                         video_fps_override: int | None = None) -> dict:
    """
    Parse IMU rate and window config from the feature directory path.

    Expected layout: .../inertial_feature/{rate}/{W}_frames_{S}_stride/
    Returns a dict with: imu_hz, video_fps, window_frames, stride_frames,
                         window_sec, stride_sec, T_imu.
    """
    config_name = feat_dir.name        # e.g. "12_frames_6_stride"
    rate_name   = feat_dir.parent.name  # e.g. "10hz"

    # IMU Hz
    if imu_hz_override is not None:
        imu_hz = imu_hz_override
    else:
        m_rate = re.match(r"(\d+)hz", rate_name.lower())
        if not m_rate:
            raise ValueError(
                f"Cannot parse IMU rate from folder {rate_name!r}. "
                "Use --imu-hz to specify explicitly."
            )
        imu_hz = int(m_rate.group(1))

    # Video fps
    if video_fps_override is not None:
        video_fps = video_fps_override
    else:
        fps_key = rate_name.lower()
        if fps_key not in RATE_TO_FPS:
            raise ValueError(
                f"No video fps mapping for rate folder {rate_name!r}. "
                "Use --video-fps to specify explicitly."
            )
        video_fps = RATE_TO_FPS[fps_key]

    # Window/stride in video frames
    m_cfg = re.match(r"(\d+)_frames_(\d+)_stride", config_name)
    if not m_cfg:
        raise ValueError(f"Cannot parse window config from folder {config_name!r}.")
    window_frames = int(m_cfg.group(1))
    stride_frames = int(m_cfg.group(2))

    window_sec = window_frames / video_fps
    stride_sec = stride_frames / video_fps
    T_imu      = round(imu_hz * window_sec)

    return {
        "imu_hz":        imu_hz,
        "video_fps":     video_fps,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "window_sec":    window_sec,
        "stride_sec":    stride_sec,
        "T_imu":         T_imu,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EGRU — exact copy from emmanuela/models/egru.py (best performing model).
# Copyright (c) 2023 Khaleelulla Khan Nazeer; incorporates work Copyright 2020 LMNT, Inc.
# Licensed under the Apache License, Version 2.0 (see event_snn/models/egru.py).
# ═══════════════════════════════════════════════════════════════════════════════

class _SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dz_du = dampening_factor * torch.maximum(
            1 - pseudo_derivative_support * torch.abs(inp),
            torch.zeros(1, device=grad_output.device),
        )
        return grad_output * dz_du, None, None


def _EGRUScript(
    training, zoneout_prob, dampening_factor, pseudo_derivative_support,
    inp, h0, kernel, recurrent_kernel, bias, recurrent_bias, thr, zoneout_mask,
):
    T  = inp.shape[0]
    h  = [torch.zeros_like(h0)]
    o  = [torch.zeros_like(h0)]
    y  = [h0]
    Wx = inp @ kernel + bias  # (T, B, 3H)

    for t in range(T):
        Rh = y[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, dim=1)
        vh = torch.chunk(Rh,    3, dim=1)

        z     = torch.sigmoid(vx[0] + vh[0])
        r     = torch.sigmoid(vx[1] + vh[1])
        g     = torch.tanh(   vx[2] + r * vh[2])
        cur_h = z * h[t] + (1 - z) * g

        if zoneout_prob:
            if training:
                cur_h = (cur_h - h[t]) * zoneout_mask[t] + h[t]
            else:
                cur_h = zoneout_prob * h[t] + (1 - zoneout_prob) * cur_h

        event = _SpikeFunction.apply(cur_h - thr, dampening_factor, pseudo_derivative_support)
        o.append(event)
        h.append(cur_h - event * thr)
        y.append(event * cur_h)

    y = torch.stack(y)
    h = torch.stack(h)
    o = torch.stack(o)

    tr_vals = torch.zeros_like(y)
    alpha   = 0.9
    for t in range(1, T + 1):
        tr_vals[t] = alpha * tr_vals[t - 1] + (1 - alpha) * y[t]

    return y, h, o, tr_vals


class _BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, zoneout):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.zoneout     = zoneout

    def _permute(self, x):
        return x.permute(1, 0, 2) if self.batch_first else x

    def _get_zoneout_mask(self, inp):
        if self.zoneout:
            mask = inp.new_empty(inp.shape[0], inp.shape[1], self.hidden_size)
            mask.bernoulli_(1.0 - self.zoneout)
        else:
            mask = inp.new_empty(0, 0, 0)
        return mask


class EGRU(_BaseRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
        dropout: float = 0.0,
        zoneout: float = 0.0,
        dampening_factor: float = 0.7,
        pseudo_derivative_support: float = 1.0,
        thr_mean: float = 0.3,
        weight_initialization_gain: float = 1.0,
    ):
        super().__init__(input_size, hidden_size, batch_first, zoneout)
        assert 0 < thr_mean < 1

        self.dropout                  = dropout
        self.weight_initialization_gain = weight_initialization_gain

        self.kernel           = nn.Parameter(torch.empty(input_size,  hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias             = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias   = nn.Parameter(torch.empty(hidden_size * 3))
        self._reset_parameters()

        self.dampening_factor = nn.Parameter(
            torch.tensor([dampening_factor]), requires_grad=False
        )
        self.pseudo_derivative_support = nn.Parameter(
            torch.tensor([pseudo_derivative_support]), requires_grad=False
        )

        beta_b = 3
        beta_a = beta_b * thr_mean / (1.0 - thr_mean)
        dist   = torch.distributions.beta.Beta(beta_a, beta_b)
        self.thr = nn.Parameter(dist.sample(torch.Size([hidden_size])))

    def _reset_parameters(self):
        g = self.weight_initialization_gain
        for name, param in self.named_parameters():
            if name in ("kernel", "recurrent_kernel"):
                nn.init.xavier_normal_(param, gain=g)
            elif name in ("bias", "recurrent_bias"):
                nn.init.zeros_(param)

    def forward(self, inp, state=None):
        inp = self._permute(inp)
        B   = inp.shape[1]
        h0  = inp.new_zeros(B, self.hidden_size) if state is None else state[0]
        self.thr.data.clamp_(0.0, 1.0)

        rc_drop = F.dropout(self.recurrent_kernel, self.dropout, self.training)
        y, h, o, trace = _EGRUScript(
            self.training, self.zoneout,
            self.dampening_factor, self.pseudo_derivative_support,
            inp.contiguous(), h0.contiguous(),
            self.kernel.contiguous(), rc_drop.contiguous(),
            self.bias.contiguous(), self.recurrent_bias.contiguous(),
            self.thr.contiguous(), self._get_zoneout_mask(inp).contiguous(),
        )
        return self._permute(y[1:])  # (B, T, H) if batch_first


class EGRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
        zoneout: float = 0.0,
        thr_mean: float = 0.3,
        dampening_factor: float = 0.7,
    ):
        super().__init__()
        self.egru = EGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            zoneout=zoneout,
            thr_mean=thr_mean,
            dampening_factor=dampening_factor,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits (B, C, T)"""
        out    = self.egru(x)          # (B, T, H)
        logits = self.head(out)        # (B, T, C)
        return logits.permute(0, 2, 1)  # (B, C, T)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_npy(path: Path, T_imu: int) -> tuple[np.ndarray, int]:
    """
    Load a sbj_*.npy feature file → (N, T, F), return (array, F).

    The npy stores flattened windows: shape (N, T*F).
    T is derived from T_imu; F (channels) is auto-detected.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D npy (N, T*F), got {arr.shape} in {path.name}")
    N, total = arr.shape
    if total % T_imu != 0:
        raise ValueError(
            f"{path.name}: feature dim {total} not divisible by T_imu={T_imu}. "
            "Check --feat-dir matches the window config, or set --imu-hz / --video-fps."
        )
    F = total // T_imu
    return arr.reshape(N, T_imu, F), F


def labels_from_json(
    ann_path: Path,
    subject_id: int,
    num_windows: int,
    window_sec: float,
    stride_sec: float,
    min_overlap: float = 0.6,
) -> np.ndarray:
    """
    Assign activity labels to npy windows from annotation JSON segments.

    Window i spans [i * stride_sec, i * stride_sec + window_sec].
    A label is assigned when the best-matching annotation segment covers
    >= min_overlap of the window.  Returns -1 for unlabeled/transition windows.
    """
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]

    key = f"sbj_{subject_id}"
    if key not in db:
        print(f"    Warning: {key} not found in {ann_path.name}, all windows = -1")
        return np.full(num_windows, -1, dtype=np.int64)

    anns   = db[key]["annotations"]
    labels = np.full(num_windows, -1, dtype=np.int64)

    for i in range(num_windows):
        win_s = i * stride_sec
        win_e = win_s + window_sec

        best_ratio = 0.0
        best_label = -1
        for ann in anns:
            seg_s, seg_e = ann["segment"]
            overlap = max(0.0, min(win_e, seg_e) - max(win_s, seg_s))
            ratio   = overlap / window_sec
            if ratio > best_ratio:
                best_ratio = ratio
                if ratio >= min_overlap:
                    # Prefer label_id field; fall back to CLASS_TO_IDX lookup
                    best_label = ann.get(
                        "label_id",
                        CLASS_TO_IDX.get(ann.get("label", ""), -1),
                    )

        labels[i] = best_label

    return labels


def labels_from_csv(
    csv_path: Path,
    num_windows: int,
    window_sec: float,
    stride_sec: float,
    min_label_ratio: float = 0.6,
) -> np.ndarray:
    """
    Derive per-window labels from a raw 50 Hz WEAR CSV.

    T_csv and S_csv (window/stride in raw samples) are computed from window_sec
    and stride_sec, matching the npy generation parameters.
    """
    T_csv = round(CSV_HZ * window_sec)
    S_csv = round(CSV_HZ * stride_sec)

    df = pd.read_csv(csv_path, low_memory=False)
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {csv_path}")

    lbl     = df["label"]
    n_rows  = len(df)
    labels: list[int] = []

    for start in range(0, n_rows - T_csv + 1, S_csv):
        window = lbl.iloc[start : start + T_csv].to_numpy()
        valid  = [
            str(x).strip() for x in window
            if isinstance(x, str) and str(x).strip().lower() not in ("", "null", "nan")
        ]
        if len(valid) < T_csv * min_label_ratio:
            labels.append(-1)
            continue
        top_label, top_count = Counter(valid).most_common(1)[0]
        if top_count / T_csv >= min_label_ratio and top_label in CLASS_TO_IDX:
            labels.append(CLASS_TO_IDX[top_label])
        else:
            labels.append(-1)

    n_derived = len(labels)
    diff      = n_derived - num_windows
    if diff == 0:
        pass
    elif abs(diff) <= 2:
        labels = labels[:num_windows] if diff > 0 else labels + [-1] * (-diff)
    else:
        raise ValueError(
            f"{csv_path.name}: CSV windowing → {n_derived} windows, npy → {num_windows}. "
            f"(T_csv={T_csv}, S_csv={S_csv}, rows={n_rows})"
        )

    return np.asarray(labels, dtype=np.int64)


def get_split_subjects(ann_path: Path) -> tuple[list[int], list[int]]:
    """
    Return (train_ids, val_ids) from subset field in a wear_split_N.json.

    Each split file is one fold of leave-one-subject-out cross-validation:
    the subject with subset="Validation" is the held-out test subject.
    """
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]

    train_ids: list[int] = []
    val_ids:   list[int] = []
    for key, info in db.items():
        m = re.match(r"sbj_(\d+)", key)
        if not m:
            continue
        sid    = int(m.group(1))
        subset = info.get("subset", "").lower()
        (val_ids if subset == "validation" else train_ids).append(sid)

    return sorted(train_ids), sorted(val_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization & helpers
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_per_channel(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = x.mean(axis=(0, 1), keepdims=True)
        std  = x.std( axis=(0, 1), keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def make_class_weights(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    cnt = torch.bincount(y, minlength=num_classes).float().to(device).clamp(min=1.0)
    w   = 1.0 / cnt
    return w * (num_classes / w.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Loss & inference
# ═══════════════════════════════════════════════════════════════════════════════

def ce_loss(logits: torch.Tensor, targets: torch.Tensor,
            weight: torch.Tensor | None = None) -> torch.Tensor:
    """logits (B, C, T), targets (B,) → CE on time-averaged logits."""
    return F.cross_entropy(logits.mean(dim=-1), targets, weight=weight)


def classify(logits: torch.Tensor) -> torch.Tensor:
    return logits.mean(dim=-1).argmax(dim=1)


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute unweighted macro-F1 over class indices [0, num_classes)."""
    f1_scores: list[float] = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def majority_vote_filter(
    y_pred: np.ndarray,
    sid: np.ndarray,
    stride_sec: float,
    filter_sec: float,
) -> np.ndarray:
    """Apply per-subject majority vote smoothing over a temporal window."""
    if filter_sec <= 0:
        return y_pred.copy()
    win = max(1, int(round(filter_sec / max(stride_sec, 1e-8))))
    half = win // 2
    out = y_pred.copy()
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        seq = y_pred[idx]
        smooth = np.empty_like(seq)
        for i in range(len(seq)):
            lo = max(0, i - half)
            hi = min(len(seq), i + half + 1)
            vals, cnts = np.unique(seq[lo:hi], return_counts=True)
            smooth[i] = vals[np.argmax(cnts)]
        out[idx] = smooth
    return out


def _segments_from_sequence(
    y: np.ndarray,
    sid: np.ndarray,
    window_sec: float,
    stride_sec: float,
    probs: np.ndarray | None = None,
) -> list[dict]:
    """
    Build temporal segments from per-window labels.
    Segment timing uses window start=i*stride and end=start+window_sec.
    """
    segs: list[dict] = []
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        ys = y[idx]
        ps = probs[idx] if probs is not None else None
        if len(ys) == 0:
            continue
        start_i = 0
        cur = int(ys[0])
        for i in range(1, len(ys) + 1):
            if i == len(ys) or int(ys[i]) != cur:
                end_i = i - 1
                t0 = start_i * stride_sec
                t1 = end_i * stride_sec + window_sec
                score = 1.0
                if ps is not None and cur >= 0:
                    score = float(ps[start_i : end_i + 1, cur].mean())
                segs.append({"sid": int(s), "label": cur, "t0": t0, "t1": t1, "score": score})
                if i < len(ys):
                    start_i = i
                    cur = int(ys[i])
    return segs


def _tiou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]))
    union = max(a["t1"], b["t1"]) - min(a["t0"], b["t0"])
    return 0.0 if union <= 0 else inter / union


def _ap_for_class(gt: list[dict], pred: list[dict], tiou_thr: float) -> float:
    gt_by_sid: dict[int, list[dict]] = {}
    for g in gt:
        gt_by_sid.setdefault(g["sid"], []).append(g)
    npos = len(gt)
    if npos == 0:
        return 0.0

    used: dict[int, set[int]] = {sid: set() for sid in gt_by_sid}
    pred_sorted = sorted(pred, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(pred_sorted), dtype=np.float64)
    fp = np.zeros(len(pred_sorted), dtype=np.float64)

    for i, p in enumerate(pred_sorted):
        cand = gt_by_sid.get(p["sid"], [])
        best_j = -1
        best_iou = 0.0
        for j, g in enumerate(cand):
            if j in used[p["sid"]]:
                continue
            iou = _tiou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= tiou_thr:
            tp[i] = 1.0
            used[p["sid"]].add(best_j)
        else:
            fp[i] = 1.0

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / max(npos, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def temporal_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sid: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    window_sec: float,
    stride_sec: float,
    tiou_thresholds: Iterable[float],
    exclude_class: int | None = None,
) -> tuple[dict[float, float], float]:
    gt_segments = _segments_from_sequence(y_true, sid, window_sec, stride_sec, probs=None)
    pred_segments = _segments_from_sequence(y_pred, sid, window_sec, stride_sec, probs=probs)

    class_ids = [c for c in range(num_classes) if c != exclude_class]
    ap_by_thr: dict[float, float] = {}
    for thr in tiou_thresholds:
        cls_aps = []
        for c in class_ids:
            gt_c = [g for g in gt_segments if g["label"] == c]
            pred_c = [p for p in pred_segments if p["label"] == c]
            cls_aps.append(_ap_for_class(gt_c, pred_c, float(thr)))
        ap_by_thr[float(thr)] = float(np.mean(cls_aps)) if cls_aps else 0.0
    avg_map = float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0
    return ap_by_thr, avg_map


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, weight) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = ce_loss(model(xb), yb, weight)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model, x_val, y_val, device, weight, num_classes: int
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    logits = model(x_val.to(device))
    loss   = ce_loss(logits, y_val.to(device), weight).item()
    probs_t = F.softmax(logits.mean(dim=-1), dim=1)
    probs   = probs_t.cpu().numpy()
    pred   = classify(logits).cpu().numpy()
    acc    = float((pred == y_val.numpy()).mean())
    f1     = macro_f1_score(y_val.numpy(), pred, num_classes=num_classes)
    return loss, acc, f1, pred, probs


def fit(
    x_train, y_train, x_val, y_val,
    *,
    in_features: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    hidden: int,
    dropout: float,
    zoneout: float,
    thr_mean: float,
    dampening_factor: float,
    use_class_weights: bool,
    num_classes: int,
    milestones: list[int],
    lr_gamma: float,
) -> tuple:
    model = EGRUClassifier(
        input_size=in_features,
        hidden_size=hidden,
        num_classes=num_classes,
        dropout=dropout,
        zoneout=zoneout,
        thr_mean=thr_mean,
        dampening_factor=dampening_factor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_gamma
    )
    weight = make_class_weights(y_train, num_classes, device) if use_class_weights else None
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    tr_losses, va_losses, va_accs, va_f1s = [], [], [], []
    for ep in range(epochs):
        tr_l = train_epoch(model, loader, optimizer, device, weight)
        va_l, va_a, va_f1, _, _ = evaluate(model, x_val, y_val, device, weight, num_classes)
        scheduler.step()
        tr_losses.append(tr_l)
        va_losses.append(va_l)
        va_accs.append(va_a)
        va_f1s.append(va_f1)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  epoch {ep+1:3d}/{epochs} | "
                f"train {tr_l:.4f} | val {va_l:.4f} | acc {va_a:.3f} | f1 {va_f1:.3f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

    return model, tr_losses, va_losses, va_accs, va_f1s


# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training(train_loss, val_loss, val_acc, out_path, milestones):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(train_loss, label="train CE", color="C0")
    ax1.plot(val_loss,   label="val CE",   color="C1")
    for m in milestones:
        ax1.axvline(m, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CE loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(val_acc, label="val acc", color="C2", linestyle="--")
    ax2.set_ylabel("Val accuracy")
    ax2.legend(loc="lower right")
    fig.suptitle("EGRU — inertial features (npy + annotation labels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, out_path):
    k  = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=5, color="black")
    fig.colorbar(im, ax=ax, label="Recall per class")
    fig.suptitle("Validation confusion matrix — EGRU inertial features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_architecture(model, in_features, T_imu, out_path):
    nh    = model.egru.hidden_size
    ncls  = model.head.out_features
    thr_m = model.egru.thr.mean().item()
    n_p   = sum(p.numel() for p in model.parameters())

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 24); ax.set_ylim(0, 5); ax.set_axis_off()

    def node(cx, cy, w, h, text, fc):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            linewidth=1.1, edgecolor="#37474f", facecolor=fc,
        ))
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=8.5, color="#212121", linespacing=1.25)

    def arr(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="#424242", lw=1.3))

    y = 2.5
    node(2.2,  y, 3.2, 2.2, f"Input\n(B, T={T_imu}, F={in_features})", "#bbdefb")
    arr(3.8, 4.8, y)
    node(6.5,  y, 3.4, 2.2,
         f"EGRU (emmanuela)\nbatch_first=True\nhidden={nh}\nthr≈{thr_m:.2f}", "#ffe0b2")
    arr(8.2, 9.2, y)
    node(10.8, y, 2.8, 1.8, f"Output\n(B, T, {nh})", "#c8e6c9")
    arr(12.2, 13.2, y)
    node(14.7, y, 2.8, 1.8, f"Linear\n({nh}→{ncls})\nper timestep", "#d1c4e9")
    arr(16.1, 17.1, y)
    node(19.0, y, 3.6, 2.4,
         f"Logits (B,{ncls},T)\nCE(mean_T)\nargmax(mean_T)", "#f8bbd9")

    ax.text(12, 4.55,
            f"EGRUClassifier — inertial features, T={T_imu}, F={in_features}",
            ha="center", fontsize=10, fontweight="600", color="#263238")
    ax.text(12, 0.45, f"Parameters: {n_p:,}",
            ha="center", fontsize=9, color="#546e7a")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    # Default paths for this project
    import nwear_paths as P  # imported here so the module also works without the repo on sys.path

    FEAT_DIR_DEFAULT = P.IMU_FEAT_DIR  # 2 s windows, 1 s stride, 50 Hz
    ANN_DIR_DEFAULT = P.ANN_DIR

    ap = argparse.ArgumentParser(
        description="EGRU on WEAR inertial feature .npy files + annotation JSON labels"
    )
    # ── Data ──────────────────────────────────────────────────────────────────
    ap.add_argument("--feat-dir", type=Path, default=FEAT_DIR_DEFAULT,
                    help="Directory containing sbj_*.npy feature files.")
    ap.add_argument("--ann-dir",  type=Path, default=ANN_DIR_DEFAULT,
                    help="Directory with wear_split_*.json annotation files.")
    ap.add_argument("--csv-dir",  type=Path, default=None,
                    help="Directory with raw sbj_*.csv files (label fallback).")
    ap.add_argument("--split",    type=int,  default=None,
                    help="Annotation split index (e.g. 1 → wear_split_1.json). "
                         "Determines train/val from the 'subset' field. "
                         "Omit to use --val-subjects holdout instead.")
    ap.add_argument("--ann-file", type=Path, default=None,
                    help="Direct path to a split JSON file (e.g. wear_test_split_1.json). "
                         "Overrides --ann-dir + --split path construction.")
    ap.add_argument("--val-subjects", type=int, default=3,
                    help="Subjects held out for validation when --split is not set "
                         "(last N by ID, default: 3).")
    ap.add_argument("--min-overlap", type=float, default=0.6,
                    help="Min annotation overlap ratio to assign a label (default: 0.6).")
    # ── Window override (for non-standard configs like right_w) ───────────────
    ap.add_argument("--imu-hz",   type=int,  default=None,
                    help="Override IMU Hz (auto-detected from path by default).")
    ap.add_argument("--video-fps", type=int, default=None,
                    help="Override video fps (auto-detected from path by default).")
    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument("--out-dir", type=Path,
                    default=None, help="Default: $NWEAR_OUTPUT/imu_snn/loso54/seed_<s>/train_split_<n>")
    # ── Training ──────────────────────────────────────────────────────────────
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=5e-3)
    ap.add_argument("--lr-milestones", type=int,   nargs="+", default=None)
    ap.add_argument("--lr-gamma",      type=float, default=0.2)
    ap.add_argument("--hidden",        type=int,   default=128)
    ap.add_argument("--dropout",       type=float, default=0.0)
    ap.add_argument("--zoneout",       type=float, default=0.0)
    ap.add_argument("--thr-mean",      type=float, default=0.3)
    ap.add_argument("--dampening",     type=float, default=0.7)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--include-null-class", action="store_true",
                    help="Keep unlabeled windows as an explicit NULL class.")
    ap.add_argument("--postprocess-majority-sec", type=float, default=0.0,
                    help="Temporal majority-vote smoothing window in seconds (e.g., 10).")
    ap.add_argument("--compute-map", action="store_true",
                    help="Compute temporal mAP at tIoU thresholds.")
    ap.add_argument("--map-tiou", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7],
                    help="tIoU thresholds for mAP.")
    ap.add_argument("--exclude-null-in-map", action="store_true",
                    help="Exclude NULL class from mAP averaging.")
    ap.add_argument("--seed",          type=int,   default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = args.feat_dir.expanduser().resolve()
    if args.out_dir is None:
        tag = f"train_split_{args.split}" if args.split is not None else "run"
        args.out_dir = P.IMU_RUNS / f"seed_{args.seed}" / tag
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not feat_dir.is_dir():
        raise SystemExit(f"feat-dir not found: {feat_dir}")

    # ── Auto-detect window parameters ────────────────────────────────────────
    wp = detect_window_params(feat_dir, args.imu_hz, args.video_fps)
    T_imu = wp["T_imu"]
    print(f"\nWindow config: {feat_dir.parent.name}/{feat_dir.name}")
    print(f"  imu_hz={wp['imu_hz']}  video_fps={wp['video_fps']}  "
          f"window={wp['window_sec']:.2f}s  stride={wp['stride_sec']:.2f}s  T_imu={T_imu}")

    milestones = args.lr_milestones or [args.epochs // 3, 2 * args.epochs // 3]

    # ── Resolve label source ─────────────────────────────────────────────────
    use_json = args.ann_dir is not None and args.ann_dir.is_dir()
    use_csv  = args.csv_dir is not None and args.csv_dir.is_dir()
    if not use_json and not use_csv:
        raise SystemExit(
            "No label source found. "
            "Provide --ann-dir (annotation JSON, recommended) or --csv-dir (raw CSV fallback)."
        )

    # Pick annotation file for labels and (optionally) train/val split
    ann_path = None
    if args.ann_file is not None:
        ann_path = args.ann_file.expanduser().resolve()
        if not ann_path.is_file():
            raise SystemExit(f"--ann-file not found: {ann_path}")
        print(f"Annotation file: {ann_path.name}")
    elif use_json:
        split_n  = args.split if args.split is not None else 1
        candidate = args.ann_dir / f"wear_split_{split_n}.json"
        if candidate.is_file():
            ann_path = candidate
        else:
            available = sorted(args.ann_dir.glob("wear_split_*.json"))
            if not available:
                use_json = False
                print(f"No wear_split_*.json in {args.ann_dir}, falling back to CSV labels.")
            else:
                ann_path = available[0]
                print(f"wear_split_{split_n}.json not found; using {ann_path.name}")
        if ann_path:
            print(f"Annotation file: {ann_path.name}")

    # ── Collect subjects from npy files ──────────────────────────────────────
    npy_files = sorted(
        feat_dir.glob("sbj_*.npy"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not npy_files:
        raise SystemExit(f"No sbj_*.npy files in {feat_dir}")

    subject_ids = [int(p.stem.split("_")[1]) for p in npy_files]
    print(f"\nFound {len(subject_ids)} subjects: {subject_ids}")

    # ── Determine train / val subject split ───────────────────────────────────
    if ann_path is not None:
        train_sids, val_sids = get_split_subjects(ann_path)
        sid_set    = set(subject_ids)
        train_sids = [s for s in train_sids if s in sid_set]
        val_sids   = [s for s in val_sids   if s in sid_set]
    else:
        n_val      = min(args.val_subjects, len(subject_ids) - 1)
        train_sids = subject_ids[:-n_val]
        val_sids   = subject_ids[-n_val:]

    print(f"  Train subjects ({len(train_sids)}): {train_sids}")
    print(f"  Val   subjects ({len(val_sids)}):   {val_sids}")

    class_names = list(ACTIVITY_CLASSES)
    null_label_idx: int | None = None
    if args.include_null_class:
        null_label_idx = len(class_names)
        class_names = class_names + [NULL_CLASS_NAME]
    num_classes = len(class_names)

    # ── Load all subjects ─────────────────────────────────────────────────────
    all_X:   list[np.ndarray] = []
    all_y:   list[np.ndarray] = []
    all_sid: list[np.ndarray] = []
    in_features: int | None   = None

    for sid in subject_ids:
        npy_path = feat_dir / f"sbj_{sid}.npy"
        X, F = load_npy(npy_path, T_imu)

        if in_features is None:
            in_features = F
        elif F != in_features:
            print(f"  Warning: sbj_{sid} has F={F}, expected F={in_features} — skipping")
            continue

        N = len(X)

        # Labels
        if use_json and ann_path:
            y = labels_from_json(
                ann_path, sid, N,
                wp["window_sec"], wp["stride_sec"],
                min_overlap=args.min_overlap,
            )
            label_src_str = f"annotation JSON ({ann_path.name})"
        else:
            csv_path = args.csv_dir / f"sbj_{sid}.csv"
            if not csv_path.is_file():
                print(f"  Skipping sbj_{sid}: CSV not found at {csv_path}")
                continue
            y = labels_from_csv(
                csv_path, N,
                wp["window_sec"], wp["stride_sec"],
            )
            label_src_str = f"raw CSV ({args.csv_dir.name})"

        if args.include_null_class:
            y_proc = np.where(y >= 0, y, null_label_idx)
            mask = np.ones_like(y_proc, dtype=bool)
        else:
            y_proc = y
            mask = y_proc >= 0

        n_used = int(mask.sum())
        print(f"  sbj_{sid}: {n_used}/{N} windows used (F={F})")

        all_X.append(X[mask])
        all_y.append(y_proc[mask])
        all_sid.append(np.full(n_used, sid, dtype=np.int64))

    if not all_X:
        raise SystemExit("No windows loaded. Check paths and label source.")

    all_X   = np.concatenate(all_X,   axis=0)
    all_y   = np.concatenate(all_y,   axis=0)
    all_sid = np.concatenate(all_sid, axis=0)

    # ── Split train / val arrays ──────────────────────────────────────────────
    tr_mask = np.isin(all_sid, train_sids)
    va_mask = np.isin(all_sid, val_sids)
    x_tr_raw, y_tr_raw = all_X[tr_mask], all_y[tr_mask]
    x_va_raw, y_va_raw = all_X[va_mask], all_y[va_mask]
    sid_val_raw = all_sid[va_mask]

    if len(x_tr_raw) == 0:
        raise SystemExit("No training windows. Check subject IDs and label assignment.")
    if len(x_va_raw) == 0:
        raise SystemExit("No validation windows. Check subject IDs and label assignment.")

    # ── Normalize (fit on train, apply to val) ────────────────────────────────
    x_tr_n, mean, std = normalize_per_channel(x_tr_raw)
    x_va_n, _,    _   = normalize_per_channel(x_va_raw, mean, std)

    x_train = torch.tensor(x_tr_n,   dtype=torch.float32)
    y_train = torch.tensor(y_tr_raw, dtype=torch.long)
    x_val   = torch.tensor(x_va_n,   dtype=torch.float32)
    y_val   = torch.tensor(y_va_raw, dtype=torch.long)

    print(
        f"\nLabel source : {label_src_str}"
        f"\nTrain windows: {len(x_train)} | Val windows: {len(x_val)}"
        f"\nIn features  : {in_features} | T_imu: {T_imu} | Device: {device}"
        f"\nClasses      : {num_classes} ({'with NULL' if args.include_null_class else 'no NULL'})"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining EGRU (hidden={args.hidden}, epochs={args.epochs}) ...")
    model, tr_loss, va_loss, va_acc, va_f1 = fit(
        x_train, y_train, x_val, y_val,
        in_features=in_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        hidden=args.hidden,
        dropout=args.dropout,
        zoneout=args.zoneout,
        thr_mean=args.thr_mean,
        dampening_factor=args.dampening,
        use_class_weights=not args.no_class_weights,
        num_classes=num_classes,
        milestones=milestones,
        lr_gamma=args.lr_gamma,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = args.out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved -> {model_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training(tr_loss, va_loss, va_acc,
                  args.out_dir / "01_training.png", milestones)

    _, final_acc_raw, final_f1_raw, pred_raw, probs_raw = evaluate(
        model, x_val, y_val, device, None, num_classes
    )
    pred_eval = pred_raw
    if args.postprocess_majority_sec > 0:
        pred_eval = majority_vote_filter(
            pred_raw, sid_val_raw, wp["stride_sec"], args.postprocess_majority_sec
        )
    final_acc = float((pred_eval == y_val.numpy()).mean())
    final_f1 = macro_f1_score(y_val.numpy(), pred_eval, num_classes=num_classes)

    map_by_tiou: dict[float, float] = {}
    map_avg = 0.0
    if args.compute_map:
        exclude_cls = null_label_idx if (args.exclude_null_in_map and null_label_idx is not None) else None
        map_by_tiou, map_avg = temporal_map(
            y_true=y_val.numpy(),
            y_pred=pred_eval,
            sid=sid_val_raw,
            probs=probs_raw,
            num_classes=num_classes,
            window_sec=wp["window_sec"],
            stride_sec=wp["stride_sec"],
            tiou_thresholds=args.map_tiou,
            exclude_class=exclude_cls,
        )

    plot_confusion(y_val.numpy(), pred_eval, class_names,
                   args.out_dir / "02_confusion_val.png")

    plot_architecture(model, in_features, T_imu,
                      args.out_dir / "05_model_architecture.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = "\n".join([
        "Pipeline: EGRU on WEAR inertial features (npy + annotation JSON labels)",
        f"Feat dir : {feat_dir}",
        f"Ann dir  : {args.ann_dir}",
        f"Label src: {label_src_str}",
        f"Window   : {feat_dir.parent.name}/{feat_dir.name}",
        f"  imu_hz={wp['imu_hz']}  video_fps={wp['video_fps']}  "
        f"window={wp['window_sec']:.2f}s  stride={wp['stride_sec']:.2f}s  T_imu={T_imu}",
        f"Eval classes   : {class_names}",
        f"Train subjects : {sorted(train_sids)}",
        f"Val   subjects : {sorted(val_sids)}",
        f"Train windows  : {len(x_train)} | Val windows: {len(x_val)}",
        f"In features    : {in_features}",
        f"Hidden: {args.hidden} | thr_mean: {args.thr_mean} | dampening: {args.dampening}",
        f"Dropout: {args.dropout} | Zoneout: {args.zoneout}",
        f"Optimizer: Adam lr={args.lr}",
        f"Scheduler: MultiStepLR milestones={milestones} gamma={args.lr_gamma}",
        f"Class weights: {not args.no_class_weights}",
        f"Include NULL class: {args.include_null_class}",
        f"Postprocess majority sec: {args.postprocess_majority_sec}",
        f"Compute mAP: {args.compute_map}",
        f"mAP tIoU thresholds: {args.map_tiou}",
        f"Exclude NULL in mAP: {args.exclude_null_in_map}",
        f"Epochs: {args.epochs}",
        f"Final val accuracy (raw): {final_acc_raw:.3f}",
        f"Final val macro-F1 (raw): {final_f1_raw:.3f}",
        f"Final val accuracy: {final_acc:.3f}",
        f"Final val macro-F1: {final_f1:.3f}",
        f"Final val mAP avg: {map_avg:.3f}",
        f"Final val mAP by tIoU: {json.dumps(map_by_tiou, sort_keys=True)}",
        f"Device: {device}",
        f"Output: {args.out_dir.resolve()}",
    ])
    (args.out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
