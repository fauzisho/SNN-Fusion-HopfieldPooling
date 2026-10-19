"""Shared DVS128 + IMU alignment and frozen-encoder feature export."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import nwear_paths as P

P.add_to_path(P.REPO_ROOT, P.IMU_SNN_DIR)

from event_snn.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from event_snn.dataset.dvs240_window_dataset import DVS240WindowDataset  # noqa: E402
from event_snn.dataset.splits import discover_h5_files  # noqa: E402
from event_snn.dataset.window_index import WindowSample  # noqa: E402
from event_snn.models.allconv_adlif_egru_dvs128 import AllConvAdLIFEGRU_DVS128  # noqa: E402

from pipeline_inertial import (  # noqa: E402
    ACTIVITY_CLASSES,
    EGRUClassifier,
    detect_window_params,
    get_split_subjects,
    labels_from_json,
    load_npy,
    normalize_per_channel,
)

NUM_CLASSES = len(ACTIVITY_CLASSES)
FEAT_DIM = NUM_CLASSES + 2  # class probs + max_prob + entropy


def frame_features(class_scores: np.ndarray) -> np.ndarray:
    scores = class_scores.astype(np.float32)
    max_prob = scores.max(axis=1, keepdims=True)
    total = np.maximum(scores.sum(axis=1, keepdims=True), 1e-8)
    norm = scores / total
    entropy = -(norm * np.log(np.maximum(norm, 1e-8))).sum(axis=1, keepdims=True)
    return np.concatenate([scores, max_prob, entropy], axis=1).astype(np.float32)


def parse_segment_start_sec(segment_key: str) -> float | None:
    name = Path(segment_key).name
    match = re.search(r"_from_([\d.]+)s?$", name)
    return float(match.group(1)) if match else None


def sample_center_sec(sample: WindowSample, sample_length_ms: int) -> float | None:
    start_sec = parse_segment_start_sec(sample.segment_key)
    if start_sec is None:
        return None
    return start_sec + (sample.center_us / 1_000_000.0)


@dataclass
class ModalityPreds:
    labels: np.ndarray
    preds: np.ndarray
    probs: np.ndarray
    times: np.ndarray


@dataclass
class SubjectAligned:
    subject_id: int
    is_val: bool
    dvs_feat: np.ndarray
    imu_feat: np.ndarray
    labels: np.ndarray
    times: np.ndarray


@dataclass
class AlignedSplit:
    val_subject: int
    split: int
    subjects: list[SubjectAligned]

    def concat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dvs_chunks, imu_chunks, y_chunks, sid_chunks, val_chunks = [], [], [], [], []
        for subj in self.subjects:
            n = len(subj.labels)
            dvs_chunks.append(subj.dvs_feat)
            imu_chunks.append(subj.imu_feat)
            y_chunks.append(subj.labels)
            sid_chunks.append(np.full(n, subj.subject_id, dtype=np.int64))
            val_chunks.append(np.full(n, subj.is_val, dtype=bool))
        return (
            np.concatenate(dvs_chunks),
            np.concatenate(imu_chunks),
            np.concatenate(y_chunks),
            np.concatenate(sid_chunks),
            np.concatenate(val_chunks),
        )


def load_dvs_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    raw = ckpt["config"]
    data = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw["model"])
    train_cfg = TrainConfig(**raw["train"])
    model = AllConvAdLIFEGRU_DVS128(
        channels=model_cfg.channels,
        feat_neurons=model_cfg.feat_neurons,
        classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
        input_height=data.input_height,
        input_width=data.input_width,
        device=device,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, data, train_cfg


def load_imu_checkpoint(
    ckpt_path: Path,
    in_features: int,
    *,
    hidden: int,
    include_null: bool,
    device: torch.device,
) -> EGRUClassifier:
    num_classes = NUM_CLASSES + (1 if include_null else 0)
    model = EGRUClassifier(
        input_size=in_features,
        hidden_size=hidden,
        num_classes=num_classes,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_imu_train_norm(
    feat_dir: Path,
    ann_path: Path,
    train_sids: list[int],
    *,
    T_imu: int,
    window_sec: float,
    stride_sec: float,
    min_overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for sid in train_sids:
        x, _ = load_npy(feat_dir / f"sbj_{sid}.npy", T_imu)
        y = labels_from_json(ann_path, sid, len(x), window_sec, stride_sec, min_overlap)
        mask = y >= 0
        if mask.any():
            chunks.append(x[mask])
    if not chunks:
        raise RuntimeError("No IMU training windows for normalization.")
    x_tr = np.concatenate(chunks, axis=0)
    _, mean, std = normalize_per_channel(x_tr)
    return mean, std


@torch.no_grad()
def infer_dvs_subject(
    model: torch.nn.Module,
    data_cfg: DataConfig,
    dvs_root: Path,
    *,
    cache_dir: Path | None,
    batch_size: int,
    device: torch.device,
) -> ModalityPreds:
    files = discover_h5_files(dvs_root)
    dataset = DVS240WindowDataset(
        files,
        data_cfg,
        augment=False,
        normalize=data_cfg.normalize_clip,
        cache_dir=cache_dir,
        return_meta=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    labels: list[int] = []
    preds: list[int] = []
    probs: list[np.ndarray] = []
    times: list[float] = []
    sample_idx = 0
    for batch_x, batch_y, _keys, _center_us in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        logits_pooled = logits.mean(dim=-1) if logits.dim() == 3 else logits
        batch_probs = torch.softmax(logits_pooled, dim=1).cpu().numpy()
        batch_pred = logits_pooled.argmax(dim=1).cpu().numpy()
        for i in range(batch_x.shape[0]):
            sample = dataset.samples[sample_idx]
            center = sample_center_sec(sample, data_cfg.sample_length_ms)
            sample_idx += 1
            if center is None:
                continue
            labels.append(int(batch_y[i]))
            preds.append(int(batch_pred[i]))
            probs.append(batch_probs[i])
            times.append(center)

    return ModalityPreds(
        labels=np.asarray(labels, dtype=np.int64),
        preds=np.asarray(preds, dtype=np.int64),
        probs=np.stack(probs, axis=0).astype(np.float32),
        times=np.asarray(times, dtype=np.float64),
    )


@torch.no_grad()
def infer_imu_subject(
    model: EGRUClassifier,
    feat_dir: Path,
    ann_path: Path,
    subject_id: int,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    min_overlap: float,
    device: torch.device,
) -> ModalityPreds:
    wp = detect_window_params(feat_dir)
    T_imu = wp["T_imu"]
    window_sec = wp["window_sec"]
    stride_sec = wp["stride_sec"]

    x, _ = load_npy(feat_dir / f"sbj_{subject_id}.npy", T_imu)
    y = labels_from_json(ann_path, subject_id, len(x), window_sec, stride_sec, min_overlap)
    centers = np.array(
        [i * stride_sec + window_sec / 2.0 for i in range(len(x))],
        dtype=np.float64,
    )
    x_n, _, _ = normalize_per_channel(x, mean, std)

    x_t = torch.tensor(x_n, dtype=torch.float32, device=device)
    logits = model(x_t)
    logits_t = logits.mean(dim=-1)
    probs = torch.softmax(logits_t, dim=1).cpu().numpy()
    preds = logits_t.argmax(dim=1).cpu().numpy()

    return ModalityPreds(
        labels=y.astype(np.int64),
        preds=preds.astype(np.int64),
        probs=probs.astype(np.float32),
        times=centers,
    )


def align_subject(
    dvs: ModalityPreds,
    imu: ModalityPreds,
    *,
    max_time_diff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dvs_idx, imu_idx, labels, times = [], [], [], []
    for i, t_d in enumerate(dvs.times):
        j = int(np.argmin(np.abs(imu.times - t_d)))
        if abs(imu.times[j] - t_d) > max_time_diff:
            continue
        d_lbl = int(dvs.labels[i])
        i_lbl = int(imu.labels[j])
        if i_lbl < 0 or i_lbl != d_lbl:
            continue
        dvs_idx.append(i)
        imu_idx.append(j)
        labels.append(d_lbl)
        times.append(t_d)
    return (
        np.asarray(dvs_idx, dtype=np.int64),
        np.asarray(imu_idx, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
    )


def collect_aligned_split(
    *,
    seed: int,
    val_subject: int,
    dvs_ckpt: Path,
    imu_ckpt: Path,
    feat_dir: Path,
    ann_path: Path,
    dataset_base: Path,
    dvs_cache_dir: Path | None,
    max_time_diff: float,
    imu_hidden: int,
    min_overlap: float,
    device: torch.device,
    batch_size: int = 16,
    subject_ids: list[int] | None = None,
) -> AlignedSplit:
    split = val_subject + 1
    train_sids, val_sids = get_split_subjects(ann_path)
    if val_subject not in val_sids:
        raise ValueError(f"sbj_{val_subject} not in val subjects for {ann_path.name}: {val_sids}")

    wp = detect_window_params(feat_dir)
    mean, std = load_imu_train_norm(
        feat_dir,
        ann_path,
        train_sids,
        T_imu=wp["T_imu"],
        window_sec=wp["window_sec"],
        stride_sec=wp["stride_sec"],
        min_overlap=min_overlap,
    )

    dvs_model, data_cfg, train_cfg = load_dvs_checkpoint(dvs_ckpt, device)
    imu_model = load_imu_checkpoint(
        imu_ckpt,
        in_features=int(load_npy(feat_dir / f"sbj_{train_sids[0]}.npy", wp["T_imu"])[1]),
        hidden=imu_hidden,
        include_null=True,
        device=device,
    )
    bs = batch_size or train_cfg.batch_size

    subjects: list[SubjectAligned] = []
    all_sids = sorted(set(train_sids) | set(val_sids))
    if subject_ids is not None:
        want = set(int(s) for s in subject_ids)
        all_sids = [s for s in all_sids if s in want]
    for sid in all_sids:
        dvs_root = dataset_base / f"sbj_{sid}_events_dvs128"
        dvs = infer_dvs_subject(
            dvs_model,
            data_cfg,
            dvs_root,
            cache_dir=dvs_cache_dir,
            batch_size=bs,
            device=device,
        )
        imu = infer_imu_subject(
            imu_model,
            feat_dir,
            ann_path,
            sid,
            mean,
            std,
            min_overlap=min_overlap,
            device=device,
        )
        d_idx, m_idx, labels, times = align_subject(dvs, imu, max_time_diff=max_time_diff)
        if len(labels) == 0:
            print(f"  sbj_{sid}: no aligned pairs", flush=True)
            continue
        order = np.argsort(times)
        dvs_feat = frame_features(dvs.probs[d_idx][:, :NUM_CLASSES])
        imu_feat = frame_features(imu.probs[m_idx][:, :NUM_CLASSES])
        subjects.append(
            SubjectAligned(
                subject_id=sid,
                is_val=sid == val_subject,
                dvs_feat=dvs_feat[order],
                imu_feat=imu_feat[order],
                labels=labels[order],
                times=times[order],
            )
        )
        print(f"  sbj_{sid}: {len(labels)} aligned pairs", flush=True)

    return AlignedSplit(val_subject=val_subject, split=split, subjects=subjects)
