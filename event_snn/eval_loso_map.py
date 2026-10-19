#!/usr/bin/env python3
"""Compute WEAR-style segment mAP@tIoU {0.3..0.7} for LOSO DVS128 checkpoints.

Re-infers the held-out test subject, stitches activity-folder windows onto one
timeline, converts window preds → segments, and scores with ANETdetection
against GT segments built from the same window labels.

Note: this is pipeline-consistent segment mAP on the DVS window timeline
(not the official IMU/camera wear_split JSON clock).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
from torch.utils.data import DataLoader

from event_snn.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from event_snn.dataset.labels import FINE_ACTIVITY_CLASSES
from event_snn.dataset.loso import loso_train_test_split
from event_snn.training.losses import logits_to_pred
from event_snn.training.postprocess import macro_metrics
from event_snn.training.trainer import V2EAicoTrainer, _collate_with_meta

WEAR_REPO = P.WEAR_REPO
TIOU = [0.3, 0.4, 0.5, 0.6, 0.7]
TIOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
DEFAULT_OUT = P.EVENT_RUNS


def _load_anet():
    path = WEAR_REPO / "postprocessing" / "utils.py"
    spec = importlib.util.spec_from_file_location("wear_pp_utils", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.ANETdetection, mod.convert_samples_to_segments


ANETdetection, _convert_samples_to_segments = _load_anet()


def _config_from_checkpoint(raw: dict, out_dir: Path) -> ExperimentConfig:
    data_raw = dict(raw["data"])
    data_raw["dataset_root"] = Path(data_raw["dataset_root"])
    data = DataConfig(**{k.name: data_raw[k.name] for k in fields(DataConfig) if k.name in data_raw})
    model = ModelConfig(**{k.name: raw["model"][k.name] for k in fields(ModelConfig) if k.name in raw["model"]})
    train_raw = dict(raw["train"])
    if train_raw.get("cache_dir") is not None:
        train_raw["cache_dir"] = Path(train_raw["cache_dir"])
    train = TrainConfig(**{k.name: train_raw[k.name] for k in fields(TrainConfig) if k.name in train_raw})
    return ExperimentConfig(data=data, model=model, train=train, out_dir=out_dir)


def stitch_timeline(
    segment_keys: list[str],
    start_us: list[int],
    window_ms: int,
) -> np.ndarray:
    """Map per-folder start_us onto one contiguous subject timeline (seconds centers → use starts)."""
    window_us = window_ms * 1000
    order = sorted(range(len(segment_keys)), key=lambda i: (segment_keys[i], start_us[i]))
    global_start = np.zeros(len(segment_keys), dtype=np.int64)
    offset = 0
    prev_key = None
    folder_max_end = 0
    for i in order:
        key = segment_keys[i]
        if prev_key is not None and key != prev_key:
            offset += folder_max_end
            folder_max_end = 0
        global_start[i] = offset + int(start_us[i])
        folder_max_end = max(folder_max_end, int(start_us[i]) + window_us)
        prev_key = key
    return global_start


def build_gt_json(
    video_id: str,
    global_start_us: np.ndarray,
    labels: np.ndarray,
    window_ms: int,
    path: Path,
) -> None:
    """Merge contiguous same-label windows into GT annotations."""
    window_us = window_ms * 1000
    order = np.argsort(global_start_us)
    anns = []
    cur_lab = None
    cur_t0 = None
    cur_t1 = None
    for i in order:
        lab = int(labels[i])
        t0 = float(global_start_us[i]) / 1e6
        t1 = float(global_start_us[i] + window_us) / 1e6
        if cur_lab is None:
            cur_lab, cur_t0, cur_t1 = lab, t0, t1
        elif lab == cur_lab and abs(t0 - cur_t1) < 1e-6:
            cur_t1 = t1
        else:
            anns.append(
                {
                    "label": FINE_ACTIVITY_CLASSES[cur_lab],
                    "segment": [cur_t0, cur_t1],
                    "label_id": cur_lab,
                }
            )
            cur_lab, cur_t0, cur_t1 = lab, t0, t1
    if cur_lab is not None:
        anns.append(
            {
                "label": FINE_ACTIVITY_CLASSES[cur_lab],
                "segment": [cur_t0, cur_t1],
                "label_id": cur_lab,
            }
        )
    payload = {
        "label_dict": {name: i for i, name in enumerate(FINE_ACTIVITY_CLASSES)},
        "database": {
            video_id: {
                "subset": "Validation",
                "duration": float(anns[-1]["segment"][1]) if anns else 0.0,
                "annotations": anns,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def eval_fold(
    run_dir: Path,
    dataset_base: Path,
    device: torch.device,
    batch_size: int,
    mv_kernel: int,
) -> dict:
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(ckpt["config"], run_dir)
    cfg.train.device = str(device)
    cfg.train.batch_size = batch_size
    cfg.train.majority_vote_k = 0  # we handle meta ourselves

    test_subject = int(run_dir.name.replace("test_sbj_", ""))
    seed = int(run_dir.parent.name.replace("seed_", ""))
    # recreate LOSO split (val=test subject when val_on_held_out)
    train_subjects = [s for s in range(18) if s != test_subject]
    split, _, _ = loso_train_test_split(
        dataset_base,
        train_subjects=train_subjects,
        test_subject=test_subject,
        events_tier=cfg.data.events_tier,
        val_ratio=0.10,
        val_on_held_out=True,
        seed=cfg.data.seed,
    )

    trainer = V2EAicoTrainer(cfg)
    model = trainer.build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    test_ds = trainer._make_dataset(split.test, augment=False, return_meta=True)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, cfg.train.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_with_meta,
    )

    preds: list[int] = []
    labels: list[int] = []
    keys: list[str] = []
    starts: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_keys, batch_times in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            pred = logits_to_pred(logits)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch_y.tolist())
            keys.extend(batch_keys)
            # dataset returns center_us; convert to start_us
            half = cfg.data.sample_length_ms * 1000 // 2
            starts.extend(int(t) - half for t in batch_times)

    preds_a = np.asarray(preds, dtype=np.int64)
    labels_a = np.asarray(labels, dtype=np.int64)

    if mv_kernel > 1:
        # light temporal smoothing over sorted timeline within each folder
        order = sorted(range(len(keys)), key=lambda i: (keys[i], starts[i]))
        smoothed = preds_a.copy()
        half_k = mv_kernel // 2
        from collections import defaultdict

        by_key: dict[str, list[int]] = defaultdict(list)
        for i in order:
            by_key[keys[i]].append(i)
        for idxs in by_key.values():
            arr = preds_a[idxs]
            n = len(arr)
            out = np.empty(n, dtype=np.int64)
            for j in range(n):
                left = max(0, j - half_k)
                right = min(n, j + half_k + 1)
                if right - left < mv_kernel:
                    if left == 0:
                        right = min(n, mv_kernel)
                    else:
                        left = max(0, n - mv_kernel)
                out[j] = np.bincount(arr[left:right]).argmax()
            smoothed[idxs] = out
        preds_a = smoothed

    cls = macro_metrics(labels_a.tolist(), preds_a.tolist())

    global_start = stitch_timeline(keys, starts, cfg.data.sample_length_ms)
    video_id = f"sbj_{test_subject}"

    def merge_windows(lab_arr: np.ndarray) -> dict:
        window_s = cfg.data.sample_length_ms / 1000.0
        order = np.argsort(global_start)
        f_ids, f_lab, f_t0, f_t1, f_sc = [], [], [], [], []
        cur = None
        for i in order:
            lab = int(lab_arr[i])
            t0 = float(global_start[i]) / 1e6
            t1 = t0 + window_s
            if cur is None:
                cur = [lab, t0, t1]
            elif lab == cur[0] and abs(t0 - cur[2]) < 1e-6:
                cur[2] = t1
            else:
                f_ids.append(video_id)
                f_lab.append(cur[0])
                f_t0.append(cur[1])
                f_t1.append(cur[2])
                f_sc.append(1.0)
                cur = [lab, t0, t1]
        if cur is not None:
            f_ids.append(video_id)
            f_lab.append(cur[0])
            f_t0.append(cur[1])
            f_t1.append(cur[2])
            f_sc.append(1.0)
        return {
            "video-id": f_ids,
            "label": np.asarray(f_lab, dtype=np.float64),
            "t-start": np.asarray(f_t0, dtype=np.float64),
            "t-end": np.asarray(f_t1, dtype=np.float64),
            "score": np.asarray(f_sc, dtype=np.float64),
        }

    pred_seg = merge_windows(preds_a)

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / f"gt_sbj_{test_subject}.json"
        build_gt_json(video_id, global_start, labels_a, cfg.data.sample_length_ms, gt_path)
        det = ANETdetection(str(gt_path), "validation", tiou_thresholds=TIOU, num_workers=2)
        v_map, _ = det.evaluate(pred_seg)

    v_map = np.asarray(v_map, dtype=np.float64)
    out = {
        "Seed": seed,
        "TestSubject": test_subject,
        "P": 100.0 * cls["precision"],
        "R": 100.0 * cls["recall"],
        "F1": 100.0 * cls["f1"],
        "Acc": 100.0 * cls["accuracy"],
        "mAP_avg": float(v_map.mean() * 100.0),
        **{f"mAP_{k}": float(v * 100.0) for k, v in zip(TIOU_KEYS, v_map)},
        "n_windows": len(preds_a),
        "n_pred_segs": len(pred_seg["video-id"]),
    }
    (run_dir / "map_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def write_paper_table(rows: list[dict], out_root: Path, mv_kernel: int) -> None:
    p = [r["P"] for r in rows]
    r = [r["R"] for r in rows]
    f1 = [r["F1"] for r in rows]
    maps = {k: [r[f"mAP_{k}"] for r in rows] for k in TIOU_KEYS}
    avg = [r["mAP_avg"] for r in rows]
    acc = [r["Acc"] for r in rows]

    lines = [
        "# experimentv2e — paper-style results (WEAR format)",
        "",
        "Format matches WEAR (`dataset/3699776.pdf`): **Model · CL · P · R · F1 · mAP@0.3–0.7 · Avg**.",
        "",
        f"LOSO mean over **{len(rows)}** folds (3 seeds × 18 subjects). Clip / window **2s**.",
        f"Segment mAP from re-inferred windows → merged segments (`eval_loso_map.py`), "
        f"MV kernel **{mv_kernel}** (0=off). GT timeline = stitched DVS activity folders.",
        "",
        "| Model | CL | P | R | F1 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg |",
        "|-------|----|--:|--:|--:|--------:|--------:|--------:|--------:|--------:|----:|",
        (
            f"| AllConvAdLIFEGRU+aug (LOSO18) | 2s | "
            f"{statistics.mean(p):.2f} | {statistics.mean(r):.2f} | {statistics.mean(f1):.2f} | "
            f"{statistics.mean(maps['0.3']):.2f} | {statistics.mean(maps['0.4']):.2f} | "
            f"{statistics.mean(maps['0.5']):.2f} | {statistics.mean(maps['0.6']):.2f} | "
            f"{statistics.mean(maps['0.7']):.2f} | **{statistics.mean(avg):.2f}** |"
        ),
        "",
        f"Acc **{statistics.mean(acc):.2f}% ± {statistics.stdev(acc) if len(acc)>1 else 0:.2f}**.",
        "",
        "| | mean ± std |",
        "|--|--|",
        f"| P | {mean_std(p)} |",
        f"| R | {mean_std(r)} |",
        f"| F1 | {mean_std(f1)} |",
        f"| mAP@0.3 | {mean_std(maps['0.3'])} |",
        f"| mAP@0.4 | {mean_std(maps['0.4'])} |",
        f"| mAP@0.5 | {mean_std(maps['0.5'])} |",
        f"| mAP@0.6 | {mean_std(maps['0.6'])} |",
        f"| mAP@0.7 | {mean_std(maps['0.7'])} |",
        f"| **Avg mAP** | **{mean_std(avg)}** |",
        "",
        "CSV: `results_dvs128_paper_table_map.csv`",
        "",
    ]
    (out_root / "RESULTS_DVS128_PAPER_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out_root / "results_dvs128_paper_table_map.csv"
    fields = ["Seed", "TestSubject", "Acc", "P", "R", "F1"] + [f"mAP_{k}" for k in TIOU_KEYS] + [
        "mAP_avg",
        "n_windows",
        "n_pred_segs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["Seed"], x["TestSubject"])):
            w.writerow(row)
    print(f"Wrote {out_root / 'RESULTS_DVS128_PAPER_TABLE.md'}")
    print(f"Wrote {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mv-kernel", type=int, default=0, help="Majority vote over consecutive windows (0=off)")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="Only this seed (0=all)")
    ap.add_argument("--test-subject", type=int, default=-1, help="Only this subject (-1=all)")
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only write per-fold map_metrics.json (for Slurm array tasks)",
    )
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device} mv={args.mv_kernel}")

    folds = sorted(args.out_root.glob("seed_*/test_sbj_*"))
    if args.seed:
        folds = [f for f in folds if f.parent.name == f"seed_{args.seed}"]
    if args.test_subject >= 0:
        folds = [f for f in folds if f.name == f"test_sbj_{args.test_subject}"]
    if args.max_folds:
        folds = folds[: args.max_folds]

    rows = []
    for fold in folds:
        print(f"\n=== {fold.parent.name}/{fold.name} ===")
        row = eval_fold(fold, args.dataset_base, device, args.batch_size, args.mv_kernel)
        rows.append(row)
        print(
            f"  Acc={row['Acc']:.1f} F1={row['F1']:.1f} "
            f"mAP={row['mAP_avg']:.1f} (0.3={row['mAP_0.3']:.1f} … 0.7={row['mAP_0.7']:.1f}) "
            f"segs={row['n_pred_segs']}"
        )

    if rows and not args.no_aggregate:
        write_paper_table(rows, args.out_root, args.mv_kernel)


if __name__ == "__main__":
    main()
