#!/usr/bin/env python3
"""Plot validation confusion matrix from a LOSO checkpoint (model.pt)."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import nwear_paths as P

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from event_snn.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from event_snn.dataset.labels import (
    COARSE_ACTIVITY_CLASSES,
    FINE_ACTIVITY_CLASSES,
    FINE_IDX_TO_COARSE_IDX,
)
from event_snn.training.losses import logits_to_pred
from event_snn.training.trainer import V2EAicoTrainer


def _config_from_checkpoint(raw: dict, out_dir: Path) -> ExperimentConfig:
    data_raw = dict(raw["data"])
    data_raw["dataset_root"] = Path(data_raw["dataset_root"])
    if "events_tier" not in data_raw:
        data_raw["events_tier"] = "dvs128" if "dvs128" in str(out_dir) else "dvs240"
    data = DataConfig(**data_raw)
    model = ModelConfig.from_dict(raw["model"])
    train_defaults = {k.name: getattr(TrainConfig(), k.name) for k in fields(TrainConfig)}
    train_defaults.update(raw["train"])
    train = TrainConfig(**train_defaults)
    if train.cache_dir is not None:
        train.cache_dir = Path(train.cache_dir)
    return ExperimentConfig(data=data, model=model, train=train, out_dir=out_dir)


from sklearn.metrics import f1_score


def fine_labels_to_coarse(labels: list[int] | np.ndarray) -> np.ndarray:
    mapping = np.array(FINE_IDX_TO_COARSE_IDX, dtype=np.int64)
    arr = np.asarray(labels, dtype=np.int64)
    return mapping[arr]


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(8, k * 1.4), max(7, k * 1.2)))
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
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )
    fig.colorbar(im, ax=ax, label="Recall per class")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LOSO val confusion matrix from model.pt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "output/loso_dvs128_train0-10_test11_allconv_egru_valfix/model.pt"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Merge 18 fine classes into 7 coarse groups for confusion matrix",
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="test",
        help="Which split to plot (default: test = held-out subject for LOSO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir or checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = _config_from_checkpoint(checkpoint["config"], out_dir)
    cfg.out_dir = out_dir
    cfg.train.device = str(device)

    cache_default = out_dir / "tensor_cache"
    shared_d128 = (
        out_dir.parent / "loso_dvs128_train0-10_test11_allconv_egru_valfix" / "tensor_cache"
    )
    shared_d240 = out_dir.parent / "loso_train1-17_test0" / "tensor_cache"
    if cfg.train.cache_dir is None:
        if cfg.data.events_tier == "dvs128" and shared_d128.is_dir():
            cfg.train.cache_dir = shared_d128
        elif shared_d240.is_dir():
            cfg.train.cache_dir = shared_d240
        elif cache_default.is_dir():
            cfg.train.cache_dir = cache_default

    metrics_path = out_dir / "metrics.json"
    meta: dict = {}
    if metrics_path.is_file():
        import json

        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_subjects = meta["train_subjects"]
        test_subject = meta["test_subject"]
    else:
        train_subjects = list(range(1, 18))
        test_subject = 0

    val_on_held_out = (
        meta.get("split_sizes", {}).get("val") == meta.get("split_sizes", {}).get("test")
        and meta.get("val", {}).get("accuracy") == meta.get("test", {}).get("accuracy")
    )

    trainer = V2EAicoTrainer(cfg)
    split, train_sids, test_sid = trainer.build_loso_splits(
        args.dataset_base,
        train_subjects=train_subjects,
        test_subject=test_subject,
        val_on_held_out=val_on_held_out,
    )
    _, val_loader, test_loader = trainer.build_loaders(split)
    eval_loader = test_loader if args.eval_split == "test" else val_loader
    eval_sid = test_sid if args.eval_split == "test" else "train holdout"
    eval_label = f"sbj_{test_sid}" if args.eval_split == "test" else "train-subject holdout"
    model = trainer.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            pred = logits_to_pred(logits)
            if args.coarse:
                pred = torch.tensor(fine_labels_to_coarse(pred.cpu().tolist()), dtype=torch.long)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch_y.tolist())

    y_true = np.array(labels, dtype=np.int64)
    y_pred = np.array(preds, dtype=np.int64)
    if args.coarse:
        y_true = fine_labels_to_coarse(y_true)
        class_names = COARSE_ACTIVITY_CLASSES
        suffix = "_coarse"
        scheme = "7 coarse classes"
    else:
        class_names = FINE_ACTIVITY_CLASSES
        suffix = ""
        scheme = "18 fine classes"

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(len(class_names))))
    )
    best_epoch = int(checkpoint.get("best_epoch", -1)) + 1

    model_name = cfg.model.model_name
    tier = cfg.data.events_tier.upper()
    out_path = out_dir / f"02_confusion_{args.eval_split}{suffix}.png"
    plot_confusion(
        y_true,
        y_pred,
        class_names,
        out_path,
        title=(
            f"Confusion — {model_name} {tier} ({scheme}) | "
            f"train sbj {train_sids} | {args.eval_split} {eval_label} | "
            f"acc={acc * 100:.1f}% macro-F1={f1 * 100:.1f}% | best epoch {best_epoch}"
        ),
    )
    print(f"Saved {out_path}")
    print(f"{args.eval_split} accuracy ({scheme}): {acc * 100:.2f}%  macro-F1: {f1 * 100:.2f}%")
    print(f"Windows: {len(y_true)} on {eval_label}")


if __name__ == "__main__":
    main()
