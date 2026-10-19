#!/usr/bin/env python3
"""Train the event SNN (Vision SNN) for one leave-one-subject-out fold.

Defaults reproduce the 54 paper runs (subjects 0-17 x seeds 1-3); pass
--augment --val-on-held-out as in ``cluster/event_snn_loso54.sbatch``:

    python -m event_snn.train_loso --test-subject 0 --seed 1 --augment --val-on-held-out \
        --train-subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
        --out-dir "$NWEAR_OUTPUT/event_snn/loso54/seed_1/test_sbj_0" \
        --cache-dir "$NWEAR_OUTPUT/event_snn/tensor_cache"

Note: with --val-on-held-out the "val" set is the held-out subject, and model.pt is
the epoch with the best accuracy on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import nwear_paths as P

from event_snn.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from event_snn.training.trainer import V2EAicoTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO training of the event SNN (AllConvAdLIFEGRU_DVS128)"
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=P.DATA_ROOT,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=P.EVENT_RUNS / "seed_1" / "test_sbj_0",
    )
    parser.add_argument(
        "--train-subjects",
        type=int,
        nargs="+",
        default=list(range(1, 18)),
    )
    parser.add_argument("--test-subject", type=int, default=0)
    parser.add_argument(
        "--model",
        type=str,
        choices=("allconv_adlif_egru_dvs128",),
        default="allconv_adlif_egru_dvs128",
    )
    parser.add_argument(
        "--events-tier",
        type=str,
        choices=("dvs240", "dvs128"),
        default="dvs128",
        help="H5 event folder tier: sbj_N_events_dvs240 or sbj_N_events_dvs128",
    )
    parser.add_argument("--feat-neurons", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument(
        "--val-on-held-out",
        action="store_true",
        help="Use all held-out test-subject clips for val (paper/IMU LOSO style)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--sampling-time-ms", type=int, default=20)
    parser.add_argument("--sample-length-ms", type=int, default=2000)
    parser.add_argument(
        "--window-stride-ms",
        type=int,
        default=0,
        help="Sliding window stride in ms (0 = one window per H5 clip)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Train-time event dropout / spatial jitter / flip",
    )
    parser.add_argument(
        "--normalize-clip",
        action="store_true",
        help="Divide each clip by its max count before the model",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Balanced cross-entropy class weights from train windows",
    )
    parser.add_argument(
        "--majority-vote-k",
        type=int,
        default=0,
        help="Sliding majority vote over k windows at eval (0=off, 10≈10s)",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 = off (paper runs)")
    parser.add_argument("--early-stop-min-epochs", type=int, default=10)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache preprocessed tensors (.npy). Default: <out-dir>/tensor_cache",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable tensor cache (on-the-fly H5 -> myCount each epoch)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint_last.pt or model.pt in out-dir",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _data_config(args: argparse.Namespace) -> DataConfig:
    kwargs: dict = dict(
        dataset_root=args.dataset_base,
        sampling_time_ms=args.sampling_time_ms,
        sample_length_ms=args.sample_length_ms,
        window_stride_ms=args.window_stride_ms,
        normalize_clip=args.normalize_clip,
        augment_train=args.augment,
        seed=args.seed,
        events_tier=args.events_tier,
    )
    if args.events_tier == "dvs128":
        kwargs.update(
            sensor_height=128,
            sensor_width=128,
            ds_factor=1,
            model_width=128,
        )
    return DataConfig(**kwargs)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if args.no_cache:
        cache_dir = None
    elif args.cache_dir is not None:
        cache_dir = args.cache_dir
    else:
        cache_dir = out_dir / "tensor_cache"
    cfg = ExperimentConfig(
        data=_data_config(args),
        model=ModelConfig(
            model_name=args.model,
            channels=args.channels,
            dropout=args.dropout,
            feat_neurons=args.feat_neurons,
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_epochs=args.early_stop_min_epochs,
            cache_dir=cache_dir,
            weight_decay=args.weight_decay,
            class_weights=args.class_weights,
            majority_vote_k=args.majority_vote_k,
            device=args.device,
        ),
        out_dir=out_dir,
    )

    trainer = V2EAicoTrainer(cfg)
    results = trainer.train_loso(
        dataset_base=args.dataset_base,
        train_subjects=args.train_subjects,
        test_subject=args.test_subject,
        val_ratio=args.val_ratio,
        val_on_held_out=args.val_on_held_out,
        resume_from=args.resume,
    )

    print("\nFinished.")
    print(f"Best epoch: {results['best_epoch'] + 1}")
    if args.val_on_held_out:
        print(f"Val (sbj_{args.test_subject}, held-out subject): {results['val']}")
    else:
        print(f"Val ({args.val_ratio:.0%} holdout from train subjects): {results['val']}")
    print(f"Test (sbj_{args.test_subject}): {results['test']}")
    if "test_vote" in results:
        print(f"Test voted (k={args.majority_vote_k}): {results['test_vote']}")
    print(f"Output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
