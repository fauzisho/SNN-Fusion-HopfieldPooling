#!/usr/bin/env python3
"""Build the held-out aligned feature cache that every SIM B evaluation reads.

For one LOSO fold (seed, held-out subject) this runs the frozen event SNN and IMU
SNN on the held-out subject, aligns event clips and inertial windows (nearest window
within 1.0 s, labels must agree, >= 60 % label overlap) and stores the 20-D features
per modality (18 class probabilities + max probability + entropy) with labels and
window centre times:

    $NWEAR_OUTPUT/simb/feature_cache/seed_<s>/sbj_<k>/aligned_val_with_times.pt

The alignment itself lives in ``dvs_imu_fusion_data.collect_aligned_split``.

    python simb/build_aligned_cache.py --seed 1 --val-subject 0 --device cuda:0
"""

from __future__ import annotations

import argparse

import torch

import nwear_paths as P

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)
from dvs_imu_fusion_data import collect_aligned_split  # noqa: E402
from simb_common import save_subj_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--val-subject", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()

    seed, sbj = args.seed, args.val_subject
    cache = P.SIMB_CACHE / f"seed_{seed}" / f"sbj_{sbj}" / "aligned_val_with_times.pt"
    if args.skip_if_done and cache.is_file():
        print(f"exists: {cache}")
        return
    device = torch.device(args.device)
    aligned = collect_aligned_split(
        seed=seed,
        val_subject=sbj,
        dvs_ckpt=P.EVENT_RUNS / f"seed_{seed}" / f"test_sbj_{sbj}" / "model.pt",
        imu_ckpt=P.IMU_RUNS / f"seed_{seed}" / f"train_split_{sbj + 1}" / "model.pt",
        feat_dir=P.IMU_FEAT_DIR,
        ann_path=P.ANN_DIR / f"wear_split_{sbj + 1}.json",
        dataset_base=P.DATA_ROOT,
        dvs_cache_dir=P.EVENT_TENSOR_CACHE,
        max_time_diff=1.0,
        imu_hidden=128,
        min_overlap=0.6,
        device=device,
        batch_size=args.batch_size,
        subject_ids=[sbj],
    )
    subj = next(s for s in aligned.subjects if s.is_val)
    save_subj_cache(cache, subj)
    print(f"Wrote {cache} n={len(subj.labels)}", flush=True)


if __name__ == "__main__":
    main()
