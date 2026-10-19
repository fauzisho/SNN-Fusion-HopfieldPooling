# Known issues and deviations

The submitted thesis and the paper are the reference: this repository reproduces their
numbers exactly (`results/`, `evaluation/build_table1.py`), and the code is kept as it
was run. The points below are notes for anyone who re-runs or extends the pipeline;
they describe how the reported results were produced, not changes to them.
Ordered by relevance for future work.

## 1. Subjects 16 and 17 in the SIM B evaluation

The event data itself is complete: subjects 16 and 17 were re-converted on 2026-08-11
(`sbj_16_events_dvs128_new`, `sbj_17_events_dvs128_new`: 829 and 855 clips, all 18
activities, cut with the 60 fps annotations). The first conversion
(`sbj_16_events_dvs128`, `sbj_17_events_dvs128`) contains only **12 and 11 of the 18
activities** (several clip downloads were empty), and **the SIM B results were computed
on that first conversion**:

* `simb/dvs_imu_fusion_data.py` and `simb/eval_simb_native_fusion.py` build the path
  `sbj_<k>_events_dvs128` without the `_new` suffix, so the aligned feature caches, the
  SIM B heads and all SIM B evaluations used the incomplete folders (the caches of the
  folds that hold out 16 / 17 contain 12 / 11 classes; held-out subject 17 scores exactly
  11/18 = 61.11 % F1 for every head after the vote). The copies of these results on S3
  (`chapter 4/vote10_native_vs_aligned/`) are the same files.
* Macro F1 is averaged over all 18 classes, and a class without ground truth scores 0,
  so the 6 folds that hold out subject 16 or 17 cap F1 at 66.7 % and 61.1 %. Every SIM B
  F1 (and every aligned-window F1, SIM A included) is about 4 points lower than the
  per-class performance. Accuracy is not affected. Rankings within a table do not change.
* Because a missing class always contributes 0, the exact correction without new runs is
  `F1_present = F1_18 x 18 / n_present_classes` per fold.
* The code in `simb/` is kept as run, so it reproduces the reported numbers. To use the
  complete conversion in new work, resolve the folder with
  `event_snn.dataset.loso.subject_events_dir(dataset_base, k, "dvs128")` in those two
  scripts and re-run SIM B (about 20 GPU-minutes per fold).
* **Mixed data in the event SNN runs.** `event_snn/dataset/loso.py` silently prefers a
  `*_new` folder when one exists. The `_new` folders appeared on 2026-08-11, after folds
  0-15 had been trained (with the incomplete 16/17 as training data). The folds that hold
  out 16 and 17 were trained later and therefore saw the `_new` data. **A re-run today
  uses `_new` everywhere** and will not reproduce the SIM B numbers exactly.

## 2. IMU SNN input layout differs from the paper text

The WEAR inertial windows (`sbj_k.npy`, 1200 values per row) are stored channel-major
(12 channels x 100 samples). `imu_snn/pipeline_inertial.py: load_npy` reshapes them as
`(N, 100, 12)` without a transpose, so each of the 100 "time steps" the EGRU sees is 12
consecutive samples of one axis, and the per-"channel" standardisation runs over
position k mod 12. The paper describes 12 standardised channels x 100 samples.

All IMU results (training, evaluation, fusion features, NeuroBench) use the same loader,
so they are internally consistent. Fixing the reshape (`reshape(N, 12, 100).transpose(0, 2, 1)`)
changes the model and requires retraining; it is the first thing to try for a better
IMU SNN.

## 3. IMU normalisation in fusion excludes NULL

Training standardises with statistics over all training windows (NULL included).
`simb/dvs_imu_fusion_data.py: load_imu_train_norm` recomputes them over activity windows
only, so the fusion features come from slightly differently normalised inputs.

## 4. Event SNN checkpoint is selected on the held-out subject

`--val-on-held-out` makes the validation set equal to the held-out test subject, and the
trainer keeps the epoch with the best validation accuracy as `model.pt` (best epochs
10-30). The paper's statement that the held-out subject never selects a model holds for
the fusion heads, not for the event encoder. For new work, select on a held-out training
subject or keep the last epoch (`model_last.pt` is saved too).

The IMU SNN keeps its last epoch (no selection).

## 5. SIM A single Hopfield head: training details differ from the text

`sima/train_hopfield_late_fusion_tridet.py` is not seeded, saves the **last** epoch (not
the lowest training loss), and uses hflayers' default pooling dropout 0.0 with pooling
hidden size 32. The double heads (SIM A and SIM B) keep the lowest-training-loss epoch,
use dropout 0.1, pooling hidden 64 and gradient clipping 1.0.

## 6. BF1 duty cycle uses a voted (non-causal) A&D stream

The SIM A BF1 duty-cycle home stream is `fusion_preds.npz["ad_preds"]`, i.e. A&D after a
centred 501-frame majority vote (`--mv-kernel 501` in `pipeline/05_sima_heads.sh`). The
paper describes the gating rule as causal; for BF1 the input stream is not. The always-on
BF1 score-pick value of Table IV (77.58) is frame-level with the voted A&D. Table I uses
raw frames (no vote on A&D).

## 7. Clip cutting was not recovered

The script that cut the WEAR videos into per-segment 2 s clips was not found.
`data_preparation/cut_wear_clips.py` is a reimplementation that reproduces the naming the
pipeline needs. The paper's clips were cut with the 30 fps annotations (start times such
as `281.04`); the `_new` re-conversions of subjects 16/17 used the 60 fps annotations.
The paper's clips also went through a 240x180 intermediate (two lossy encodes:
1920x1080 -> 240x180 -> 128x128). New conversions should go straight to 128x128
(`clips_to_dvs128.sh`).

## 8. v2e settings

v2e warns that `--cutoff_hz 200` has no effect at the Super SloMo sample rate (~120 Hz,
about 2x upsampling). The H5 files hold `events` as `(N, 4)` uint32 with columns
`(t_us, x, y, p)`; older notes that say `(x, y, p, t)` are wrong.

## 9. Things that only exist as outputs

* The SIM A baseline checkpoints (A&D, camera TriDet) were moved off the compute
  machine; only their 50 Hz frame dumps remain. The inertial TriDet checkpoints exist.
* The TriDet frame dumps were first exported with score threshold 0.0 (NULL never
  predicted) and re-thresholded at 0.1 with `sima/patch_tridet_frames_threshold.py`.
  `sima/infer_tridet_frames.py` now defaults to 0.1, so fresh exports need no patch.
* Three Table I scoring scripts existed only as one-off scripts and were recovered:
  `evaluation/eval_simb_aligned_vote10.py`, `evaluation/eval_sima_fusion_aligned_vote10.py`,
  `evaluation/eval_sima_fusion_native_vote10.py`.
* The sub-rows of Table II (conv. + AdLIF, EGRU to end) were derived by hand from the
  per-run NeuroBench CSV; `evaluation/build_energy_table.py` rebuilds the main rows.
* `simb/benchmark_fusion_neurobench.py` uses analytical head MACs (85,632 single,
  1,797,376 double); its derivation is in the script header.

## 10. Open discrepancy with the thesis

A&D with an 18-way argmax scores 77.87 % aligned F1 in the paper and 77.64 % in the
thesis appendix. Both claim the same rule; the 0.23-point gap is not explained.

## 11. Training-length provenance

The event SNN Slurm script defaulted to 20 epochs with patience 8, but all 54 runs ran 30
epochs without early stopping (later submit scripts and resumes). The release uses 30 /
no early stop, which matches every saved `summary.txt`.
