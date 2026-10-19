# Reproducing the paper

Three levels, from minutes to weeks of GPU time.

| Level | What you get | Needs |
| --- | --- | --- |
| A. Tables from shipped results | Table I, SIM B mAP, Table II main rows | Python + numpy only |
| B. Re-score frozen models | every number from the saved checkpoints / frame dumps | released checkpoints, WEAR data |
| C. Full re-run | everything from WEAR video | 1 GPU per fold, ~15 GB of events, ~110 GB tensor cache |

## 0. Setup

```bash
git clone https://github.com/fauzisho/SNN-Fusion-HopfieldPooling.git
cd SNN-Fusion-HopfieldPooling
python -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r environment/requirements.txt
# edit the paths at the top of setup_env.sh (or export them), then
source setup_env.sh
```

The paper runs used the Singularity image built from `environment/wearsnn_v2e.def`
(PyTorch 2.2.2, CUDA 12.1, Python 3.10). Run it with `--cleanenv` and
`PYTHONNOUSERSITE=1` so packages in `~/.local` do not leak in.

**WEAR (SIM A only).** Clone the official repository at the commit the patch was made
against and apply the local changes:

```bash
git clone https://github.com/mariusbock/wear.git third_party/wear
cd third_party/wear && git checkout 65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba
git apply ../../sima/wear_patches/wear_local.patch
mkdir -p cluster && cp ../../sima/wear_patches/run_*_split.py cluster/
cd camera_baseline/tridet/libs/utils && python setup.py build_ext --inplace   # TriDet NMS
```

**Event data.** The converted DVS128 clips (18 subjects, 16.5 GB) are public: [download page](https://thesiswearsnn.nbg1.your-objectstorage.com/n-wear-events-dvs128/index.html), or

```bash
bash data_preparation/download_events.sh          # into $NWEAR_DATA, verifies SHA-256
```

With these, step `01_convert_events.sh` can be skipped.

**Data** (`$NWEAR_DATA`, layout in `nwear_paths.py`): download WEAR
(https://mariusbock.github.io/wear/): the 60 fps annotations, the processed inertial
features `50hz/120_frames_60_stride`, the I3D features `60fps/120_frames_60_stride`
and, for event conversion, the raw videos. Place `SuperSloMo39.ckpt` (linked from the
v2e README; md5 `0fd4fb0a1e002d2aa8bd7c7e01001626`) in `$NWEAR_DATA`.

## A. Tables from the shipped results (no GPU)

```bash
python evaluation/build_table1.py                    # Table I + SIM B mAP
python evaluation/build_energy_table.py --from-shipped  # Table II main rows
```

`results/` holds one row per fold (54) for every Table I cell, the per-run NeuroBench
counts of both encoders, the SIM A energy summary and the duty-cycle summaries
(Table IV: `duty_cycle_simb_float_loso54.json`, `duty_cycle_sima_*_loso54.json`).

## B. Re-score the frozen models

Put the released checkpoints under `$NWEAR_OUTPUT` in the layout of `nwear_paths.py`
(`event_snn/loso54/seed_<s>/test_sbj_<k>/model.pt`,
`imu_snn/loso54/seed_<s>/train_split_<k+1>/model.pt`, `sima/bf{1,2}_*/seed_<s>/split_<k+1>/`,
`simb/heads/seed_<s>/sbj_<k>/`), then run `pipeline/08_aggregate.sh`. The aligned feature
cache is rebuilt per fold with `simb/build_aligned_cache.py` (GPU, minutes per fold).

## C. Full re-run

Each per-fold step takes `<seed> <held-out subject>` or runs as a Slurm array
(task t -> seed `(t-1)/18+1`, subject `(t-1)%18`):

```bash
VIDEO=/path/sbj_3.mp4 bash pipeline/01_convert_events.sh 3       # per subject, 0-17
sbatch --array=1-54 cluster/array.sbatch pipeline/02_train_event_snn.sh
sbatch --array=1-54 cluster/array.sbatch pipeline/03_train_imu_snn.sh
sbatch --array=1-54 cluster/array.sbatch pipeline/04_sima_baselines.sh
sbatch --array=1-54 cluster/array.sbatch pipeline/05_sima_heads.sh
sbatch --array=1-54 cluster/array.sbatch pipeline/06_simb_heads.sh   # after 02 and 03
sbatch --array=1-54 cluster/array.sbatch pipeline/07_energy_fold.sh
bash pipeline/08_aggregate.sh                                        # after all folds
```

Without Slurm: `bash pipeline/02_train_event_snn.sh 1 0` and loop over seeds 1-3 and
subjects 0-17.

Expect small differences from the paper, since GPU non-determinism affects every
training run.

## Where each paper number comes from

| Paper | Script | Output |
| --- | --- | --- |
| Table I, SIM A single modality | `evaluation/eval_sima_single_modality_vote10.py` | `results_sima_single_modality_vote10.csv` |
| Table I, SIM A fusion, native | `evaluation/eval_sima_fusion_native_vote10.py` | `results_sima_fusion_native_vote10.csv` |
| Table I, SIM A fusion, aligned | `evaluation/eval_sima_fusion_aligned_vote10.py` | `results_sima_fusion_aligned_vote10.csv` |
| Table I, SIM B, native | `simb/eval_simb_native_fusion.py` (+ `--combine`) | `results_simb_native_vote10.csv` |
| Table I, SIM B, aligned | `evaluation/eval_simb_aligned_vote10.py` | `results_simb_fusion_vote10.csv` |
| SIM B mAP (69.54 / 63.21 / 59.89) | `simb/eval_float_fusion_simb_map.py` | `results_simb_fusion_pr_map.csv` |
| Table II, encoders | `event_snn/benchmark_dvs_neurobench.py`, `event_snn/profile_dvs_compute_spikes.py`, `imu_snn/benchmark_imu_neurobench.py` | `energy_neurobench_*.csv` |
| Table II, SIM A rows + SIM A heads | `sima/benchmark_sima_baseline_energy.py` | `energy_sima_summary.json` |
| Table II, SIM B heads | `simb/benchmark_fusion_neurobench.py` (analytical) | — |
| Table III | Table II + literature bounds in the paper (I3D 216 GFLOPs, RGB 0.5-2 W, DVS 23 mW) | — |
| Table IV, SIM A | `sima/eval_bf{1,2}_duty_cycle_loso54.py`, `sima/eval_hopfield_duty_cycle_loso54.py` | `duty_cycle_sima_*.json` |
| Table IV, SIM B | `simb/eval_simb_duty_cycle_loso54.py` | `duty_cycle_simb_float_loso54.json` |
