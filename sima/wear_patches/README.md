# WEAR patches (SIM A baselines)

The conventional baselines (Attend-and-Discriminate, camera TriDet, inertial TriDet) are
trained with the official WEAR code, which is licensed CC BY-NC-SA 4.0 and is therefore
not copied into this repository. Apply these changes to a checkout of
https://github.com/mariusbock/wear at commit `65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba`:

```bash
git clone https://github.com/mariusbock/wear.git "$WEAR_REPO"
cd "$WEAR_REPO" && git checkout 65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba
git apply /path/to/SNN-Fusion-HopfieldPooling/sima/wear_patches/wear_local.patch
mkdir -p cluster && cp /path/to/SNN-Fusion-HopfieldPooling/sima/wear_patches/run_*_split.py cluster/
```

| File | Change |
| --- | --- |
| `wear_local.patch` → `main.py` | optional neptune import; `WEAR_LOG_ROOT` overrides the log directory; tolerate splits without validation windows |
| `wear_local.patch` → `inertial_baseline/train.py` | always save `ckpts/final_<split>.pth.tar` |
| `wear_local.patch` → `camera_baseline/tridet/main.py` | save a final checkpoint with `state_dict` and `state_dict_ema` |
| `run_ad_split.py`, `run_tridet_split.py` (new) | train one LOSO split for one seed into `--out-dir`; expand `${NWEAR_DATA}` in the configs |

The configs in `sima/configs/` are the 2 s variants used in the paper (A&D: 100 samples
at 50 Hz, 50 % overlap, kernel 21, lr 1e-4, 100 epochs; TriDet: 120 frames / stride 60,
AdamW 1e-4, 100 epochs).
