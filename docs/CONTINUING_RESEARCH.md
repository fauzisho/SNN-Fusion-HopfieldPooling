# Continuing this research

A guide for whoever picks this up next: how the pieces fit, what to consider first, and
which directions the paper leaves open.

## 1. Mental model of the code

```
WEAR video ──cut/resize/v2e──► event clips (H5) ──► event_snn ──┐
                                                                ├─► frozen 18-class scores ─► simb heads ─► evaluation
WEAR inertial windows (.npy) ───────────────────────► imu_snn ──┘          (20-D / window)

WEAR I3D + inertial ──► WEAR baselines (A&D, TriDet) ──► 50 Hz frame dumps ─► sima heads ─► evaluation
```

* **Everything downstream of the encoders works on frozen outputs.** Changing a fusion
  head never needs encoder retraining; changing an encoder means rebuilding the aligned
  cache (`simb/build_aligned_cache.py`) and retraining the heads.
* **A fold is (seed, held-out subject).** Every stage writes
  `seed_<s>/{test_sbj_<k> | train_split_<k+1> | split_<k+1> | sbj_<k>}`. The naming
  differs per stage for historical reasons; `nwear_paths.py` documents all of them.
* **Two window sets, two vote settings.** Native windows (2 s / 1 s stride on the
  inertial timeline) and aligned windows (event clip and inertial window coincide). Report
  both, with and without the 10 s vote: the vote reorders the heads (paper, Sec. IV-B).
* **Macro F1 is over all 18 classes.** Read `KNOWN_ISSUES.md` item 1 before comparing
  F1 across folds or papers.

## 2. Before building on the results

The reported results (thesis and paper) are reproduced exactly by this repository. For
new experiments, consider these first (details in `KNOWN_ISSUES.md`):

1. **Event data of subjects 16/17** (item 1). The SIM B stage read the first conversion;
   new runs can use the complete `_new` folders by resolving the folder with
   `subject_events_dir`. Expect different SIM B numbers than reported.
2. **IMU input layout** (item 2). Transposing the channel-major windows and retraining
   is the obvious first improvement of the IMU SNN.
3. **Model selection** (item 4). Select the event checkpoint on a validation subject
   from the training set, or keep the last epoch.
4. **Clip cutting** (item 7). `data_preparation/cut_wear_clips.py` is a reimplementation;
   verify its boundaries before regenerating the event dataset.

## 3. Open research directions

From the paper's discussion and the thesis:

* **Physical DVS.** All events are simulated with v2e from 60 fps video. Validate the
  event encoder on a real head-mounted DVS recording; the v2e settings are in
  `data_preparation/clips_dvs128_to_v2e.sh`.
* **Neuromorphic hardware.** Energies are operation-count proxies
  (E_MAC 4.6 pJ, E_SOP 10 pJ, E_neuron 5 pJ). Port the encoders to Loihi 2 / Lava or
  SpiNNaker 2 and measure. The EGRU and SE-AdLIF need custom neuron models there.
* **A spiking fusion head.** The Hopfield heads are float attention on a host. The thesis
  explored spiking Hopfield pooling (not included here); the fusion head is 0.11 % of the
  pipeline energy, so the motivation is deployability rather than energy.
* **A cheaper event encoder.** The event SNN is 99 % of the pipeline cost, and the
  EGRU-512 head is two thirds of it (Table II). Smaller recurrent heads, lower input
  resolution or event-driven convolution are the main levers.
* **Gating.** The IMU trigger thresholds (entropy 0.05, max-prob 0.90, K = 4) were tuned
  on SIM A and reused. Tune them on SIM B, learn the trigger, or gate the DVS sensor
  itself (about 9.6 mJ of the 47 mJ sensor budget per window).
* **NULL / rest rejection.** Events exist only inside annotated activities, so all
  scores exclude NULL. Converting NULL segments too would allow the full 19-class
  WEAR protocol.
* **Temporal context vs. vote.** Double Hopfield pooling and the 10 s vote buy the same
  context. A causal temporal head (past windows only) would make the comparison fair to
  a deployed device.

## 4. Adding something new

* **A new fusion head (SIM B):** add the module next to `SingleHopfieldDVSIMU` /
  `DoubleHopfieldDVSIMU` in `simb/train_hopfield_dvs_imu_fusion.py`, train it per fold,
  and add its predictions to `evaluation/eval_simb_aligned_vote10.py` and
  `simb/eval_simb_native_fusion.py` so it is scored on both window sets with the same
  vote rules. `evaluation/build_table1.py` picks up new columns by name.
* **A new event encoder:** implement it in `event_snn/models/`, register it in
  `event_snn/training/trainer.py: build_model` and `train_loso.py --model`, and make
  `simb/dvs_imu_fusion_data.py: load_dvs_checkpoint` construct it. The fusion code only
  needs 18 class logits per window.
* **Energy of a new model:** wrap it like `event_snn/neurobench_integration/` (hook the
  connection layers, register custom neurons in NeuroBench's neuron-op table, add any
  recurrence analytically) so operators NeuroBench does not count are not silently zero.

## 5. Compute notes

* Original hardware: Slurm cluster with RTX 2080 Ti GPUs (11 GB), one GPU per fold.
* The event tensor cache (float16, T = 100, 128 x 128) is ~110 GB for 18 subjects;
  the converted H5 events are ~15 GB.
* All heads train in minutes; the event SNN dominates training time.
