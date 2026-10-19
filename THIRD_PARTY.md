# Third-party code and data

| Component | Where | Origin | Licence |
| --- | --- | --- | --- |
| EGRU (event-based GRU) | `event_snn/models/egru.py`, inline copy in `imu_snn/pipeline_inertial.py` | K. K. Nazeer (2023), incorporating LMNT haste (2020); Subramoney et al., ICLR 2023 | Apache-2.0 (header kept) |
| SE-AdLIF neuron | `event_snn/models/adlif.py` | after Baronig et al. 2025 (IGITUGraz/SE-adlif) | see upstream |
| WEAR baselines, `ANETdetection`, `combine_predictions` | not copied; used from `$WEAR_REPO` (patch in `sima/wear_patches/`) | Bock et al., https://github.com/mariusbock/wear | CC BY-NC-SA 4.0 |
| Hopfield layers | pip `hopfield-layers` | Ramsauer et al., https://github.com/ml-jku/hopfield-layers | see upstream |
| SpikingJelly | pip `spikingjelly` | Fang et al. | see upstream |
| NeuroBench 2.3.0 | pip / `$NEUROBENCH_ROOT` (unmodified) | Yik et al. | Apache-2.0 |
| v2e, Super SloMo checkpoint | pip / `$NWEAR_DATA` | SensorsINI | see upstream |

**Data.** WEAR is released under CC BY-NC-SA 4.0. The event version produced by
`data_preparation/` is a derivative work and must be shared under the same licence with
attribution to WEAR.
