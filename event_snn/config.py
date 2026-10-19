"""Default experiment configuration (emmanuela-aligned for DVS240 v2e)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import nwear_paths as P


@dataclass
class DataConfig:
    dataset_root: Path = P.DATA_ROOT / "sbj_0_events_dvs240"
    events_tier: str = "dvs240"  # "dvs240" | "dvs128" — H5 folder suffix
    sensor_height: int = 180
    sensor_width: int = 240
    ds_factor: int = 2
    model_width: int = 160  # letterbox pad 120 -> 160 (emmanuela AICO width)
    sampling_time_ms: int = 20
    sample_length_ms: int = 2000
    window_stride_ms: int = 0
    normalize_clip: bool = False
    augment_train: bool = False
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    seed: int = 1

    @property
    def input_height(self) -> int:
        return self.sensor_height // self.ds_factor

    @property
    def input_width(self) -> int:
        return self.model_width

    @property
    def time_bins(self) -> int:
        return self.sample_length_ms // self.sampling_time_ms


@dataclass
class ModelConfig:
    model_name: str = "allconv_adlif_egru_dvs128"
    channels: int = 8
    dropout: float = 0.05
    num_classes: int = 18
    feat_neurons: int = 512

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        """Rebuild from a checkpoint's saved config, ignoring keys of older model variants."""
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 60
    learning_rate: float = 0.005
    num_workers: int = 4
    prefetch_factor: int = 2
    early_stop_patience: int = 0
    early_stop_min_epochs: int = 5
    cache_dir: Path | None = None
    weight_decay: float = 0.0
    class_weights: bool = False
    majority_vote_k: int = 0
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: Path = Path(__file__).resolve().parent / "output" / "sbj0_pilot"
