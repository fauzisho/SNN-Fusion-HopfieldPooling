"""Training and evaluation loop for event_snn."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from event_snn.config import ExperimentConfig
from event_snn.dataset.dvs240_window_dataset import DVS240WindowDataset
from event_snn.dataset.loso import loso_train_test_split
from event_snn.dataset.splits import SplitIndex, discover_h5_files, stratified_split
from event_snn.models.allconv_adlif_egru_dvs128 import AllConvAdLIFEGRU_DVS128
from event_snn.training.losses import calc_ce_loss, logits_to_pred
from event_snn.training.postprocess import apply_majority_vote, macro_metrics

# Only the paper model is shipped; the thesis also tried PLIF / AICO / Hopfield variants.
V2EModel = AllConvAdLIFEGRU_DVS128


def _collate_with_meta(batch: list[tuple]) -> tuple:
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    keys = [item[2] for item in batch]
    times = [item[3] for item in batch]
    return xs, ys, keys, times


class V2EAicoTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._class_weight_tensor: torch.Tensor | None = None

    def build_splits(self) -> SplitIndex:
        files = discover_h5_files(self.cfg.data.dataset_root)
        return stratified_split(
            files,
            train_ratio=self.cfg.data.train_ratio,
            val_ratio=self.cfg.data.val_ratio,
            test_ratio=self.cfg.data.test_ratio,
            seed=self.cfg.data.seed,
        )

    def build_loso_splits(
        self,
        dataset_base: Path,
        train_subjects: list[int],
        test_subject: int,
        val_ratio: float = 0.10,
        val_on_held_out: bool = False,
    ) -> tuple[SplitIndex, list[int], int]:
        return loso_train_test_split(
            dataset_base,
            train_subjects=train_subjects,
            test_subject=test_subject,
            events_tier=self.cfg.data.events_tier,
            val_ratio=val_ratio,
            val_on_held_out=val_on_held_out,
            seed=self.cfg.data.seed,
        )

    def _make_dataset(
        self,
        files: list[Path],
        *,
        augment: bool,
        return_meta: bool,
    ) -> DVS240WindowDataset:
        return DVS240WindowDataset(
            files,
            self.cfg.data,
            augment=augment,
            normalize=self.cfg.data.normalize_clip,
            cache_dir=self.cfg.train.cache_dir,
            return_meta=return_meta,
            seed=self.cfg.data.seed,
        )

    def _compute_class_weights(self, train_ds: DVS240WindowDataset) -> torch.Tensor:
        labels = np.array([sample.label for sample in train_ds.samples], dtype=np.int64)
        classes = np.arange(self.cfg.model.num_classes)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def build_loaders(self, split: SplitIndex) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_cfg = self.cfg.data
        cache_dir = self.cfg.train.cache_dir
        train_ds = self._make_dataset(split.train, augment=data_cfg.augment_train, return_meta=False)
        val_ds = self._make_dataset(split.val, augment=False, return_meta=self.cfg.train.majority_vote_k > 0)
        test_ds = self._make_dataset(split.test, augment=False, return_meta=self.cfg.train.majority_vote_k > 0)

        if self.cfg.train.class_weights:
            self._class_weight_tensor = self._compute_class_weights(train_ds)

        nw = self.cfg.train.num_workers
        loader_kwargs: dict = dict(
            batch_size=self.cfg.train.batch_size,
            num_workers=nw,
            pin_memory=self.device.type == "cuda",
        )
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.cfg.train.prefetch_factor

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_kwargs = dict(loader_kwargs)
        if self.cfg.train.majority_vote_k > 0:
            val_kwargs["collate_fn"] = _collate_with_meta
        val_loader = DataLoader(val_ds, shuffle=False, **val_kwargs)
        test_kwargs = dict(loader_kwargs)
        if self.cfg.train.majority_vote_k > 0:
            test_kwargs["collate_fn"] = _collate_with_meta
        test_loader = DataLoader(test_ds, shuffle=False, **test_kwargs)
        return train_loader, val_loader, test_loader

    def build_model(self) -> V2EModel:
        if self.cfg.model.model_name != "allconv_adlif_egru_dvs128":
            raise ValueError(
                f"model {self.cfg.model.model_name!r} is not part of this release; "
                "use --model allconv_adlif_egru_dvs128"
            )
        model = AllConvAdLIFEGRU_DVS128(
            channels=self.cfg.model.channels,
            feat_neurons=self.cfg.model.feat_neurons,
            classes=self.cfg.model.num_classes,
            dropout=self.cfg.model.dropout,
            input_height=self.cfg.data.input_height,
            input_width=self.cfg.data.input_width,
            device=self.device,
        )
        print(f"Model: {model.__class__.__name__}")
        return model.to(self.device)

    def _run_epoch(
        self,
        model: V2EModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> tuple[float, float]:
        is_train = optimizer is not None
        model.train(is_train)
        losses: list[float] = []
        preds: list[int] = []
        labels: list[int] = []

        for batch in loader:
            batch_x = batch[0].to(self.device, non_blocking=True)
            batch_y = batch[1].to(self.device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = calc_ce_loss(logits, batch_y, class_weights=self._class_weight_tensor)
            if is_train:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            pred = logits_to_pred(logits)
            preds.extend(pred.detach().cpu().tolist())
            labels.extend(batch_y.detach().cpu().tolist())

        acc = accuracy_score(labels, preds) if labels else 0.0
        return float(sum(losses) / max(len(losses), 1)), float(acc)

    @staticmethod
    def _macro_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        return macro_metrics(y_true, y_pred)

    def evaluate(self, model: V2EModel, loader: DataLoader) -> dict[str, float]:
        model.eval()
        preds: list[int] = []
        labels: list[int] = []
        segment_keys: list[str] = []
        center_us: list[int] = []
        with_meta = self.cfg.train.majority_vote_k > 0

        with torch.no_grad():
            for batch in loader:
                if with_meta:
                    batch_x, batch_y, batch_keys, batch_times = batch
                else:
                    batch_x, batch_y = batch
                    batch_keys, batch_times = None, None
                batch_x = batch_x.to(self.device, non_blocking=True)
                logits = model(batch_x)
                pred = logits_to_pred(logits)
                preds.extend(pred.cpu().tolist())
                labels.extend(batch_y.tolist())
                if with_meta:
                    segment_keys.extend(batch_keys)
                    center_us.extend(batch_times)

        metrics = self._macro_metrics(labels, preds)
        if with_meta and self.cfg.train.majority_vote_k > 1:
            voted_preds, voted_labels = apply_majority_vote(
                segment_keys,
                center_us,
                preds,
                labels,
                self.cfg.train.majority_vote_k,
            )
            vote_metrics = self._macro_metrics(voted_labels, voted_preds)
            metrics["vote"] = vote_metrics
        return metrics

    def train(
        self,
        split: SplitIndex | None = None,
        *,
        loso_meta: tuple[list[int], int] | None = None,
        resume_from: Path | None = None,
    ) -> dict[str, object]:
        if split is None:
            split = self.build_splits()
        train_loader, val_loader, test_loader = self.build_loaders(split)
        model = self.build_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        last_state: dict[str, torch.Tensor] | None = None
        last_epoch = -1
        patience_left = self.cfg.train.early_stop_patience
        start_epoch = 0

        if resume_from is not None:
            resume_path = Path(resume_from)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device)
            model.load_state_dict(ckpt["state_dict"])
            best_epoch = int(ckpt.get("best_epoch", -1))
            best_val_acc = float(ckpt.get("best_val_acc", -1.0))
            last_epoch = int(ckpt.get("last_epoch", best_epoch))
            if ckpt.get("best_state") is not None:
                best_state = ckpt["best_state"]
            if ckpt.get("last_state") is not None:
                last_state = ckpt["last_state"]
            if ckpt.get("history") is not None:
                history = ckpt["history"]
            if ckpt.get("patience_left") is not None:
                patience_left = int(ckpt["patience_left"])
            start_epoch = last_epoch + 1
            print(
                f"Resuming from {resume_path.name}: "
                f"last epoch {last_epoch + 1}, best epoch {best_epoch + 1}, "
                f"continuing at epoch {start_epoch + 1}"
            )

        train_ds = train_loader.dataset
        print(
            f"Split sizes: train={len(train_ds)} val={len(val_loader.dataset)} "
            f"test={len(test_loader.dataset)}"
        )
        print(
            f"Input (H,W,T)=({self.cfg.data.input_height},{self.cfg.data.input_width},"
            f"{self.cfg.data.time_bins}) bin={self.cfg.data.sampling_time_ms}ms "
            f"stride={self.cfg.data.window_stride_ms}ms tier={self.cfg.data.events_tier}"
        )
        print(
            f"Loader: batch={self.cfg.train.batch_size} workers={self.cfg.train.num_workers} "
            f"cache={self.cfg.train.cache_dir}"
        )
        print(
            f"Generalization: augment={self.cfg.data.augment_train} "
            f"normalize={self.cfg.data.normalize_clip} "
            f"weight_decay={self.cfg.train.weight_decay} "
            f"class_weights={self.cfg.train.class_weights} "
            f"majority_vote_k={self.cfg.train.majority_vote_k}"
        )
        if self.cfg.train.early_stop_patience > 0:
            print(
                f"Early stop: patience={self.cfg.train.early_stop_patience} "
                f"min_epochs={self.cfg.train.early_stop_min_epochs}"
            )

        for epoch in range(start_epoch, self.cfg.train.num_epochs):
            train_loss, train_acc = self._run_epoch(model, train_loader, optimizer)
            val_loss, val_acc = self._run_epoch(model, val_loader, None)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            last_epoch = epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if self.cfg.train.early_stop_patience > 0:
                    patience_left = self.cfg.train.early_stop_patience
            elif self.cfg.train.early_stop_patience > 0:
                patience_left -= 1
            print(
                f"epoch {epoch + 1:03d}/{self.cfg.train.num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )
            self._save_training_checkpoint(
                model,
                best_state=best_state,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                last_state=last_state,
                last_epoch=last_epoch,
                history=history,
                patience_left=patience_left,
            )
            if (
                self.cfg.train.early_stop_patience > 0
                and epoch + 1 >= self.cfg.train.early_stop_min_epochs
                and patience_left <= 0
            ):
                print(
                    f"Early stop at epoch {epoch + 1} "
                    f"(best epoch {best_epoch + 1}, val acc {best_val_acc:.3f})"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_metrics = self.evaluate(model, val_loader)
        test_metrics = self.evaluate(model, test_loader)

        split_sizes = {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }

        ckpt_payload = self._checkpoint_payload(
            model,
            best_state=best_state,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            last_state=last_state,
            last_epoch=last_epoch,
            history=history,
            patience_left=patience_left,
        )
        torch.save(ckpt_payload, self.out_dir / "model.pt")
        if last_state is not None:
            last_payload = dict(ckpt_payload)
            last_payload["state_dict"] = last_state
            last_payload["best_epoch"] = last_epoch
            torch.save(last_payload, self.out_dir / "model_last.pt")
        self._save_summary(
            split,
            best_epoch,
            val_metrics,
            test_metrics,
            loso_meta=loso_meta,
            window_sizes=split_sizes,
        )
        self._plot_history(history)

        result: dict[str, object] = {
            "best_epoch": best_epoch,
            "val": val_metrics,
            "test": test_metrics,
            "split_sizes": split_sizes,
        }
        if "vote" in val_metrics:
            result["val_vote"] = val_metrics["vote"]
        if "vote" in test_metrics:
            result["test_vote"] = test_metrics["vote"]
        return result

    def train_loso(
        self,
        dataset_base: Path,
        train_subjects: list[int],
        test_subject: int,
        val_ratio: float = 0.10,
        val_on_held_out: bool = False,
        resume_from: Path | None = None,
    ) -> dict[str, object]:
        split, train_sids, test_sid = self.build_loso_splits(
            dataset_base,
            train_subjects,
            test_subject,
            val_ratio=val_ratio,
            val_on_held_out=val_on_held_out,
        )
        print(f"LOSO train subjects: {train_sids}")
        print(f"LOSO test subject: sbj_{test_sid}")
        if val_on_held_out:
            print(f"LOSO val: sbj_{test_sid} (held-out subject, paper/IMU style)")
        else:
            print(f"LOSO val: {val_ratio:.0%} holdout from train subjects")
        return self.train(split, loso_meta=(train_sids, test_sid), resume_from=resume_from)

    def _checkpoint_payload(
        self,
        model: V2EModel,
        *,
        best_state: dict[str, torch.Tensor] | None,
        best_epoch: int,
        best_val_acc: float,
        last_state: dict[str, torch.Tensor] | None,
        last_epoch: int,
        history: dict[str, list[float]],
        patience_left: int,
    ) -> dict[str, object]:
        state_dict = best_state if best_state is not None else model.state_dict()
        return {
            "state_dict": state_dict,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "last_epoch": last_epoch,
            "best_state": best_state,
            "last_state": last_state,
            "history": history,
            "patience_left": patience_left,
            "config": {
                "data": {
                    **asdict(self.cfg.data),
                    "dataset_root": str(self.cfg.data.dataset_root),
                },
                "model": asdict(self.cfg.model),
                "train": asdict(self.cfg.train),
                "out_dir": str(self.cfg.out_dir),
            },
        }

    def _save_training_checkpoint(
        self,
        model: V2EModel,
        *,
        best_state: dict[str, torch.Tensor] | None,
        best_epoch: int,
        best_val_acc: float,
        last_state: dict[str, torch.Tensor] | None,
        last_epoch: int,
        history: dict[str, list[float]],
        patience_left: int,
    ) -> None:
        payload = self._checkpoint_payload(
            model,
            best_state=best_state,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            last_state=last_state,
            last_epoch=last_epoch,
            history=history,
            patience_left=patience_left,
        )
        torch.save(payload, self.out_dir / "checkpoint_last.pt")

    def _write_metrics_block(self, handle, metrics: dict[str, float], prefix: str = "") -> None:
        for key, value in metrics.items():
            if key == "vote":
                continue
            if isinstance(value, dict):
                handle.write(f"\n{prefix}--- {key.upper()} ---\n")
                for sub_key, sub_value in value.items():
                    handle.write(f"{prefix}{sub_key}: {sub_value:.4f}\n")
            else:
                handle.write(f"{prefix}{key}: {value:.4f}\n")

    def _save_summary(
        self,
        split: SplitIndex,
        best_epoch: int,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        *,
        loso_meta: tuple[list[int], int] | None = None,
        window_sizes: dict[str, int] | None = None,
    ) -> None:
        summary_path = self.out_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            if loso_meta is not None:
                train_sids, test_sid = loso_meta
                handle.write(
                    f"Experiment: experimentv2e {self.cfg.model.model_name} LOSO\n"
                )
                handle.write(f"Dataset base: {self.cfg.data.dataset_root}\n")
                handle.write(f"Train subjects: {train_sids}\n")
                handle.write(f"Test subject: sbj_{test_sid}\n")
            else:
                handle.write(
                    f"Experiment: experimentv2e {self.cfg.model.model_name} pilot\n"
                )
                handle.write(f"Dataset: {self.cfg.data.dataset_root}\n")
                handle.write(
                    f"Split 70/10/20 -> train={len(split.train)} val={len(split.val)} "
                    f"test={len(split.test)}\n"
                )
            handle.write(
                f"Samples -> train={window_sizes['train']} val={window_sizes['val']} "
                f"test={window_sizes['test']} windows\n"
                if window_sizes
                else f"Samples -> train={len(split.train)} val={len(split.val)} "
                f"test={len(split.test)} H5 files\n"
            )
            handle.write(
                f"Input HxWxT={self.cfg.data.input_height}x{self.cfg.data.input_width}x"
                f"{self.cfg.data.time_bins}  bin={self.cfg.data.sampling_time_ms}ms  "
                f"window={self.cfg.data.sample_length_ms}ms  "
                f"stride={self.cfg.data.window_stride_ms}ms\n"
            )
            handle.write(
                f"Model name={self.cfg.model.model_name} channels={self.cfg.model.channels} "
                f"dropout={self.cfg.model.dropout}\n"
            )
            handle.write(
                f"Augment={self.cfg.data.augment_train} normalize={self.cfg.data.normalize_clip} "
                f"weight_decay={self.cfg.train.weight_decay} "
                f"class_weights={self.cfg.train.class_weights} "
                f"majority_vote_k={self.cfg.train.majority_vote_k}\n"
            )
            handle.write(
                f"Train batch={self.cfg.train.batch_size} lr={self.cfg.train.learning_rate} "
                f"epochs={self.cfg.train.num_epochs}\n"
            )
            handle.write(f"Best epoch (val acc): {best_epoch + 1}\n\n")
            handle.write("--- VAL ---\n")
            self._write_metrics_block(handle, val_metrics)
            handle.write("\n--- TEST ---\n")
            self._write_metrics_block(handle, test_metrics)

        meta: dict[str, object] = {
            "best_epoch": best_epoch,
            "val": val_metrics,
            "test": test_metrics,
            "split_sizes": {
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
            "generalization": {
                "window_stride_ms": self.cfg.data.window_stride_ms,
                "augment_train": self.cfg.data.augment_train,
                "normalize_clip": self.cfg.data.normalize_clip,
                "weight_decay": self.cfg.train.weight_decay,
                "class_weights": self.cfg.train.class_weights,
                "majority_vote_k": self.cfg.train.majority_vote_k,
            },
        }
        if loso_meta is not None:
            train_sids, test_sid = loso_meta
            meta["train_subjects"] = train_sids
            meta["test_subject"] = test_sid
        (self.out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _plot_history(self, history: dict[str, list[float]]) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history["val_acc"], label="val acc")
        axes[1].set_title("Validation accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / "training_curves.png", dpi=150)
        plt.close(fig)
