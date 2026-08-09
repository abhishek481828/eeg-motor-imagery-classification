#!/usr/bin/env python3
"""Main Model Training Script for 2-Class EEG Motor Imagery (Runs 4, 8, 12).

Enforces CPU-safe defaults (batch_size=8, num_workers=0, pin_memory=False).
Enforces strict real EDF file validation. Raises FileNotFoundError if raw PhysioNet data is missing.
Prints dataset summary statistics before starting training.
"""

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from eeg_mi.data.dataset import EEGDataset
from eeg_mi.models.factory import create_model, get_model_summary
from eeg_mi.preprocessing.pipeline import PreprocessingPipeline
from eeg_mi.tracking.mlflow_utils import (
    init_mlflow,
    log_environment_metadata,
    log_experiment_params,
)
from eeg_mi.training.seed import set_seed
from eeg_mi.training.trainer import Trainer
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("TrainScript")


def load_subject_splits(manifest_path: Path) -> dict[str, Any]:
    """Load train/val/test subject split manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. Run `python scripts/create_subject_splits.py` first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def load_real_edf_dataset(
    raw_data_dir: Path,
    subject_ids: list[int],
    allowed_runs: list[int],
    pipeline: PreprocessingPipeline,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load real EDF files for specified subjects and runs.

    Raises FileNotFoundError if no EDF files are found (NO mock data fallback).
    """
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: PhysioNet raw data directory '{raw_data_dir.resolve()}' does not exist!\n"
            "The pipeline will NEVER use mock synthetic data for training.\n"
            "Please follow data/README.md instructions to download raw PhysioNet EDF files."
        )

    all_windows = []
    all_labels = []
    all_metadata = []

    sub_dirs = [raw_data_dir / f"S{sub:03d}" for sub in subject_ids]
    edf_files = []
    for sdir in sub_dirs:
        if sdir.exists():
            for r in allowed_runs:
                matches = list(sdir.glob(f"S{sdir.name[1:]}R{r:02d}.edf")) + list(
                    sdir.glob(f"S{sdir.name[1:]}R{r:02d}.EDF")
                )
                edf_files.extend(matches)

    if not edf_files:
        raise FileNotFoundError(
            f"CRITICAL ERROR: No EDF files for runs {allowed_runs} found under '{raw_data_dir.resolve()}'!\n"
            "Pipeline aborted. Real PhysioNet EDF recordings are required."
        )

    logger.info(
        f"Found {len(edf_files)} real EDF files across {len(subject_ids)} subjects. Preprocessing..."
    )

    for fpath in edf_files:
        wins, lbls, meta = pipeline.process_recording(fpath)
        if len(wins) > 0:
            all_windows.append(wins)
            all_labels.append(lbls)
            all_metadata.extend(meta)

    if not all_windows:
        raise ValueError(f"No valid event windows extracted from EDF files in {raw_data_dir}")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    return X, y, all_metadata


def train_pipeline(cfg: DictConfig) -> None:
    """Execute complete model training workflow with real EDF validation."""
    set_seed(cfg.get("seed", 42))

    # Automatic Device Selection: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(cfg.get("device", "auto"))
    print(f"\n[Hardware Initialization] Selected PyTorch Compute Device: {device}\n")

    # Load subject split manifest
    split_manifest_path = Path(cfg.get("split_manifest", "data/splits/subject_split.json"))
    splits = load_subject_splits(split_manifest_path)
    train_subjects = splits["splits"]["train"]
    val_subjects = splits["splits"]["validation"]

    # If dev subset configured, filter subject lists
    if "subject_subset" in cfg.data:
        dev_subs = list(cfg.data.subject_subset)
        train_subjects = [s for s in train_subjects if s in dev_subs]
        val_subjects = [s for s in val_subjects if s in dev_subs]
        logger.info(f"Using development subset ({len(dev_subs)} subjects)")

    # Preprocessing pipeline setup
    raw_dir = Path(cfg.data.get("raw_data_dir", "data/raw/physionet"))
    allowed_runs = list(cfg.data.get("relevant_runs", [4, 8, 12]))
    pipeline = PreprocessingPipeline(
        l_freq=cfg.data.get("l_freq", 7.0),
        h_freq=cfg.data.get("h_freq", 30.0),
        notch_freq=cfg.data.get("notch_freq", 60.0),
        window_duration=cfg.data.get("window_duration", 3.0),
        allowed_runs=allowed_runs,
    )

    # STRICT CHECK: Load real EDF files or raise FileNotFoundError
    logger.info("Loading REAL PhysioNet EDF recordings (No mock data permitted)...")
    X_train, y_train, train_meta = load_real_edf_dataset(
        raw_dir, train_subjects, allowed_runs, pipeline
    )
    X_val, y_val, val_meta = load_real_edf_dataset(raw_dir, val_subjects, allowed_runs, pipeline)

    # Print Dataset Summary before training
    num_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    sfreq = cfg.data.get("sampling_rate", 160)

    unique_cls, counts_cls = np.unique(y_train, return_counts=True)
    class_dist = dict(
        zip([int(c) for c in unique_cls], [int(cnt) for cnt in counts_cls], strict=False)
    )

    print("\n" + "=" * 70)
    print("      REAL PHYSIONET DATASET PRE-TRAINING SUMMARY")
    print("=" * 70)
    print(f"Selected Compute Device: {device}")
    print(f"Raw Directory          : {raw_dir.resolve()}")
    print(f"Allowed Runs           : {allowed_runs} (2-Class Left Fist vs Right Fist)")
    print(f"Training Subjects      : {len(train_subjects)} subjects")
    print(f"Validation Subjects    : {len(val_subjects)} subjects")
    print(f"Training Windows       : {len(X_train)} samples")
    print(f"Validation Windows     : {len(X_val)} samples")
    print(f"Samples per Class      : {class_dist}")
    print(f"Input Tensor Shape     : (batch, {num_channels}, {seq_len})")
    print(f"Sampling Frequency     : {sfreq} Hz")
    print("=" * 70 + "\n")

    # Initialize MLflow tracking
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
    exp_name = mlflow_cfg.get("experiment_name", "eeg_motor_imagery_2class")
    init_mlflow(experiment_name=exp_name, tracking_uri=tracking_uri)

    log_environment_metadata()
    log_experiment_params(OmegaConf.to_container(cfg, resolve=True))

    # Instantiate PyTorch CNN-LSTM model
    model = create_model(
        "cnn_lstm",
        num_channels=num_channels,
        num_classes=num_classes,
        sequence_length=seq_len,
    )
    logger.info(get_model_summary(model))

    # CPU-Safe DataLoader Settings
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        EEGDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "models/checkpoints"))
    checkpoint_path = checkpoint_dir / "cnn_lstm_2class_best.pt"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        patience=10,
        config_dict=OmegaConf.to_container(cfg, resolve=True),
    )

    epochs = int(cfg.get("epochs", cfg.experiments.training.get("epochs", 30)))
    logger.info(f"Starting CNN-LSTM model training for {epochs} epochs on device: {device}...")
    _ = trainer.fit(train_loader, val_loader, epochs=epochs)

    logger.info(f"Training complete. Best model checkpoint saved to: {checkpoint_path.resolve()}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
