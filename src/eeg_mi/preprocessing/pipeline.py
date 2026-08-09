"""Full EEG Preprocessing Pipeline with run-specific event remapping and zero-leakage normalization."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg_mi.data.annotations import parse_run_event_mapping
from eeg_mi.data.edf_reader import EDFReader
from eeg_mi.data.validation import validate_eeg_window
from eeg_mi.preprocessing.filters import apply_bandpass_filter, apply_notch_filter
from eeg_mi.preprocessing.segmentation import segment_epochs
from eeg_mi.utils.logging import get_logger

logger = get_logger("PreprocessingPipeline")


def extract_run_id_from_filename(filename: str) -> int | None:
    """Extract numeric run ID from filename (e.g. S001R04.edf -> 4)."""
    match = re.search(r"R(\d{2})", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic MD5 hash of preprocessing configuration."""
    raw_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:12]


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for EEG EDF files."""

    def __init__(
        self,
        l_freq: float = 7.0,
        h_freq: float = 30.0,
        notch_freq: float = 60.0,
        target_sfreq: float | None = None,
        window_duration: float = 3.0,
        window_overlap: float = 1.5,
        allowed_runs: list[int] | None = None,
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.target_sfreq = target_sfreq
        self.window_duration = window_duration
        self.window_overlap = window_overlap
        self.allowed_runs = allowed_runs or [4, 8, 12]

        self.config_dict = {
            "l_freq": l_freq,
            "h_freq": h_freq,
            "notch_freq": notch_freq,
            "target_sfreq": target_sfreq,
            "window_duration": window_duration,
            "window_overlap": window_overlap,
            "allowed_runs": self.allowed_runs,
        }
        self.config_hash = compute_config_hash(self.config_dict)

    def process_recording(
        self, edf_path: Path
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Process single EDF file with run-specific event remapping (T1=0, T2=1)."""
        edf_path = Path(edf_path)
        if not edf_path.exists():
            raise FileNotFoundError(f"PhysioNet EDF file not found: {edf_path}")

        run_id = extract_run_id_from_filename(edf_path.name)
        if run_id is not None and run_id not in self.allowed_runs:
            logger.debug(f"Skipping run {run_id} (not in allowed_runs {self.allowed_runs})")
            return np.empty((0, 64, 0)), np.empty((0,), dtype=int), []

        reader = EDFReader(edf_path)
        raw = reader.read_raw(preload=True)

        # 1. Select EEG channels only
        raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False)

        # 2. Resample if configured
        if self.target_sfreq and raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose="ERROR")

        # 3. Apply bandpass (7-30Hz) and notch filtering
        if self.l_freq > 0 or self.h_freq > 0:
            raw = apply_bandpass_filter(raw, self.l_freq, self.h_freq)
        if self.notch_freq > 0 and self.notch_freq < raw.info["sfreq"] / 2:
            raw = apply_notch_filter(raw, self.notch_freq)

        # 4. Explicit run-specific event remapping (T1: Left Fist -> 0, T2: Right Fist -> 1, ignoring T0)
        event_id_map = parse_run_event_mapping(run_id if run_id is not None else 4)

        events, event_id = mne.events_from_annotations(raw, event_id=event_id_map, verbose="ERROR")
        if len(events) == 0 or len(event_id) == 0:
            return np.empty((0, len(raw.ch_names), 0)), np.empty((0,), dtype=int), []

        # 5. Segment signal starting at event onset (tmin=0.0 to tmax=3.0s)
        data, labels = segment_epochs(
            raw,
            events,
            event_id,
            tmin=0.0,
            tmax=self.window_duration,
        )

        valid_windows = []
        valid_labels = []
        metadata_list = []

        sub_id = edf_path.parent.name
        rec_id = edf_path.stem

        for idx, (win, lbl) in enumerate(zip(data, labels, strict=False)):
            try:
                validate_eeg_window(win, expected_channels=len(raw.ch_names))
                valid_windows.append(win)
                valid_labels.append(int(lbl))

                t_start = float(events[idx][0]) / raw.info["sfreq"]
                t_end = t_start + self.window_duration

                meta = {
                    "subject_id": sub_id,
                    "recording_id": rec_id,
                    "run_id": run_id,
                    "label": int(lbl),
                    "label_name": "left_fist" if int(lbl) == 0 else "right_fist",
                    "sfreq": float(raw.info["sfreq"]),
                    "channel_names": list(raw.ch_names),
                    "segment_start_sec": t_start,
                    "segment_end_sec": t_end,
                    "config_hash": self.config_hash,
                }
                metadata_list.append(meta)
            except ValueError as ve:
                logger.warning(f"Discarded window {idx} in {rec_id}: {ve}")

        if not valid_windows:
            return np.empty((0, len(raw.ch_names), 0)), np.empty((0,), dtype=int), []

        return np.array(valid_windows), np.array(valid_labels), metadata_list
