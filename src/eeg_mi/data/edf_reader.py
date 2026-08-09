"""EDF File Reader and Lazy Loader using MNE.

Supports lazy metadata extraction without loading full EEG signals into memory.
Handles corrupted file validation and reporting.
"""

from pathlib import Path
from typing import Any

import mne


class EDFReader:
    """Class to read and inspect EDF/EDF+ files using MNE with lazy loading."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"EDF file does not exist: {self.file_path}")
        if self.file_path.suffix.lower() != ".edf":
            raise ValueError(f"Invalid file format (expected .edf): {self.file_path}")

    def read_raw(self, preload: bool = False) -> mne.io.Raw:
        """Read EDF file as an MNE Raw object.

        Args:
            preload: If False, streams header metadata without reading signal data into RAM.
        """
        try:
            raw = mne.io.read_raw_edf(self.file_path, preload=preload, verbose="ERROR")
            return raw
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse EDF header/signal from '{self.file_path.name}': {e}"
            )

    def get_info(self) -> dict[str, Any]:
        """Extract metadata from EDF without reading full signal into memory."""
        raw = self.read_raw(preload=False)
        eeg_picks = mne.pick_types(raw.info, eeg=True)

        # Annotations inspection
        annotations = raw.annotations
        events = sorted(set(annotations.description)) if annotations else []

        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path),
            "sfreq": raw.info["sfreq"],
            "num_channels": len(raw.ch_names),
            "eeg_channel_count": len(eeg_picks),
            "duration_sec": float(raw.times[-1]) if len(raw.times) > 0 else 0.0,
            "channel_names": raw.ch_names,
            "annotations": events,
            "n_samples": raw.n_times,
        }
