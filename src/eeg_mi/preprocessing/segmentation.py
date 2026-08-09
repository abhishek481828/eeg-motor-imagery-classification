"""Signal segmentation into labeled epoch windows aligned to event onset."""

import mne
import numpy as np


def segment_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    tmin: float = 0.0,
    tmax: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment raw EEG signal into epoch windows starting exactly at event onset (tmin to tmax)."""
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    # Extract zero-indexed event codes mapped in event_id
    labels = epochs.events[:, -1]
    return data, labels
