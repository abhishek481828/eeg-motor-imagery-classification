"""Unit tests for annotation parsing."""

import mne
import numpy as np

from eeg_mi.data.annotations import extract_events_from_annotations


def test_extract_events() -> None:
    """Test extracting events and mapping dictionary from raw MNE annotations."""
    info = mne.create_info(ch_names=["Cz"], sfreq=160.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 160)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onset=[0.1, 0.5], duration=[0.2, 0.2], description=["T1", "T2"])
    )

    annotations, event_id = extract_events_from_annotations(raw)
    assert len(event_id) == 2
    assert "T1" in event_id
    assert "T2" in event_id
