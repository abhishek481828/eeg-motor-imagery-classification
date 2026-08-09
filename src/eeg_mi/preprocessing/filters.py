"""Band-pass and notch signal filtering utilities."""

import mne


def apply_bandpass_filter(raw: mne.io.Raw, l_freq: float = 7.0, h_freq: float = 30.0) -> mne.io.Raw:
    """Apply band-pass filter to raw EEG signal."""
    filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    return filtered


def apply_notch_filter(raw: mne.io.Raw, freqs: float = 60.0) -> mne.io.Raw:
    """Apply notch filter to remove line noise."""
    filtered = raw.copy().notch_filter(freqs=freqs, verbose="ERROR")
    return filtered
