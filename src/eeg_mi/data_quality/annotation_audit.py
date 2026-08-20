"""Annotation and signal-quality audit for the PhysioNet EEG Motor Imagery dataset.

This module audits every EDF/EDF+ recording in the EEGMMIDB v1.0.0 dataset
(PhysioNet, 109 subjects, 64 channels) without loading or training any model.

Scientific constraints enforced here
-------------------------------------
* No model is trained or tuned.
* Test-set results (S094–S109 / 80.98%) are never referenced.
* Exclusion criteria are declared before running, not chosen post-hoc.
* Every warning and exclusion is traceable to a run-level observation.
* Subjects are not excluded based solely on an external list.

PhysioNet task / run protocol
------------------------------
The EEGMMIDB recording structure for each subject (simplified):

  Run 01     Baseline, eyes open
  Run 02     Baseline, eyes closed
  Run 03     Motor EXECUTION  – open/close left(T1) or right(T2) fist
  Run 04     Motor IMAGERY    – imagine opening/closing left(T1) or right(T2) fist
  Run 05     Motor EXECUTION  – open/close both(T1) fists or both(T2) feet
  Run 06     Motor IMAGERY    – imagine opening/closing both(T1) fists or both(T2) feet
  Run 07     Motor EXECUTION  – open/close left(T1) or right(T2) fist  (repeat)
  Run 08     Motor IMAGERY    – imagine opening/closing left(T1) or right(T2) fist (repeat)
  Run 09     Motor EXECUTION  – open/close both(T1) fists or both(T2) feet (repeat)
  Run 10     Motor IMAGERY    – imagine opening/closing both(T1) fists or both(T2) feet (repeat)
  Run 11     Motor EXECUTION  – open/close left(T1) or right(T2) fist  (repeat 2)
  Run 12     Motor IMAGERY    – imagine opening/closing left(T1) or right(T2) fist  (repeat 2)
  Run 13     Motor EXECUTION  – open/close both(T1) fists or both(T2) feet (repeat 2)
  Run 14     Motor IMAGERY    – imagine opening/closing both(T1) fists or both(T2) feet (repeat 2)

  T0 = rest; T1/T2 meaning is RUN-DEPENDENT (see table above).

For the binary left-fist vs right-fist imagery task, only runs 4, 8, 12 are used.
  T1 → left fist (class 0)
  T2 → right fist (class 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Canonical PhysioNet task mapping.  Key = run_id (int).
#: "task" describes what the subject does.
#: "t1_label" / "t2_label" describe the motor action coded by T1 and T2.
#: "use_for_binary_mi" marks the three left/right fist *imagery* runs.
PHYSIONET_RUN_PROTOCOL: dict[int, dict[str, Any]] = {
    1: {
        "task": "baseline_eyes_open",
        "t1_label": None,
        "t2_label": None,
        "use_for_binary_mi": False,
        "description": "Baseline, eyes open (no motor task)",
    },
    2: {
        "task": "baseline_eyes_closed",
        "t1_label": None,
        "t2_label": None,
        "use_for_binary_mi": False,
        "description": "Baseline, eyes closed (no motor task)",
    },
    3: {
        "task": "execution_left_right_fist",
        "t1_label": "left_fist_execution",
        "t2_label": "right_fist_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close left(T1) or right(T2) fist",
    },
    4: {
        "task": "imagery_left_right_fist",
        "t1_label": "left_fist_imagery",
        "t2_label": "right_fist_imagery",
        "use_for_binary_mi": True,
        "description": "Motor imagery – imagine left(T1) or right(T2) fist (primary binary MI run)",
    },
    5: {
        "task": "execution_both_fists_feet",
        "t1_label": "both_fists_execution",
        "t2_label": "both_feet_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close both(T1) fists or both(T2) feet",
    },
    6: {
        "task": "imagery_both_fists_feet",
        "t1_label": "both_fists_imagery",
        "t2_label": "both_feet_imagery",
        "use_for_binary_mi": False,
        "description": "Motor imagery – imagine both(T1) fists or both(T2) feet",
    },
    7: {
        "task": "execution_left_right_fist",
        "t1_label": "left_fist_execution",
        "t2_label": "right_fist_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close left(T1) or right(T2) fist (repeat)",
    },
    8: {
        "task": "imagery_left_right_fist",
        "t1_label": "left_fist_imagery",
        "t2_label": "right_fist_imagery",
        "use_for_binary_mi": True,
        "description": "Motor imagery – imagine left(T1) or right(T2) fist (repeat)",
    },
    9: {
        "task": "execution_both_fists_feet",
        "t1_label": "both_fists_execution",
        "t2_label": "both_feet_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close both(T1) fists or both(T2) feet (repeat)",
    },
    10: {
        "task": "imagery_both_fists_feet",
        "t1_label": "both_fists_imagery",
        "t2_label": "both_feet_imagery",
        "use_for_binary_mi": False,
        "description": "Motor imagery – imagine both(T1) fists or both(T2) feet (repeat)",
    },
    11: {
        "task": "execution_left_right_fist",
        "t1_label": "left_fist_execution",
        "t2_label": "right_fist_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close left(T1) or right(T2) fist (repeat 2)",
    },
    12: {
        "task": "imagery_left_right_fist",
        "t1_label": "left_fist_imagery",
        "t2_label": "right_fist_imagery",
        "use_for_binary_mi": True,
        "description": "Motor imagery – imagine left(T1) or right(T2) fist (repeat 2)",
    },
    13: {
        "task": "execution_both_fists_feet",
        "t1_label": "both_fists_execution",
        "t2_label": "both_feet_execution",
        "use_for_binary_mi": False,
        "description": "Motor execution – open/close both(T1) fists or both(T2) feet (repeat 2)",
    },
    14: {
        "task": "imagery_both_fists_feet",
        "t1_label": "both_fists_imagery",
        "t2_label": "both_feet_imagery",
        "use_for_binary_mi": False,
        "description": "Motor imagery – imagine both(T1) fists or both(T2) feet (repeat 2)",
    },
}

#: Runs used for the binary left-vs-right fist imagery task.
BINARY_MI_RUNS: frozenset[int] = frozenset(
    rid for rid, info in PHYSIONET_RUN_PROTOCOL.items() if info["use_for_binary_mi"]
)

#: Majority sampling frequency in the dataset (Hz).
MAJORITY_SFREQ: float = 160.0

#: Minimum acceptable run duration in seconds (below this → truncated).
MIN_DURATION_SECONDS: float = 120.0

#: Standard deviation threshold below which a channel is flagged as flat (µV).
FLAT_CHANNEL_STD_THRESHOLD: float = 1e-7

#: Amplitude above which a channel is flagged as a potential outlier (µV).
OUTLIER_AMPLITUDE_THRESHOLD: float = 500.0

#: Expected number of EEG channels in EEGMMIDB.
EXPECTED_N_CHANNELS: int = 64


# ---------------------------------------------------------------------------
# RunStatus enum
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):  # noqa: UP042
    """Classification of a single EDF run's suitability for binary MI analysis.

    VALID
        The run passed all checks. T1 and T2 counts are present, duration is
        normal, sampling rate is the majority value (160 Hz), and no signal
        quality issues were detected.

    VALID_WITH_WARNINGS
        The run can be used but has at least one anomaly that the analyst
        should be aware of:
          - Non-standard sampling frequency (e.g. 128 Hz instead of 160 Hz).
          - Recording shorter than ``MIN_DURATION_SECONDS``.
          - Event count deviating from the within-subject mode.
          - Flat or near-flat channels.
          - Amplitude outlier channels.

    INVALID_FOR_BINARY_MI
        The run cannot contribute usable epochs to the binary MI task because:
          - Zero T1 *or* zero T2 events (no signal for one class).
          - No annotation channel in the EDF.
          - The run ID is not one of the binary MI runs (4, 8, 12).

    CORRUPT_OR_UNREADABLE
        MNE raised an exception when trying to read the EDF header or
        annotations — the file is physically corrupt or has an unsupported
        format variant.
    """

    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INVALID_FOR_BINARY_MI = "INVALID_FOR_BINARY_MI"
    CORRUPT_OR_UNREADABLE = "CORRUPT_OR_UNREADABLE"


# ---------------------------------------------------------------------------
# RunAuditRecord dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunAuditRecord:
    """All per-run audit metrics for one (subject, run) pair.

    Fields are populated by :func:`audit_single_run`.  ``status`` and
    ``classification_reason`` are set by :func:`classify_run_status`.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    subject_id: str  # e.g. "S001"
    run_id: int  # e.g. 4
    run_label: str  # e.g. "R04"
    file_path: str  # absolute path to the .edf file
    split: str  # "train" | "validation" | "test" | "unknown"

    # ── File / header ────────────────────────────────────────────────────────
    file_exists: bool = False
    file_size_bytes: int = 0
    readable: bool = False

    # ── Recording metadata ───────────────────────────────────────────────────
    n_channels: int = 0
    n_eeg_channels: int = 0
    sfreq: float = 0.0
    duration_seconds: float = 0.0
    n_samples: int = 0

    # ── Annotation channel ───────────────────────────────────────────────────
    has_annotation_channel: bool = False
    annotation_descriptions: list[str] = field(default_factory=list)

    # ── Event counts ─────────────────────────────────────────────────────────
    n_t0_events: int = 0
    n_t1_events: int = 0
    n_t2_events: int = 0
    n_total_events: int = 0
    unexpected_event_codes: list[str] = field(default_factory=list)
    duplicate_markers: int = 0

    # ── Event timing ─────────────────────────────────────────────────────────
    t1_onsets: list[float] = field(default_factory=list)
    t2_onsets: list[float] = field(default_factory=list)
    t1_durations: list[float] = field(default_factory=list)
    t2_durations: list[float] = field(default_factory=list)
    events_outside_recording: int = 0

    # ── Signal quality ───────────────────────────────────────────────────────
    has_nan: bool = False
    has_inf: bool = False
    flat_channels: list[str] = field(default_factory=list)
    outlier_channels: list[str] = field(default_factory=list)
    n_flat_channels: int = 0
    n_outlier_channels: int = 0

    # ── Task / run validity ──────────────────────────────────────────────────
    is_binary_mi_run: bool = False  # True only for run IDs 4, 8, 12
    expected_t1_label: str | None = None  # from PHYSIONET_RUN_PROTOCOL
    expected_t2_label: str | None = None

    # ── Usable trial count ───────────────────────────────────────────────────
    usable_trial_count: int = 0  # n_t1 + n_t2 for binary MI runs, else 0

    # ── Classification (set by classify_run_status) ──────────────────────────
    status: str = RunStatus.VALID.value
    warnings: list[str] = field(default_factory=list)
    classification_reason: str = ""

    # ── Error (set on unreadable files) ─────────────────────────────────────
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise record to a flat dict suitable for CSV / JSON export."""
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "file_path": self.file_path,
            "split": self.split,
            "file_exists": self.file_exists,
            "file_size_bytes": self.file_size_bytes,
            "readable": self.readable,
            "n_channels": self.n_channels,
            "n_eeg_channels": self.n_eeg_channels,
            "sfreq": self.sfreq,
            "duration_seconds": self.duration_seconds,
            "n_samples": self.n_samples,
            "has_annotation_channel": self.has_annotation_channel,
            "n_t0_events": self.n_t0_events,
            "n_t1_events": self.n_t1_events,
            "n_t2_events": self.n_t2_events,
            "n_total_events": self.n_total_events,
            "duplicate_markers": self.duplicate_markers,
            "events_outside_recording": self.events_outside_recording,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "n_flat_channels": self.n_flat_channels,
            "n_outlier_channels": self.n_outlier_channels,
            "flat_channels": "|".join(self.flat_channels) if self.flat_channels else "",
            "outlier_channels": ("|".join(self.outlier_channels) if self.outlier_channels else ""),
            "is_binary_mi_run": self.is_binary_mi_run,
            "expected_t1_label": self.expected_t1_label or "",
            "expected_t2_label": self.expected_t2_label or "",
            "usable_trial_count": self.usable_trial_count,
            "status": self.status,
            "warnings": "|".join(self.warnings) if self.warnings else "",
            "classification_reason": self.classification_reason,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Subject → split mapping
# ---------------------------------------------------------------------------


def _get_split(subject_num: int) -> str:
    """Return the split label for a subject number (1-indexed)."""
    if 1 <= subject_num <= 77:
        return "train"
    if 78 <= subject_num <= 93:
        return "validation"
    if 94 <= subject_num <= 109:
        return "test"
    return "unknown"


# ---------------------------------------------------------------------------
# Core audit functions
# ---------------------------------------------------------------------------


def audit_single_run(edf_path: Path, run_id: int, subject_id: str) -> RunAuditRecord:  # noqa: C901
    """Read one EDF file and return a fully populated :class:`RunAuditRecord`.

    This function is **read-only**: it never writes files, trains models, or
    mutates any shared state.  It is safe to call from multiple threads.

    Parameters
    ----------
    edf_path:
        Absolute path to the ``.edf`` file.
    run_id:
        Integer run identifier (e.g. ``4`` for R04).
    subject_id:
        String subject identifier (e.g. ``"S001"``).

    Returns
    -------
    RunAuditRecord
        Fully populated record.  ``status`` and ``classification_reason``
        are set by calling :func:`classify_run_status` on the returned record.
    """
    try:
        import mne  # type: ignore[import-untyped]
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "MNE and NumPy are required for the EDF audit.  Install with: pip install mne numpy"
        ) from exc

    subject_num = int(subject_id.lstrip("S"))
    run_label = f"R{run_id:02d}"
    protocol = PHYSIONET_RUN_PROTOCOL.get(run_id, {})

    record = RunAuditRecord(
        subject_id=subject_id,
        run_id=run_id,
        run_label=run_label,
        file_path=str(edf_path),
        split=_get_split(subject_num),
        is_binary_mi_run=protocol.get("use_for_binary_mi", False),
        expected_t1_label=protocol.get("t1_label"),
        expected_t2_label=protocol.get("t2_label"),
    )

    # ── File existence ────────────────────────────────────────────────────────
    edf_path = Path(edf_path)
    record.file_exists = edf_path.exists()
    if not record.file_exists:
        record.status = RunStatus.CORRUPT_OR_UNREADABLE.value
        record.classification_reason = "EDF file does not exist on disk"
        return classify_run_status(record)

    record.file_size_bytes = edf_path.stat().st_size

    # ── Attempt to read EDF ───────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        record.readable = False
        record.status = RunStatus.CORRUPT_OR_UNREADABLE.value
        record.error_message = str(exc)
        record.classification_reason = f"MNE raised exception reading EDF: {exc}"
        return record

    record.readable = True

    # ── Header / metadata ─────────────────────────────────────────────────────
    record.n_channels = len(raw.ch_names)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    record.n_eeg_channels = int(len(eeg_picks))
    record.sfreq = float(raw.info["sfreq"])
    record.n_samples = int(raw.n_times)
    record.duration_seconds = float(raw.times[-1]) if len(raw.times) > 0 else 0.0

    # ── Annotation channel ────────────────────────────────────────────────────
    annotations = raw.annotations
    record.has_annotation_channel = annotations is not None and len(annotations) > 0
    if record.has_annotation_channel:
        record.annotation_descriptions = sorted(set(annotations.description))

    # ── Event counts ──────────────────────────────────────────────────────────
    if record.has_annotation_channel:
        descs = list(annotations.description)
        onsets = list(annotations.onset)
        durations = list(annotations.duration)

        record.n_t0_events = int(descs.count("T0"))
        record.n_t1_events = int(descs.count("T1"))
        record.n_t2_events = int(descs.count("T2"))
        record.n_total_events = len(descs)

        # Unexpected codes (anything outside T0/T1/T2)
        known_codes = {"T0", "T1", "T2"}
        record.unexpected_event_codes = sorted({d for d in descs if d not in known_codes})

        # Duplicate markers: count consecutive identical (onset, desc) pairs
        seen: set[tuple[float, str]] = set()
        dups = 0
        for onset, desc in zip(onsets, descs):
            key = (round(onset, 6), desc)
            if key in seen:
                dups += 1
            seen.add(key)
        record.duplicate_markers = dups

        # Per-event timing
        t1_idx = [i for i, d in enumerate(descs) if d == "T1"]
        t2_idx = [i for i, d in enumerate(descs) if d == "T2"]
        record.t1_onsets = [round(onsets[i], 4) for i in t1_idx]
        record.t2_onsets = [round(onsets[i], 4) for i in t2_idx]
        record.t1_durations = [round(durations[i], 4) for i in t1_idx]
        record.t2_durations = [round(durations[i], 4) for i in t2_idx]

        # Events outside recording boundaries
        rec_end = record.duration_seconds
        record.events_outside_recording = sum(
            1 for o, dur in zip(onsets, durations) if (o < 0) or (o + dur > rec_end + 0.5)
        )

    # ── Signal quality ────────────────────────────────────────────────────────
    try:
        raw.load_data(verbose="ERROR")
        data = raw.get_data()  # shape (n_channels, n_samples)

        record.has_nan = bool(np.isnan(data).any())
        record.has_inf = bool(np.isinf(data).any())

        ch_names = raw.ch_names
        for i, ch in enumerate(ch_names):
            ch_data = data[i]
            std_val = float(np.std(ch_data))
            max_amp = float(np.max(np.abs(ch_data))) * 1e6  # convert V → µV

            if std_val < FLAT_CHANNEL_STD_THRESHOLD:
                record.flat_channels.append(ch)
            if max_amp > OUTLIER_AMPLITUDE_THRESHOLD:
                record.outlier_channels.append(ch)

        record.n_flat_channels = len(record.flat_channels)
        record.n_outlier_channels = len(record.outlier_channels)

    except Exception:  # noqa: BLE001
        # Signal quality checks are best-effort; if preloading fails, skip
        pass

    # ── Usable trial count ────────────────────────────────────────────────────
    if record.is_binary_mi_run:
        record.usable_trial_count = record.n_t1_events + record.n_t2_events

    # ── Classification ────────────────────────────────────────────────────────
    return classify_run_status(record)


def classify_run_status(record: RunAuditRecord) -> RunAuditRecord:  # noqa: C901
    """Apply predeclared, rule-based classification to a :class:`RunAuditRecord`.

    Rules are applied in priority order.  The first matching rule sets the
    status.  All warnings are accumulated regardless of final status.

    **These rules are declared before running any audit** and are not chosen
    post-hoc based on downstream model accuracy.

    Classification hierarchy
    ~~~~~~~~~~~~~~~~~~~~~~~~
    1. CORRUPT_OR_UNREADABLE  – if file missing or unreadable.
    2. INVALID_FOR_BINARY_MI  – if run is not R04/R08/R12, or has zero T1/T2.
    3. VALID_WITH_WARNINGS    – if run has any of the documented anomalies.
    4. VALID                  – all checks passed.

    Parameters
    ----------
    record:
        A :class:`RunAuditRecord` populated by :func:`audit_single_run`.
        This function mutates ``record.status``, ``record.warnings``, and
        ``record.classification_reason`` in-place.

    Returns
    -------
    RunAuditRecord
        The same object, mutated.
    """
    warnings: list[str] = []

    # ── Priority 1: unreadable ─────────────────────────────────────────────
    if not record.file_exists:
        record.status = RunStatus.CORRUPT_OR_UNREADABLE.value
        record.classification_reason = "File not found on disk"
        record.warnings = warnings
        return record

    if not record.readable:
        record.status = RunStatus.CORRUPT_OR_UNREADABLE.value
        record.classification_reason = f"MNE could not parse EDF: {record.error_message[:200]}"
        record.warnings = warnings
        return record

    # ── Priority 2: invalid for binary MI ──────────────────────────────────
    if not record.is_binary_mi_run:
        record.status = RunStatus.INVALID_FOR_BINARY_MI.value
        record.classification_reason = (
            f"Run {record.run_label} is not a binary motor imagery run "
            "(only R04, R08, R12 are valid for left vs right fist imagery)"
        )
        record.warnings = warnings
        return record

    if not record.has_annotation_channel:
        record.status = RunStatus.INVALID_FOR_BINARY_MI.value
        record.classification_reason = "No annotation channel found in EDF"
        record.warnings = warnings
        return record

    if record.n_t1_events == 0:
        record.status = RunStatus.INVALID_FOR_BINARY_MI.value
        record.classification_reason = "Zero T1 events — no left-fist imagery trials available"
        record.warnings = warnings
        return record

    if record.n_t2_events == 0:
        record.status = RunStatus.INVALID_FOR_BINARY_MI.value
        record.classification_reason = "Zero T2 events — no right-fist imagery trials available"
        record.warnings = warnings
        return record

    # ── Accumulate warnings ────────────────────────────────────────────────
    if not math.isclose(record.sfreq, MAJORITY_SFREQ, rel_tol=0.01):
        warnings.append(
            f"Non-standard sampling frequency: {record.sfreq} Hz (majority is {MAJORITY_SFREQ} Hz)"
        )

    if record.duration_seconds < MIN_DURATION_SECONDS:
        warnings.append(
            f"Recording duration {record.duration_seconds:.1f} s is below "
            f"minimum threshold {MIN_DURATION_SECONDS} s (truncated run)"
        )

    if record.n_flat_channels > 0:
        warnings.append(
            f"{record.n_flat_channels} flat/near-flat channel(s) detected: "
            f"{', '.join(record.flat_channels[:5])}"
            + (" ..." if len(record.flat_channels) > 5 else "")
        )

    if record.n_outlier_channels > 0:
        warnings.append(
            f"{record.n_outlier_channels} amplitude-outlier channel(s) "
            f"(> {OUTLIER_AMPLITUDE_THRESHOLD} µV): "
            f"{', '.join(record.outlier_channels[:5])}"
            + (" ..." if len(record.outlier_channels) > 5 else "")
        )

    if record.has_nan:
        warnings.append("NaN values detected in EEG signal data")

    if record.has_inf:
        warnings.append("Infinite values detected in EEG signal data")

    if record.duplicate_markers > 0:
        warnings.append(f"{record.duplicate_markers} duplicate event marker(s) found")

    if record.events_outside_recording > 0:
        warnings.append(
            f"{record.events_outside_recording} event(s) extend beyond recording boundaries"
        )

    if record.unexpected_event_codes:
        warnings.append(f"Unexpected event codes (non T0/T1/T2): {record.unexpected_event_codes}")

    if record.n_eeg_channels != EXPECTED_N_CHANNELS:
        warnings.append(
            f"Unexpected EEG channel count: {record.n_eeg_channels} "
            f"(expected {EXPECTED_N_CHANNELS})"
        )

    # ── Priority 3: warnings ───────────────────────────────────────────────
    if record.is_binary_mi_run:
        record.usable_trial_count = record.n_t1_events + record.n_t2_events

    if warnings:
        record.status = RunStatus.VALID_WITH_WARNINGS.value
        record.classification_reason = (
            f"{len(warnings)} warning(s): "
            + "; ".join(warnings[:2])
            + (" ..." if len(warnings) > 2 else "")
        )
    else:
        record.status = RunStatus.VALID.value
        record.classification_reason = "All annotation and signal-quality checks passed"

    record.warnings = warnings
    return record


def audit_subject(
    subject_dir: Path,
    subject_id: str,
    run_ids: list[int] | None = None,
) -> list[RunAuditRecord]:
    """Audit all specified runs for one subject directory.

    Parameters
    ----------
    subject_dir:
        Path to the subject's directory (e.g. ``data/raw/physionet/S001/``).
    subject_id:
        String subject identifier (e.g. ``"S001"``).
    run_ids:
        List of run IDs to audit.  Defaults to ``[4, 8, 12]`` (binary MI runs).

    Returns
    -------
    list[RunAuditRecord]
        One record per run.
    """
    if run_ids is None:
        run_ids = [4, 8, 12]

    records: list[RunAuditRecord] = []
    for run_id in run_ids:
        run_label = f"R{run_id:02d}"
        edf_path = Path(subject_dir) / f"{subject_id}{run_label}.edf"
        record = audit_single_run(edf_path=edf_path, run_id=run_id, subject_id=subject_id)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Convenience wrapper class
# ---------------------------------------------------------------------------


class AnnotationAudit:
    """High-level wrapper around :func:`audit_single_run`.

    Intended for programmatic use in notebooks or pipeline scripts.

    Parameters
    ----------
    data_root:
        Root directory containing per-subject subdirectories
        (e.g. ``data/raw/physionet``).
    run_ids:
        Run IDs to audit.  Defaults to binary MI runs ``[4, 8, 12]``.
    """

    def __init__(
        self,
        data_root: Path,
        run_ids: list[int] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.run_ids = run_ids if run_ids is not None else [4, 8, 12]

    def audit_subject(self, subject_id: str) -> list[RunAuditRecord]:
        """Audit all configured runs for ``subject_id``."""
        subject_dir = self.data_root / subject_id
        return audit_subject(
            subject_dir=subject_dir,
            subject_id=subject_id,
            run_ids=self.run_ids,
        )

    def audit_all(self, subject_ids: list[str] | None = None) -> list[RunAuditRecord]:
        """Audit all subjects.  Returns flat list of records."""
        if subject_ids is None:
            subject_ids = [f"S{n:03d}" for n in range(1, 110)]

        all_records: list[RunAuditRecord] = []
        for sid in subject_ids:
            all_records.extend(self.audit_subject(sid))
        return all_records
