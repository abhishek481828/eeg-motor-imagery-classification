"""Unit tests verifying strict real EDF file validation and run-specific label remapping."""

from pathlib import Path

import pytest

from eeg_mi.data.annotations import parse_run_event_mapping
from eeg_mi.preprocessing.pipeline import PreprocessingPipeline, extract_run_id_from_filename
from scripts.train import load_real_edf_dataset


def test_run_event_mapping_runs_4_8_12() -> None:
    """Test explicit zero-indexed label remapping for Runs 4, 8, 12 (Left vs Right Fist Imagery)."""
    for run in [4, 8, 12]:
        mapping = parse_run_event_mapping(run)
        assert mapping == {"T1": 0, "T2": 1}
        assert mapping["T1"] == 0  # Left Fist
        assert mapping["T2"] == 1  # Right Fist
        assert "T0" not in mapping  # Rest ignored for 2-class experiment


def test_extract_run_id_from_filename() -> None:
    """Test extracting numeric run ID from filename."""
    assert extract_run_id_from_filename("S001R04.edf") == 4
    assert extract_run_id_from_filename("S002R08.EDF") == 8
    assert extract_run_id_from_filename("S109R12.edf") == 12


def test_load_real_edf_dataset_raises_file_not_found(tmp_path: Path) -> None:
    """Test load_real_edf_dataset raises FileNotFoundError if raw EDF directory is missing or empty."""
    missing_dir = tmp_path / "non_existent_physionet"
    pipeline = PreprocessingPipeline(allowed_runs=[4, 8, 12])

    with pytest.raises(FileNotFoundError, match="CRITICAL ERROR"):
        load_real_edf_dataset(
            missing_dir, subject_ids=[1, 2], allowed_runs=[4, 8, 12], pipeline=pipeline
        )


def test_pipeline_raises_file_not_found_on_missing_file() -> None:
    """Test PreprocessingPipeline raises FileNotFoundError when EDF file does not exist."""
    pipeline = PreprocessingPipeline(allowed_runs=[4, 8, 12])
    missing_file = Path("data/raw/physionet/S001/S001R04_nonexistent.edf")

    with pytest.raises(FileNotFoundError, match="PhysioNet EDF file not found"):
        pipeline.process_recording(missing_file)
