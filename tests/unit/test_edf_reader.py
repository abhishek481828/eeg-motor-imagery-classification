"""Unit tests for EDF file reading and dataset inspection validation (Phase 3)."""

from pathlib import Path

import pytest

from eeg_mi.data.edf_reader import EDFReader
from scripts.inspect_dataset import inspect_physionet_dataset


def test_edf_reader_non_existent_file() -> None:
    """Test EDFReader raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        EDFReader(Path("non_existent.edf"))


def test_edf_reader_invalid_extension(tmp_path: Path) -> None:
    """Test EDFReader raises ValueError for non-EDF extension."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("dummy content")
    with pytest.raises(ValueError, match="expected .edf"):
        EDFReader(txt_file)


def test_inspect_dataset_missing_dir(tmp_path: Path) -> None:
    """Test dataset inspection gracefully handles empty/missing directory."""
    non_existent = tmp_path / "missing_dir"
    res = inspect_physionet_dataset(non_existent)
    assert "error" in res
