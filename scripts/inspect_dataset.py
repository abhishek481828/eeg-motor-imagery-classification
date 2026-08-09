#!/usr/bin/env python3
"""Dataset Inspection Script for PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0.

Recursively scans EDF files, verifies dataset integrity, checks channel names,
sampling rates, signal durations, and annotation event markers without loading
entire signals into RAM.
"""

import argparse
import sys
from pathlib import Path

from eeg_mi.data.edf_reader import EDFReader
from eeg_mi.utils.logging import get_logger

logger = get_logger("InspectDataset")


def extract_subject_id_from_path(file_path: Path) -> int | None:
    """Extract numeric subject ID from path (e.g. S001/S001R01.edf -> 1)."""
    parent_name = file_path.parent.name
    if parent_name.startswith("S") and parent_name[1:].isdigit():
        return int(parent_name[1:])
    stem = file_path.stem
    if stem.startswith("S") and len(stem) >= 4 and stem[1:4].isdigit():
        return int(stem[1:4])
    return None


def inspect_physionet_dataset(raw_dir: Path) -> dict[str, any]:
    """Inspect EDF files in raw_dir recursively with lazy loading."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.error(f"Directory does not exist: {raw_dir}")
        return {"error": f"Directory not found: {raw_dir}"}

    edf_files = sorted(list(raw_dir.glob("**/*.edf")) + list(raw_dir.glob("**/*.EDF")))
    if not edf_files:
        logger.warning(f"No .edf files found in '{raw_dir}'")
        return {
            "num_files": 0,
            "num_subjects": 0,
            "corrupted_files": [],
        }

    subjects: set[int] = set()
    corrupted_files: list[dict[str, str]] = []
    valid_recordings = 0
    all_events: set[str] = set()
    sample_metadata = None

    logger.info(f"Scanning {len(edf_files)} EDF files in {raw_dir}...")

    for fpath in edf_files:
        sub_id = extract_subject_id_from_path(fpath)
        if sub_id is not None:
            subjects.add(sub_id)

        try:
            reader = EDFReader(fpath)
            info = reader.get_info()
            valid_recordings += 1

            if info.get("annotations"):
                all_events.update(info["annotations"])

            if sample_metadata is None:
                sample_metadata = info
        except Exception as e:
            logger.error(f"Corrupted or unreadable EDF file: {fpath} ({e})")
            corrupted_files.append({"file": str(fpath), "error": str(e)})

    return {
        "raw_dir": str(raw_dir),
        "total_files_found": len(edf_files),
        "valid_recordings": valid_recordings,
        "num_subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "corrupted_files": corrupted_files,
        "corrupted_count": len(corrupted_files),
        "all_event_markers": sorted(all_events),
        "sample_metadata": sample_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect PhysioNet EEG Dataset EDF files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/physionet",
        help="Path to raw PhysioNet EDF files directory",
    )
    args = parser.parse_args()

    raw_path = Path(args.data_dir)
    results = inspect_physionet_dataset(raw_path)

    if "error" in results or results.get("num_files") == 0:
        print("\n" + "=" * 70)
        print("               PHYSIOET DATASET NOT FOUND")
        print("=" * 70)
        print(f"Directory checked: {raw_path.resolve()}")
        print("Please follow instructions in data/README.md to place PhysioNet data:")
        print("  data/raw/physionet/S001/S001R01.edf ... S109/")
        print("=" * 70)
        return 1

    print("\n" + "=" * 70)
    print("      PhysioNet EEG Motor Imagery Dataset Inspection Summary")
    print("=" * 70)
    print(f"Raw Directory      : {results['raw_dir']}")
    print(f"Total EDF Files    : {results['total_files_found']}")
    print(f"Valid Recordings   : {results['valid_recordings']}")
    print(f"Subject Count      : {results['num_subjects']} subjects")
    print(f"Corrupted Files    : {results['corrupted_count']}")
    print(f"All Event Markers  : {results['all_event_markers']}")

    if results["sample_metadata"]:
        info = results["sample_metadata"]
        print("\n--- Representative Recording Metadata ---")
        print(f"Sample File Name   : {info['file_name']}")
        print(f"Sampling Frequency : {info['sfreq']} Hz")
        print(f"Total Channels     : {info['num_channels']}")
        print(f"EEG Channels       : {info['eeg_channel_count']}")
        print(
            f"Duration           : {info['duration_sec']:.2f} seconds ({info['n_samples']} samples)"
        )
        print(f"Annotations/Events : {info['annotations']}")
    print("=" * 70)

    return 0 if results["corrupted_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
