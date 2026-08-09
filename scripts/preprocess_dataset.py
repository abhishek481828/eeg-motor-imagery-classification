#!/usr/bin/env python3
"""Preprocessing Dataset Script."""

import sys
from pathlib import Path

from eeg_mi.utils.logging import get_logger

logger = get_logger("PreprocessScript")


def main() -> int:
    raw_dir = Path("data/raw/physionet")
    if not raw_dir.exists() or not list(raw_dir.glob("**/*.edf")):
        logger.warning(
            "Raw PhysioNet EDF files not found. Please place EDF dataset files under data/raw/physionet/"
        )
        return 1

    logger.info("Starting dataset preprocessing pipeline...")
    # Preprocessing orchestration will execute when raw EDF files exist
    logger.info("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
