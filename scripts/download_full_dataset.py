#!/usr/bin/env python3
"""Full PhysioNet EEG Motor Imagery Dataset Downloader (Subjects S001-S109, Runs R04, R08, R12).

Downloads all 327 EDF recordings for motor imagery runs R04, R08, and R12 across subjects S001..S109.
Stores files under data/raw/physionet/ in a clean directory hierarchy.
"""

import sys
import urllib.request
from pathlib import Path

from eeg_mi.utils.logging import get_logger

logger = get_logger("FullDatasetDownloader")

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
TARGET_SUBJECTS = list(range(1, 110))  # S001 to S109
TARGET_RUNS = [4, 8, 12]  # Motor Imagery Left vs Right Fist


def download_full_dataset(raw_dir: Path = Path("data/raw/physionet")) -> int:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      PhysioNet EEG Full Dataset Downloader (S001 - S109)")
    print("=" * 75)
    print(f"Target Directory : {raw_dir.resolve()}")
    print(f"Total Subjects   : {len(TARGET_SUBJECTS)} (S001 .. S109)")
    print(f"Target Runs      : {TARGET_RUNS} (Motor Imagery Left vs Right Fist)")
    print("=" * 75 + "\n")

    download_count = 0
    skip_count = 0
    error_count = 0
    total_bytes = 0

    headers = {"User-Agent": "Mozilla/5.0"}

    for sub_idx in TARGET_SUBJECTS:
        sub_str = f"S{sub_idx:03d}"
        sub_dir = raw_dir / sub_str
        sub_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in TARGET_RUNS:
            fname = f"{sub_str}R{run_idx:02d}.edf"
            file_path = sub_dir / fname
            url = f"{BASE_URL}/{sub_str}/{fname}"

            if file_path.exists() and file_path.stat().st_size > 0:
                skip_count += 1
                total_bytes += file_path.stat().st_size
                continue

            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as resp, open(file_path, "wb") as f:
                    content = resp.read()
                    f.write(content)

                size_bytes = file_path.stat().st_size
                total_bytes += size_bytes
                download_count += 1
                print(f"[SUCCESS] Saved {fname} ({size_bytes / 1024:.1f} KB)")
            except Exception as e:
                print(f"[ERROR] Failed to download {url}: {e}")
                error_count += 1
                if file_path.exists():
                    file_path.unlink()

    print("\n" + "=" * 75)
    print("               DOWNLOAD VERIFICATION REPORT")
    print("=" * 75)
    print(f"Downloaded Files: {download_count}")
    print(f"Skipped Files   : {skip_count}")
    print(f"Failed Files    : {error_count}")
    print(f"Total Size      : {total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 75 + "\n")

    if error_count > 0:
        logger.error(f"{error_count} files failed to download.")
        return 1

    return 0


def main() -> int:
    return download_full_dataset()


if __name__ == "__main__":
    sys.exit(main())
