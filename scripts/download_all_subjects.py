#!/usr/bin/env python3
import time
import urllib.request
import sys
from pathlib import Path

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
HEADERS = {"User-Agent": "Mozilla/5.0"}
RAW_DIR = Path("data/raw/physionet")

print("Downloading all 109 PhysioNet subjects (Runs R04, R08, R12)...")
downloaded = 0
skipped = 0
failed = 0

for sub in range(1, 110):
    s_str = f"S{sub:03d}"
    s_dir = RAW_DIR / s_str
    s_dir.mkdir(parents=True, exist_ok=True)
    for run in [4, 8, 12]:
        fname = f"{s_str}R{run:02d}.edf"
        dest = s_dir / fname
        if dest.exists() and dest.stat().st_size > 2000000:
            skipped += 1
            continue

        url = f"{BASE_URL}/{s_str}/{fname}"
        success = False
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
                    f.write(resp.read())
                if dest.stat().st_size > 2000000:
                    success = True
                    downloaded += 1
                    break
            except Exception as e:
                time.sleep(1)

        if not success:
            print(f"[FAIL] {fname}")
            failed += 1

subs = sorted([int(p.name[1:]) for p in RAW_DIR.glob("S*") if p.name[1:].isdigit()])
edfs = [f for f in RAW_DIR.glob("**/*.edf") if f.stat().st_size > 2000000]
print(f"REPORT: Downloaded={downloaded}, Skipped={skipped}, Failed={failed}")
print(f"Total valid EDFs on disk: {len(edfs)} (Expected 327 across 109 subjects)")
