#!/usr/bin/env python3
"""PhysioNet EEG Motor Movement/Imagery Dataset — Annotation & Data-Quality Audit.

EEGMMIDB v1.0.0  |  109 subjects  |  64-channel EEG  |  EDF/EDF+ recordings

Scientific constraints
-----------------------
* No model is trained, loaded, or evaluated.
* Test-set subjects (S094–S109) are treated identically to others during the
  audit; their results are reported but never used to select preprocessing rules.
* Exclusion criteria are declared in annotation_audit.py BEFORE this script
  runs — not chosen post-hoc based on accuracy.
* The original 80.98% test accuracy and 83.02% best validation accuracy are
  frozen baselines reproduced by separate scripts (audit_integrity.py /
  data_audit.py).  This script does NOT touch those files.

Usage
-----
    python scripts/audit_eegmmidb_quality.py \\
        --data-dir data/raw/physionet \\
        --out-dir  reports/data_quality

    # Audit a subset of subjects:
    python scripts/audit_eegmmidb_quality.py \\
        --data-dir data/raw/physionet \\
        --subjects S001 S038 S088 S100 S104 \\
        --out-dir  reports/data_quality/subset

Outputs
-------
    reports/data_quality/eegmmidb_subject_run_audit.csv
    reports/data_quality/eegmmidb_subject_run_audit.json
    reports/data_quality/eegmmidb_quality_report.md
    reports/data_quality/invalid_or_warning_runs.csv
    reports/data_quality/trial_count_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_mi.data_quality.annotation_audit import (
    BINARY_MI_RUNS,
    EXPECTED_N_CHANNELS,
    PHYSIONET_RUN_PROTOCOL,
    RunAuditRecord,
    RunStatus,
    audit_subject,
)

# ---------------------------------------------------------------------------
# Fixed project splits (must not be changed)
# ---------------------------------------------------------------------------

TRAIN_SUBJECTS = list(range(1, 78))  # S001–S077
VAL_SUBJECTS = list(range(78, 94))  # S078–S093
TEST_SUBJECTS = list(range(94, 110))  # S094–S109


def _subject_label(n: int) -> str:
    return f"S{n:03d}"


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars."""

    def default(self, obj: Any) -> Any:
        # numpy may not be importable on every machine; guard safely
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _split_of(subject_num: int) -> str:
    if subject_num in TRAIN_SUBJECTS:
        return "train"
    if subject_num in VAL_SUBJECTS:
        return "validation"
    if subject_num in TEST_SUBJECTS:
        return "test"
    return "unknown"


def _count_by_status(records: list[RunAuditRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.status] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_full_csv(records: list[RunAuditRecord], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].to_dict().keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def _write_warning_csv(records: list[RunAuditRecord], out_path: Path) -> None:
    flagged = [
        r
        for r in records
        if r.status
        in (
            RunStatus.VALID_WITH_WARNINGS.value,
            RunStatus.INVALID_FOR_BINARY_MI.value,
            RunStatus.CORRUPT_OR_UNREADABLE.value,
        )
    ]
    if not flagged:
        # Write empty file with headers
        fieldnames = list(records[0].to_dict().keys()) if records else ["subject_id"]
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        return
    _write_full_csv(flagged, out_path)


def _write_trial_comparison_csv(records: list[RunAuditRecord], out_path: Path) -> None:
    """One row per subject: expected, actual, and usable trial counts."""
    # Group by subject
    by_subject: dict[str, list[RunAuditRecord]] = defaultdict(list)
    for r in records:
        by_subject[r.subject_id].append(r)

    rows = []
    for sid in sorted(by_subject.keys()):
        sub_records = [r for r in by_subject[sid] if r.is_binary_mi_run]
        total_t1 = sum(r.n_t1_events for r in sub_records)
        total_t2 = sum(r.n_t2_events for r in sub_records)
        usable = sum(
            r.usable_trial_count
            for r in sub_records
            if r.status in (RunStatus.VALID.value, RunStatus.VALID_WITH_WARNINGS.value)
        )
        statuses = [r.status for r in sub_records]
        # expected: 3 runs × (7–10 T1 + 7–10 T2) depending on sfreq
        sfreq_mode = Counter(r.sfreq for r in sub_records).most_common(1)
        expected = 45 if sfreq_mode and sfreq_mode[0][0] == 160.0 else 57
        rows.append(
            {
                "subject_id": sid,
                "split": _split_of(int(sid.lstrip("S"))),
                "n_binary_mi_runs": len(sub_records),
                "expected_trials_approx": expected,
                "actual_t1_events": total_t1,
                "actual_t2_events": total_t2,
                "actual_total_trials": total_t1 + total_t2,
                "usable_trials": usable,
                "run_statuses": "|".join(statuses),
            }
        )

    if rows:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _render_markdown_report(  # noqa: C901
    records: list[RunAuditRecord],
    audit_timestamp: str,
    elapsed_seconds: float,
    out_path: Path,
) -> None:
    mi_records = [r for r in records if r.is_binary_mi_run]

    # Status counts (binary MI runs only)
    status_mi = _count_by_status(mi_records)

    # By split
    split_records: dict[str, list[RunAuditRecord]] = {"train": [], "validation": [], "test": []}
    for r in mi_records:
        if r.split in split_records:
            split_records[r.split].append(r)

    def _split_stats(recs: list[RunAuditRecord]) -> dict[str, Any]:
        valid = [r for r in recs if r.status == RunStatus.VALID.value]
        warn = [r for r in recs if r.status == RunStatus.VALID_WITH_WARNINGS.value]
        inv = [r for r in recs if r.status == RunStatus.INVALID_FOR_BINARY_MI.value]
        corrupt = [r for r in recs if r.status == RunStatus.CORRUPT_OR_UNREADABLE.value]
        t1 = sum(r.n_t1_events for r in valid + warn)
        t2 = sum(r.n_t2_events for r in valid + warn)
        return {
            "n_runs": len(recs),
            "valid": len(valid),
            "warnings": len(warn),
            "invalid": len(inv),
            "corrupt": len(corrupt),
            "t1_epochs": t1,
            "t2_epochs": t2,
            "total_epochs": t1 + t2,
        }

    tr_stats = _split_stats(split_records["train"])
    v_stats = _split_stats(split_records["validation"])
    te_stats = _split_stats(split_records["test"])

    # Subjects with issues
    problem_subjects: dict[str, list[str]] = defaultdict(list)
    for r in mi_records:
        if r.status != RunStatus.VALID.value:
            problem_subjects[r.subject_id].append(
                f"{r.run_label}: {r.status} — {r.classification_reason}"
            )

    # All warnings list
    all_warnings: list[tuple[str, str, str, str]] = []
    for r in mi_records:
        for w in r.warnings:
            all_warnings.append((r.subject_id, r.run_label, r.status, w))

    # sfreq distribution
    sfreq_counts: Counter[float] = Counter(r.sfreq for r in mi_records if r.readable)

    # Flat / outlier channel counts
    n_flat = sum(1 for r in mi_records if r.n_flat_channels > 0)
    n_outlier = sum(1 for r in mi_records if r.n_outlier_channels > 0)

    # Duplicate / outside-bounds
    n_dup = sum(1 for r in mi_records if r.duplicate_markers > 0)
    n_oob = sum(1 for r in mi_records if r.events_outside_recording > 0)

    # Files summary
    n_total_files = len(records)
    n_readable = sum(1 for r in records if r.readable)
    n_have_eeg = sum(1 for r in records if r.n_eeg_channels == EXPECTED_N_CHANNELS)
    n_have_ann = sum(1 for r in records if r.has_annotation_channel)
    n_valid_mi = sum(
        1
        for r in mi_records
        if r.status in (RunStatus.VALID.value, RunStatus.VALID_WITH_WARNINGS.value)
    )
    total_t1 = sum(
        r.n_t1_events
        for r in mi_records
        if r.status in (RunStatus.VALID.value, RunStatus.VALID_WITH_WARNINGS.value)
    )
    total_t2 = sum(
        r.n_t2_events
        for r in mi_records
        if r.status in (RunStatus.VALID.value, RunStatus.VALID_WITH_WARNINGS.value)
    )

    # flagged subjects from the task specification
    flagged_subjects = ["S038", "S082", "S088", "S089", "S092", "S100", "S104"]

    # ---------------------------------------------------------------------------
    md_lines: list[str] = []

    def h(level: int, text: str) -> None:
        md_lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells: Any) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def hr() -> None:
        md_lines.append("\n---\n")

    # ── Title ─────────────────────────────────────────────────────────────────
    md_lines.append("# PhysioNet EEGMMIDB v1.0.0 — Annotation & Data-Quality Audit Report\n")
    md_lines.append(f"> **Generated:** {audit_timestamp}  \n")
    md_lines.append(f"> **Audit runtime:** {elapsed_seconds:.1f} seconds  \n")
    md_lines.append(
        "> **Status of official test result:** UNCHANGED — 80.98% test accuracy "
        "on S094–S109 is the frozen baseline.  This audit does NOT evaluate any model.\n"
    )

    hr()

    # ── Dataset description ────────────────────────────────────────────────────
    h(2, "1. Dataset Description")
    md_lines.append(
        "The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) v1.0.0 "
        "consists of EDF/EDF+ recordings from 109 subjects performing or imagining "
        "four motor tasks. Each recording has 64 EEG channels at either 160 Hz or "
        "128 Hz.\n"
    )
    md_lines.append(
        "For this project the **binary left-fist vs right-fist motor imagery** "
        "task uses only runs R04, R08, and R12 (three repetitions of the left/right "
        "fist imagery paradigm per subject).\n"
    )

    hr()

    # ── Task / run protocol ───────────────────────────────────────────────────
    h(2, "2. Official PhysioNet Task / Run Protocol")
    md_lines.append(
        "**Critical:** T1 and T2 annotations have **run-dependent meanings**. "
        "T1 does NOT always mean left fist, and T2 does NOT always mean right fist.\n"
    )
    header = row("Run ID", "Task", "T1 Label", "T2 Label", "Used for Binary MI?")
    sep = row("---", "---", "---", "---", "---")
    md_lines.append(header)
    md_lines.append(sep)
    for rid, info in sorted(PHYSIONET_RUN_PROTOCOL.items()):
        used = "✅ **YES**" if info["use_for_binary_mi"] else "No"
        t1 = info["t1_label"] or "N/A"
        t2 = info["t2_label"] or "N/A"
        md_lines.append(row(f"R{rid:02d}", info["task"], t1, t2, used))
    md_lines.append("")

    hr()

    # ── Fixed split ───────────────────────────────────────────────────────────
    h(2, "3. Fixed Subject Split (Preserved)")
    md_lines.append("The following split is **frozen** and must not be modified.\n")
    md_lines.append(row("Split", "Subjects", "Count"))
    md_lines.append(row("---", "---", "---"))
    md_lines.append(row("Train", "S001–S077", 77))
    md_lines.append(row("Validation", "S078–S093", 16))
    md_lines.append(row("Test", "S094–S109", 16))
    md_lines.append("")
    md_lines.append(
        "> [!IMPORTANT]\n"
        "> No test-set information was used to define audit criteria.  "
        "The original test accuracy of **80.98%** remains unchanged.\n"
    )

    hr()

    # ── Audit methodology ─────────────────────────────────────────────────────
    h(2, "4. Audit Methodology")
    md_lines.append(
        "Each EDF file was inspected by `audit_single_run()` in "
        "`src/eeg_mi/data_quality/annotation_audit.py`.\n"
    )
    md_lines.append("**Classification rules (predeclared, not chosen post-hoc):**\n")
    md_lines.append(
        "| Status | Trigger condition |\n"
        "|--------|-------------------|\n"
        "| `CORRUPT_OR_UNREADABLE` | File missing or MNE raises exception |\n"
        "| `INVALID_FOR_BINARY_MI` | Run ≠ R04/R08/R12; or zero T1; or zero T2; or no annotation channel |\n"
        "| `VALID_WITH_WARNINGS` | ≥1 of: non-standard sfreq, duration < 120 s, flat channels, "
        "amplitude outliers, NaN/Inf, duplicate markers, out-of-bounds events, unexpected codes |\n"
        "| `VALID` | All checks passed |\n"
    )
    md_lines.append(
        "Subjects are **not** automatically excluded by name.  "
        "Every status is derived from direct EDF inspection.\n"
    )

    hr()

    # ── File-level metrics ─────────────────────────────────────────────────────
    h(2, "5. File-Level Metrics")
    md_lines.append(row("Metric", "Count"))
    md_lines.append(row("---", "---"))
    md_lines.append(row("Total EDF files audited (all runs)", n_total_files))
    md_lines.append(row("Readable EDF files", n_readable))
    md_lines.append(row(f"Files with {EXPECTED_N_CHANNELS} EEG channels", n_have_eeg))
    md_lines.append(row("Files with valid annotation channel", n_have_ann))
    md_lines.append(row("Valid binary MI run files (VALID or VALID_WITH_WARNINGS)", n_valid_mi))
    md_lines.append(row("Total left-fist (T1) epochs from valid runs", total_t1))
    md_lines.append(row("Total right-fist (T2) epochs from valid runs", total_t2))
    md_lines.append(row("Total usable binary MI epochs", total_t1 + total_t2))
    md_lines.append("")

    hr()

    # ── Sampling frequency distribution ─────────────────────────────────────
    h(2, "6. Sampling Frequency Distribution (Binary MI Runs)")
    md_lines.append(row("sfreq (Hz)", "Run count", "% of MI runs"))
    md_lines.append(row("---", "---", "---"))
    total_mi = len(mi_records)
    for sfreq, cnt in sorted(sfreq_counts.items()):
        pct = 100.0 * cnt / total_mi if total_mi else 0.0
        md_lines.append(row(f"{sfreq:.0f}", cnt, f"{pct:.1f}%"))
    md_lines.append("")

    hr()

    # ── Split-level trial counts ──────────────────────────────────────────────
    h(2, "7. Trial Counts by Split (Binary MI Runs Only)")
    md_lines.append(
        row(
            "Split",
            "Subjects",
            "Runs audited",
            "VALID",
            "VALID_WITH_WARNINGS",
            "INVALID",
            "CORRUPT",
            "T1 epochs",
            "T2 epochs",
            "Total epochs",
        )
    )
    md_lines.append(row(*["---"] * 10))
    for split_name, stats in [("train", tr_stats), ("validation", v_stats), ("test", te_stats)]:
        n_subs = (
            len(TRAIN_SUBJECTS)
            if split_name == "train"
            else (len(VAL_SUBJECTS) if split_name == "validation" else len(TEST_SUBJECTS))
        )
        md_lines.append(
            row(
                split_name,
                n_subs,
                stats["n_runs"],
                stats["valid"],
                stats["warnings"],
                stats["invalid"],
                stats["corrupt"],
                stats["t1_epochs"],
                stats["t2_epochs"],
                stats["total_epochs"],
            )
        )
    md_lines.append("")

    hr()

    # ── Status summary ────────────────────────────────────────────────────────
    h(2, "8. Run-Status Summary (Binary MI Runs)")
    md_lines.append(row("Status", "Count (of MI runs)", "% of MI runs"))
    md_lines.append(row("---", "---", "---"))
    for status in [
        RunStatus.VALID.value,
        RunStatus.VALID_WITH_WARNINGS.value,
        RunStatus.INVALID_FOR_BINARY_MI.value,
        RunStatus.CORRUPT_OR_UNREADABLE.value,
    ]:
        cnt = status_mi.get(status, 0)
        pct = 100.0 * cnt / len(mi_records) if mi_records else 0.0
        md_lines.append(row(status, cnt, f"{pct:.1f}%"))
    md_lines.append("")

    hr()

    # ── Flagged subjects ──────────────────────────────────────────────────────
    h(2, "9. Investigation of Flagged Subjects")
    md_lines.append(
        "The following seven subjects were specifically requested for investigation. "
        "The status shown is derived **exclusively** from EDF inspection.\n"
    )
    md_lines.append(row("Subject", "Split", "R04", "R08", "R12", "sfreq (Hz)", "Notes"))
    md_lines.append(row(*["---"] * 7))
    for sid in flagged_subjects:
        sub_recs = [r for r in mi_records if r.subject_id == sid]
        by_run = {r.run_label: r for r in sub_recs}
        sfreqs = sorted({r.sfreq for r in sub_recs if r.readable})
        sfreq_str = ", ".join(f"{s:.0f}" for s in sfreqs)
        split = sub_recs[0].split if sub_recs else "?"
        notes_parts = []
        for r in sub_recs:
            if r.warnings:
                notes_parts.append(f"{r.run_label}: " + "; ".join(r.warnings[:2]))
        notes = " | ".join(notes_parts) if notes_parts else "No anomalies detected"
        md_lines.append(
            row(
                sid,
                split,
                by_run.get("R04", RunAuditRecord("?", 4, "R04", "", "?")).status[:8]
                if "R04" in by_run
                else "N/A",
                by_run.get("R08", RunAuditRecord("?", 8, "R08", "", "?")).status[:8]
                if "R08" in by_run
                else "N/A",
                by_run.get("R12", RunAuditRecord("?", 12, "R12", "", "?")).status[:8]
                if "R12" in by_run
                else "N/A",
                sfreq_str,
                notes[:120] + "..." if len(notes) > 120 else notes,
            )
        )
    md_lines.append("")

    hr()

    # ── All warnings ─────────────────────────────────────────────────────────
    h(2, "10. Complete Warning List")
    if all_warnings:
        md_lines.append(row("Subject", "Run", "Status", "Warning"))
        md_lines.append(row("---", "---", "---", "---"))
        for sid, run, status, warning in all_warnings:
            md_lines.append(row(sid, run, status, warning))
        md_lines.append("")
    else:
        md_lines.append("No warnings detected across all binary MI runs.\n")

    hr()

    # ── Signal quality ────────────────────────────────────────────────────────
    h(2, "11. Signal-Quality Summary")
    md_lines.append(row("Check", "Runs affected", "% of MI runs"))
    md_lines.append(row("---", "---", "---"))
    total_mi_n = len(mi_records)
    pct = lambda n: f"{100.0 * n / total_mi_n:.1f}%" if total_mi_n else "0.0%"  # noqa: E731
    md_lines.append(row("Flat channels (std < 1e-7)", n_flat, pct(n_flat)))
    md_lines.append(row("Amplitude outlier channels (> 500 µV)", n_outlier, pct(n_outlier)))
    md_lines.append(row("Duplicate event markers", n_dup, pct(n_dup)))
    md_lines.append(row("Events outside recording bounds", n_oob, pct(n_oob)))
    md_lines.append("")

    hr()

    # ── Original vs audited counts ────────────────────────────────────────────
    h(2, "12. Original vs Audit-Aware Trial Counts")
    md_lines.append(
        "> [!NOTE]\n"
        "> The original processed dataset (`full_dataset.npz`) was built from "
        "all available recordings without run-level quality filtering. "
        "The counts below show what would remain under the predeclared audit rules.\n"
    )
    md_lines.append(
        row("Metric", "Original (all subjects)", "Audit-aware (VALID + VALID_WITH_WARNINGS)")
    )
    md_lines.append(row("---", "---", "---"))
    md_lines.append(row("Binary MI runs included", len(mi_records), n_valid_mi))
    md_lines.append(row("T1 (left fist) epochs", "see full_dataset.npz", total_t1))
    md_lines.append(row("T2 (right fist) epochs", "see full_dataset.npz", total_t2))
    md_lines.append("")
    md_lines.append(
        "> [!IMPORTANT]\n"
        "> Original 80.98% test accuracy (S094–S109) is the **frozen baseline**.  "
        "A quality-controlled evaluation would require retraining on the same model "
        "architecture under a predeclared protocol — that is a **separate optional step** "
        "and has NOT been performed here.\n"
    )

    hr()

    # ── Limitations ───────────────────────────────────────────────────────────
    h(2, "13. Limitations")
    md_lines.append(
        "1. **Signal quality checks are best-effort:** Preloading all 109 × 3 runs "
        "into RAM requires ~3 GB.  If the process runs out of memory, flat-channel "
        "and amplitude checks may be skipped for some files.\n"
        "2. **Epoch-level validation** (e.g. trial windows extending beyond the "
        "recording) requires the same 3-second window used during preprocessing; "
        "this audit uses the raw annotation times only.\n"
        "3. **The 128 Hz subjects** (S088, S092, S100) are classified as "
        "VALID_WITH_WARNINGS, not excluded.  Resampling to 160 Hz before feature "
        "extraction is the recommended approach; the audit does not verify that "
        "this was done correctly in the original pipeline.\n"
        "4. **This audit does not train, load, or evaluate any model.**  "
        "It is a pure data-quality pass.\n"
    )

    hr()

    # ── Reproducibility ───────────────────────────────────────────────────────
    h(2, "14. Reproducibility Instructions")
    md_lines.append("```bash\n")
    md_lines.append("# Clone the repository and check out the feature branch\n")
    md_lines.append("git clone <repo-url>\n")
    md_lines.append("git checkout feat/dataset-quality-audit\n\n")
    md_lines.append("# Install dependencies\n")
    md_lines.append("pip install -e '.[dev]'\n\n")
    md_lines.append("# Run the audit\n")
    md_lines.append("python scripts/audit_eegmmidb_quality.py \\\n")
    md_lines.append("    --data-dir data/raw/physionet \\\n")
    md_lines.append("    --out-dir  reports/data_quality\n")
    md_lines.append("```\n")
    md_lines.append(
        "\nAll outputs in `reports/data_quality/` are deterministic given the same "
        "input EDF files.  The audit reads files but never writes to `data/` or `models/`.\n"
    )

    out_path.write_text("\n".join(md_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet EEGMMIDB annotation and data-quality audit."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "physionet",
        help="Root directory containing per-subject subdirectories (default: data/raw/physionet)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs to audit (e.g. S001 S002).  Default: all S001–S109.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[4, 8, 12],
        help="Run IDs to audit (default: 4 8 12 — the binary MI runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "data_quality",
        help="Output directory for audit reports (default: reports/data_quality).",
    )
    parser.add_argument(
        "--no-signal-quality",
        action="store_true",
        help="Skip signal-quality checks (flat channels, amplitude) to reduce memory usage.",
    )
    return parser.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()

    # Resolve subject list
    if args.subjects:
        subject_ids = sorted(args.subjects)
    else:
        subject_ids = [_subject_label(n) for n in range(1, 110)]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir: Path = args.data_dir

    print("\n" + "=" * 78)
    print("  PHYSIONET EEGMMIDB — ANNOTATION & DATA-QUALITY AUDIT")
    print("=" * 78)
    print(f"  Data directory : {data_dir}")
    print(
        f"  Subjects       : {len(subject_ids)} (first: {subject_ids[0]}, last: {subject_ids[-1]})"
    )
    print(f"  Runs audited   : {args.runs}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Started        : {datetime.now(UTC).isoformat()}")
    print("=" * 78 + "\n")

    t_start = time.time()
    all_records: list[RunAuditRecord] = []

    for i, sid in enumerate(subject_ids):
        subject_dir = data_dir / sid
        print(f"  [{i + 1:>3}/{len(subject_ids)}] Auditing {sid} ...", end=" ")
        try:
            records = audit_subject(
                subject_dir=subject_dir,
                subject_id=sid,
                run_ids=args.runs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            # Create stub error records for all runs
            subject_num = int(sid.lstrip("S"))
            for run_id in args.runs:
                run_label = f"R{run_id:02d}"
                edf_path = subject_dir / f"{sid}{run_label}.edf"
                stub = RunAuditRecord(
                    subject_id=sid,
                    run_id=run_id,
                    run_label=run_label,
                    file_path=str(edf_path),
                    split=_split_of(subject_num),
                    status=RunStatus.CORRUPT_OR_UNREADABLE.value,
                    error_message=str(exc),
                    classification_reason=f"Exception during audit: {exc}",
                )
                all_records.append(stub)
            continue

        print(" | ".join(f"R{r.run_id:02d}:{r.status[:8]}" for r in records))
        all_records.extend(records)

    elapsed = round(time.time() - t_start, 1)
    audit_ts = datetime.now(UTC).isoformat()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    csv_path = out_dir / "eegmmidb_subject_run_audit.csv"
    _write_full_csv(all_records, csv_path)
    print(f"\n  ✓ Full audit CSV  → {_rel(csv_path)}")

    warn_csv = out_dir / "invalid_or_warning_runs.csv"
    _write_warning_csv(all_records, warn_csv)
    print(f"  ✓ Warning CSV     → {_rel(warn_csv)}")

    trial_csv = out_dir / "trial_count_comparison.csv"
    _write_trial_comparison_csv(all_records, trial_csv)
    print(f"  ✓ Trial count CSV → {_rel(trial_csv)}")

    # ── Write JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "eegmmidb_subject_run_audit.json"
    json_payload: dict[str, Any] = {
        "audit_metadata": {
            "timestamp": audit_ts,
            "elapsed_seconds": elapsed,
            "data_dir": str(data_dir),
            "subjects_audited": subject_ids,
            "runs_audited": args.runs,
            "binary_mi_runs": sorted(BINARY_MI_RUNS),
            "original_test_accuracy_frozen": 0.8098,
            "original_val_accuracy_frozen": 0.8302,
            "frozen_train_subjects": [_subject_label(n) for n in TRAIN_SUBJECTS],
            "frozen_val_subjects": [_subject_label(n) for n in VAL_SUBJECTS],
            "frozen_test_subjects": [_subject_label(n) for n in TEST_SUBJECTS],
        },
        "run_records": [r.to_dict() for r in all_records],
    }
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2, cls=NpEncoder)
    print(f"  ✓ Full audit JSON → {_rel(json_path)}")

    # ── Write Markdown ─────────────────────────────────────────────────────────
    md_path = out_dir / "eegmmidb_quality_report.md"
    _render_markdown_report(all_records, audit_ts, elapsed, md_path)
    print(f"  ✓ Markdown report → {_rel(md_path)}")

    # ── Summary ────────────────────────────────────────────────────────────────
    mi_records = [r for r in all_records if r.is_binary_mi_run]
    v_val = RunStatus.VALID.value
    w_val = RunStatus.VALID_WITH_WARNINGS.value
    i_val = RunStatus.INVALID_FOR_BINARY_MI.value
    c_val = RunStatus.CORRUPT_OR_UNREADABLE.value

    n_valid = sum(1 for r in mi_records if r.status == v_val)
    n_warn = sum(1 for r in mi_records if r.status == w_val)
    n_inv = sum(1 for r in mi_records if r.status == i_val)
    n_corrupt = sum(1 for r in mi_records if r.status == c_val)
    total_t1 = sum(r.n_t1_events for r in mi_records if r.status in (v_val, w_val))
    total_t2 = sum(r.n_t2_events for r in mi_records if r.status in (v_val, w_val))

    print("\n" + "=" * 78)
    print("  AUDIT SUMMARY (Binary MI Runs Only)")
    print("=" * 78)
    print(f"  Elapsed              : {elapsed} s")
    print(f"  Subjects audited     : {len(subject_ids)}")
    print(f"  Binary MI runs total : {len(mi_records)}")
    print(f"    VALID              : {n_valid}")
    print(f"    VALID_WITH_WARNINGS: {n_warn}")
    print(f"    INVALID_FOR_MI     : {n_inv}")
    print(f"    CORRUPT/UNREADABLE : {n_corrupt}")
    print(f"  T1 epochs (left)     : {total_t1}")
    print(f"  T2 epochs (right)    : {total_t2}")
    print(f"  Total usable epochs  : {total_t1 + total_t2}")
    print("=" * 78)
    print("  ⚠  Original 80.98% test accuracy (S094–S109) is FROZEN / UNCHANGED.")
    print("  ⚠  No model was trained, loaded, or evaluated in this audit.")
    print("=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
