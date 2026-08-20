# PhysioNet EEGMMIDB v1.0.0 — Annotation & Data-Quality Audit Report

> **Generated:** 2026-08-20T07:22:56.916739+00:00
> **Scope:** Binary Motor Imagery Runs (R04, R08, R12) across 109 PhysioNet subjects
> **Audit runtime:** 12.6 seconds

> **Status of official test result:** UNCHANGED — 80.98% test accuracy on S094–S109 is the frozen baseline.  This audit does NOT evaluate any model.


---


## 1. Dataset Description

The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) v1.0.0 consists of EDF/EDF+ recordings from 109 subjects performing or imagining four motor tasks. Each recording has 64 EEG channels at either 160 Hz or 128 Hz.

For this project the **binary left-fist vs right-fist motor imagery** task uses only runs R04, R08, and R12 (three repetitions of the left/right fist imagery paradigm per subject).


---


## 2. Official PhysioNet Task / Run Protocol

**Critical:** T1 and T2 annotations have **run-dependent meanings**. T1 does NOT always mean left fist, and T2 does NOT always mean right fist.

| Run ID | Task | T1 Label | T2 Label | Used for Binary MI? |
| --- | --- | --- | --- | --- |
| R01 | baseline_eyes_open | N/A | N/A | No |
| R02 | baseline_eyes_closed | N/A | N/A | No |
| R03 | execution_left_right_fist | left_fist_execution | right_fist_execution | No |
| R04 | imagery_left_right_fist | left_fist_imagery | right_fist_imagery | ✅ **YES** |
| R05 | execution_both_fists_feet | both_fists_execution | both_feet_execution | No |
| R06 | imagery_both_fists_feet | both_fists_imagery | both_feet_imagery | No |
| R07 | execution_left_right_fist | left_fist_execution | right_fist_execution | No |
| R08 | imagery_left_right_fist | left_fist_imagery | right_fist_imagery | ✅ **YES** |
| R09 | execution_both_fists_feet | both_fists_execution | both_feet_execution | No |
| R10 | imagery_both_fists_feet | both_fists_imagery | both_feet_imagery | No |
| R11 | execution_left_right_fist | left_fist_execution | right_fist_execution | No |
| R12 | imagery_left_right_fist | left_fist_imagery | right_fist_imagery | ✅ **YES** |
| R13 | execution_both_fists_feet | both_fists_execution | both_feet_execution | No |
| R14 | imagery_both_fists_feet | both_fists_imagery | both_feet_imagery | No |


---


## 3. Fixed Subject Split (Preserved)

The following split is **frozen** and must not be modified.

| Split | Subjects | Count |
| --- | --- | --- |
| Train | S001–S077 | 77 |
| Validation | S078–S093 | 16 |
| Test | S094–S109 | 16 |

> [!IMPORTANT]
> No test-set information was used to define audit criteria.  The original test accuracy of **80.98%** remains unchanged.


---


## 4. Audit Methodology

Each EDF file was inspected by `audit_single_run()` in `src/eeg_mi/data_quality/annotation_audit.py`.

**Classification rules (predeclared, not chosen post-hoc):**

| Status | Trigger condition |
|--------|-------------------|
| `CORRUPT_OR_UNREADABLE` | File missing or MNE raises exception |
| `INVALID_FOR_BINARY_MI` | Run ≠ R04/R08/R12; or zero T1; or zero T2; or no annotation channel |
| `VALID_WITH_WARNINGS` | ≥1 of: non-standard sfreq, duration < 120 s, flat channels, amplitude outliers, NaN/Inf, duplicate markers, out-of-bounds events, unexpected codes |
| `VALID` | All checks passed |

Subjects are **not** automatically excluded by name.  Every status is derived from direct EDF inspection.


---


## 5. File-Level Metrics

| Metric | Count |
| --- | --- |
| Total EDF files audited (all runs) | 327 |
| Readable EDF files | 327 |
| Files with 64 EEG channels | 327 |
| Files with valid annotation channel | 327 |
| Valid binary MI run files (VALID or VALID_WITH_WARNINGS) | 327 |
| Total left-fist (T1) epochs from valid runs | 2480 |
| Total right-fist (T2) epochs from valid runs | 2438 |
| Total usable binary MI epochs | 4918 |


---


## 6. Sampling Frequency Distribution (Binary MI Runs)

| sfreq (Hz) | Run count | % of MI runs |
| --- | --- | --- |
| 128 | 9 | 2.8% |
| 160 | 318 | 97.2% |


---


## 7. Trial Counts by Split (Binary MI Runs Only)

| Split | Subjects | Runs audited | VALID | VALID_WITH_WARNINGS | INVALID | CORRUPT | T1 epochs | T2 epochs | Total epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 77 | 231 | 121 | 110 | 0 | 0 | 1750 | 1715 | 3465 |
| validation | 16 | 48 | 18 | 30 | 0 | 0 | 372 | 372 | 744 |
| test | 16 | 48 | 14 | 34 | 0 | 0 | 358 | 351 | 709 |


---


## 8. Run-Status Summary (Binary MI Runs)

| Status | Count (of MI runs) | % of MI runs |
| --- | --- | --- |
| VALID | 153 | 46.8% |
| VALID_WITH_WARNINGS | 174 | 53.2% |
| INVALID_FOR_BINARY_MI | 0 | 0.0% |
| CORRUPT_OR_UNREADABLE | 0 | 0.0% |


---


## 9. Investigation of Flagged Subjects

The following seven subjects were specifically requested for investigation. The status shown is derived **exclusively** from EDF inspection.

| Subject | Split | R04 | R08 | R12 | sfreq (Hz) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| S038 | train | VALID_WI | VALID_WI | VALID_WI | 160 | R04: 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... | R08: 7 amplitude-outlier channel(s)... |
| S082 | validation | VALID | VALID | VALID | 160 | No anomalies detected |
| S088 | validation | VALID_WI | VALID_WI | VALID_WI | 128 | R04: Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz); 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5... |
| S089 | validation | VALID | VALID | VALID | 160 | No anomalies detected |
| S092 | validation | VALID_WI | VALID_WI | VALID_WI | 128 | R04: Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz); 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1... |
| S100 | test | VALID_WI | VALID_WI | VALID_WI | 128 | R04: Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz); 18 amplitude-outlier channel(s) (> 500.0 µV): Fc3... |
| S104 | test | VALID | VALID_WI | VALID | 160 | R08: Recording duration 106.0 s is below minimum threshold 120.0 s (truncated run) |


---


## 10. Complete Warning List

| Subject | Run | Status | Warning |
| --- | --- | --- | --- |
| S001 | R04 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fp2., Af7., Af3., Af8. |
| S001 | R08 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S001 | R12 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. |
| S003 | R04 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af4. ... |
| S003 | R08 | VALID_WITH_WARNINGS | 14 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S003 | R12 | VALID_WITH_WARNINGS | 54 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S009 | R04 | VALID_WITH_WARNINGS | 23 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc6., C5.., Fp1., Fpz. ... |
| S009 | R08 | VALID_WITH_WARNINGS | 19 amplitude-outlier channel(s) (> 500.0 µV): Fc6., C6.., Fp1., Fpz., Fp2. ... |
| S009 | R12 | VALID_WITH_WARNINGS | 15 amplitude-outlier channel(s) (> 500.0 µV): Fc4., Fc6., Fp1., Fpz., Fp2. ... |
| S010 | R04 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S010 | R08 | VALID_WITH_WARNINGS | 42 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc1., Fcz., Fc4., C3.. ... |
| S010 | R12 | VALID_WITH_WARNINGS | 37 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., C5.. ... |
| S013 | R04 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S013 | R08 | VALID_WITH_WARNINGS | 44 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fcz., Fc2., Fc4. ... |
| S013 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fc3., Fp1., Fpz., Fp2., Af7. ... |
| S015 | R04 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S015 | R08 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S015 | R12 | VALID_WITH_WARNINGS | 9 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S017 | R04 | VALID_WITH_WARNINGS | 41 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S017 | R08 | VALID_WITH_WARNINGS | 32 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S017 | R12 | VALID_WITH_WARNINGS | 47 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S019 | R04 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S019 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. ... |
| S019 | R12 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S022 | R04 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S022 | R08 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Cp1., Cp2., Fp1., Fpz., Fp2. ... |
| S022 | R12 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S023 | R04 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S023 | R08 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S023 | R12 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S027 | R04 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fcz., Fp1., Fpz., Fp2., Af7. ... |
| S027 | R08 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S027 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Cp1., Fp1., Fpz., Fp2., Af7. ... |
| S028 | R04 | VALID_WITH_WARNINGS | 28 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S028 | R08 | VALID_WITH_WARNINGS | 61 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S028 | R12 | VALID_WITH_WARNINGS | 39 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S032 | R04 | VALID_WITH_WARNINGS | 14 amplitude-outlier channel(s) (> 500.0 µV): Cpz., Fp1., Fpz., Fp2., Af7. ... |
| S032 | R08 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S032 | R12 | VALID_WITH_WARNINGS | 30 amplitude-outlier channel(s) (> 500.0 µV): Fc2., Fc6., Cz.., Cp3., Cp1. ... |
| S036 | R04 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S036 | R08 | VALID_WITH_WARNINGS | 52 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S036 | R12 | VALID_WITH_WARNINGS | 20 amplitude-outlier channel(s) (> 500.0 µV): Cp3., Fp1., Fpz., Fp2., Af7. ... |
| S038 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S038 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S038 | R12 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S039 | R04 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fc2., Fp1., Fpz., Fp2., Af7. ... |
| S039 | R08 | VALID_WITH_WARNINGS | 41 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fc2., Fc4. ... |
| S039 | R12 | VALID_WITH_WARNINGS | 20 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc6., Cp3., Fp1., Fpz. ... |
| S040 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S040 | R08 | VALID_WITH_WARNINGS | 3 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fp2., Af7. |
| S040 | R12 | VALID_WITH_WARNINGS | 2 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Af7. |
| S043 | R04 | VALID_WITH_WARNINGS | 30 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S043 | R08 | VALID_WITH_WARNINGS | 20 amplitude-outlier channel(s) (> 500.0 µV): Fc6., Fp1., Fpz., Fp2., Af7. ... |
| S043 | R12 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S044 | R04 | VALID_WITH_WARNINGS | 20 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fp1., Fpz., Fp2. ... |
| S044 | R08 | VALID_WITH_WARNINGS | 21 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fp1., Fpz. ... |
| S044 | R12 | VALID_WITH_WARNINGS | 23 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S045 | R04 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S045 | R08 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S045 | R12 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S046 | R04 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S046 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): C6.., Fp1., Fpz., Fp2., Af7. ... |
| S046 | R12 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S048 | R04 | VALID_WITH_WARNINGS | 30 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fc2., Fc4. ... |
| S048 | R08 | VALID_WITH_WARNINGS | 41 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S048 | R12 | VALID_WITH_WARNINGS | 34 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S049 | R04 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af8. |
| S049 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fc6., Fp1., Fpz., Fp2., Af7. ... |
| S049 | R12 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S052 | R04 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fc3., Fp1., Fpz., Fp2., Af7. ... |
| S052 | R08 | VALID_WITH_WARNINGS | 42 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fcz., Fc2., Fc6. ... |
| S052 | R12 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af4. ... |
| S053 | R04 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S053 | R08 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S053 | R12 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S054 | R04 | VALID_WITH_WARNINGS | 15 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S054 | R08 | VALID_WITH_WARNINGS | 54 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S054 | R12 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S056 | R04 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S056 | R08 | VALID_WITH_WARNINGS | 9 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S056 | R12 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S057 | R08 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S057 | R12 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S059 | R04 | VALID_WITH_WARNINGS | 55 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S059 | R08 | VALID_WITH_WARNINGS | 43 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fc2., Fc4. ... |
| S059 | R12 | VALID_WITH_WARNINGS | 50 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fc2., Fc4. ... |
| S060 | R04 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S060 | R08 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S060 | R12 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S061 | R04 | VALID_WITH_WARNINGS | 19 amplitude-outlier channel(s) (> 500.0 µV): Fc2., Fc6., Fp1., Fpz., Fp2. ... |
| S061 | R08 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S061 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S065 | R04 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7. |
| S065 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S065 | R12 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. ... |
| S068 | R04 | VALID_WITH_WARNINGS | 14 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Cp1., Fp1., Fpz., Fp2. ... |
| S068 | R08 | VALID_WITH_WARNINGS | 47 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc1., Fcz., Fc2., Fc4. ... |
| S068 | R12 | VALID_WITH_WARNINGS | 56 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S070 | R04 | VALID_WITH_WARNINGS | 9 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S070 | R08 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S070 | R12 | VALID_WITH_WARNINGS | 48 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S071 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S071 | R08 | VALID_WITH_WARNINGS | 1 amplitude-outlier channel(s) (> 500.0 µV): P2.. |
| S071 | R12 | VALID_WITH_WARNINGS | 3 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fp2., Af8. |
| S075 | R04 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fp1., Fpz., Fp2., Af7. ... |
| S075 | R08 | VALID_WITH_WARNINGS | 27 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., C5.. ... |
| S075 | R12 | VALID_WITH_WARNINGS | 19 amplitude-outlier channel(s) (> 500.0 µV): Fcz., Fp1., Fpz., Fp2., Af7. ... |
| S077 | R04 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Cp6., Fp1., Fpz., Fp2. ... |
| S077 | R08 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc6., Fp1., Fpz., Fp2. ... |
| S077 | R12 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S079 | R04 | VALID_WITH_WARNINGS | 1 amplitude-outlier channel(s) (> 500.0 µV): P1.. |
| S079 | R08 | VALID_WITH_WARNINGS | 63 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S079 | R12 | VALID_WITH_WARNINGS | 63 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S080 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S080 | R08 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. ... |
| S080 | R12 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S083 | R04 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S083 | R08 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S083 | R12 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S084 | R04 | VALID_WITH_WARNINGS | 25 amplitude-outlier channel(s) (> 500.0 µV): Cp6., Fp1., Fpz., Fp2., Af7. ... |
| S084 | R08 | VALID_WITH_WARNINGS | 19 amplitude-outlier channel(s) (> 500.0 µV): Cp6., Fp1., Fpz., Fp2., Af7. ... |
| S084 | R12 | VALID_WITH_WARNINGS | 52 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fcz., Fc2., Fc4. ... |
| S086 | R04 | VALID_WITH_WARNINGS | 27 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fc6., C5.. ... |
| S086 | R08 | VALID_WITH_WARNINGS | 30 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S086 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fpz., Fp2., Af7., Af3., Afz. ... |
| S087 | R04 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af8. |
| S087 | R08 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fpz., Fp2., Af7., Af8. |
| S087 | R12 | VALID_WITH_WARNINGS | 5 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af8. |
| S088 | R04 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S088 | R04 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S088 | R08 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S088 | R08 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S088 | R12 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S088 | R12 | VALID_WITH_WARNINGS | 63 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S090 | R04 | VALID_WITH_WARNINGS | 14 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S090 | R08 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S090 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S091 | R04 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fp1., Fpz., Fp2., Af7. ... |
| S091 | R08 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fp1., Fpz., Fp2., Af7. ... |
| S091 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S092 | R04 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S092 | R04 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S092 | R08 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S092 | R08 | VALID_WITH_WARNINGS | 3 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fp2., Af7. |
| S092 | R12 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S092 | R12 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7. |
| S094 | R04 | VALID_WITH_WARNINGS | 22 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., C5.., Fp1., Fpz. ... |
| S094 | R08 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fc3., C5.., Fp1., Fpz., Fp2. ... |
| S094 | R12 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S095 | R04 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S095 | R08 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S095 | R12 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S096 | R04 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S096 | R08 | VALID_WITH_WARNINGS | 21 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc2., Fp1., Fpz., Fp2. ... |
| S096 | R12 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S097 | R04 | VALID_WITH_WARNINGS | 47 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fcz., Fc2., Fc4. ... |
| S097 | R08 | VALID_WITH_WARNINGS | 64 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S097 | R12 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af3., Af8. ... |
| S099 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Cp4., Fp1., Fp2., Af7., Af8. ... |
| S099 | R08 | VALID_WITH_WARNINGS | 10 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S099 | R12 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S100 | R04 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S100 | R04 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fc3., Fp1., Fpz., Fp2., Af7. ... |
| S100 | R08 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S100 | R08 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S100 | R12 | VALID_WITH_WARNINGS | Non-standard sampling frequency: 128.0 Hz (majority is 160.0 Hz) |
| S100 | R12 | VALID_WITH_WARNINGS | 11 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S101 | R04 | VALID_WITH_WARNINGS | 12 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S101 | R08 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S101 | R12 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S103 | R04 | VALID_WITH_WARNINGS | 60 amplitude-outlier channel(s) (> 500.0 µV): Fc5., Fc3., Fc1., Fcz., Fc2. ... |
| S103 | R08 | VALID_WITH_WARNINGS | 16 amplitude-outlier channel(s) (> 500.0 µV): Fc3., Fp1., Fpz., Fp2., Af7. ... |
| S103 | R12 | VALID_WITH_WARNINGS | 6 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S104 | R08 | VALID_WITH_WARNINGS | Recording duration 106.0 s is below minimum threshold 120.0 s (truncated run) |
| S106 | R04 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S106 | R08 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S106 | R12 | VALID_WITH_WARNINGS | 7 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S107 | R04 | VALID_WITH_WARNINGS | 2 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Af8. |
| S107 | R08 | VALID_WITH_WARNINGS | 8 amplitude-outlier channel(s) (> 500.0 µV): Af7., Af8., F7.., F4.., F6.. ... |
| S107 | R12 | VALID_WITH_WARNINGS | 4 amplitude-outlier channel(s) (> 500.0 µV): Fc6., Fp1., Af7., Af8. |
| S108 | R04 | VALID_WITH_WARNINGS | 13 amplitude-outlier channel(s) (> 500.0 µV): Fc6., Fp1., Fpz., Fp2., Af7. ... |
| S108 | R08 | VALID_WITH_WARNINGS | 17 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |
| S108 | R12 | VALID_WITH_WARNINGS | 18 amplitude-outlier channel(s) (> 500.0 µV): Fp1., Fpz., Fp2., Af7., Af3. ... |


---


## 11. Signal-Quality Summary

| Check | Runs affected | % of MI runs |
| --- | --- | --- |
| Flat channels (std < 1e-7) | 0 | 0.0% |
| Amplitude outlier channels (> 500 µV) | 173 | 52.9% |
| Duplicate event markers | 0 | 0.0% |
| Events outside recording bounds | 0 | 0.0% |


---


## 12. Original vs Audit-Aware Trial Counts

> [!NOTE]
> The original processed dataset (`full_dataset.npz`) was built from all available recordings without run-level quality filtering. The counts below show what would remain under the predeclared audit rules.

| Metric | Original (all subjects) | Audit-aware (VALID + VALID_WITH_WARNINGS) |
| --- | --- | --- |
| Binary MI runs included | 327 | 327 |
| T1 (left fist) epochs | see full_dataset.npz | 2480 |
| T2 (right fist) epochs | see full_dataset.npz | 2438 |

> [!IMPORTANT]
> Original 80.98% test accuracy (S094–S109) is the **frozen baseline**.  A quality-controlled evaluation would require retraining on the same model architecture under a predeclared protocol — that is a **separate optional step** and has NOT been performed here.


---


## 13. Limitations

1. **Signal quality checks are best-effort:** Preloading all 109 × 3 runs into RAM requires ~3 GB.  If the process runs out of memory, flat-channel and amplitude checks may be skipped for some files.
2. **Epoch-level validation** (e.g. trial windows extending beyond the recording) requires the same 3-second window used during preprocessing; this audit uses the raw annotation times only.
3. **The 128 Hz subjects** (S088, S092, S100) are classified as VALID_WITH_WARNINGS, not excluded.  Resampling to 160 Hz before feature extraction is the recommended approach; the audit does not verify that this was done correctly in the original pipeline.
4. **This audit does not train, load, or evaluate any model.**  It is a pure data-quality pass.


---


## 14. Reproducibility Instructions

```bash

# Clone the repository and check out the feature branch

git clone <repo-url>

git checkout feat/dataset-quality-audit


# Install dependencies

pip install -e '.[dev]'


# Run the audit

python scripts/audit_eegmmidb_quality.py \

    --data-dir data/raw/physionet \

    --out-dir  reports/data_quality

```


All outputs in `reports/data_quality/` are deterministic given the same input EDF files.  The audit reads files but never writes to `data/` or `models/`.
