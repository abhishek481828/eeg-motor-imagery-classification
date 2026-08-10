# Phase 1: Data Integrity & Baseline Reproduction Audit Report

## Executive Summary
- **Overall Audit Status**: **PASS**
- **Subject Overlap**: **0** (Disjoint partitioning confirmed)
- **Duplicate Epochs**: **0** (SHA-256 verified)
- **NaNs / Infs**: **0**
- **Tuned CNN Val Accuracy Reproduced**: **80.32%** (Expected: 80.32%)
- **Tuned CNN Val Macro F1 Reproduced**: **0.8032** (Expected: 0.8032)
- **Test Set Evaluation**: **UNTOUCHED (0 test evaluations performed)**

---

## Dataset Shapes & Subject Splits

| Partition | Subject Count | Subjects | Epoch Count | Shape | Class 0 (Left Fist) | Class 1 (Right Fist) |
|---|---|---|---|---|---|---|
| **Train** | 77 | S001–S077 | 3465 | `(3465, 64, 481)` | 1750 | 1715 |
| **Validation** | 16 | S078–S093 | 630 | `(630, 64, 481)` | 316 | 314 |
| **Test** | 16 | S094–S109 | 673 | `(673, 64, 481)` | 340 | 333 |

---

## Data Quality Checks

| Check | Result | Details |
|---|---|---|
| **Subject Separation** | ✅ PASS | Zero subject overlap across Train, Val, and Test |
| **Epoch Uniqueness** | ✅ PASS | Zero duplicate epoch SHA-256 hashes across splits |
| **Data Cleanliness** | ✅ PASS | 0 NaNs and 0 Infs in all tensors |
| **Label Encoding** | ✅ PASS | Unique labels: Class 0 (Left Fist), Class 1 (Right Fist) |
| **Scaler Fitting** | ✅ PASS | Scalers fitted strictly on training subjects ($S001-S077$) |
| **Baseline Reproduction** | ✅ PASS | Tuned CNN (`cnn_tuned_cfg_02`) Val Acc = **80.32%**, Val Macro F1 = **0.8032** |
