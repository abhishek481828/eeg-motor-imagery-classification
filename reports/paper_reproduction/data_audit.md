# Phase 1: Pipeline Verification & Data Audit (Paper Study)

## Executive Summary
- **Overall Audit Status**: **PASS**
- **Subject Overlap**: **0** (Disjoint partitioning confirmed)
- **Duplicate Epochs**: **0** (SHA-256 verified)
- **NaNs / Infs**: **0**
- **Val-Weighted Ensemble Accuracy Reproduced**: **83.02%** (Expected: 83.02%)
- **Val-Weighted Ensemble Macro F1 Reproduced**: **0.8302** (Expected: 0.8302)
- **Test Set Protection**: **CONFIRMED (0 test subjects loaded or evaluated)**

---

## Dataset Shapes & Subject Splits

| Partition | Subject Count | Subjects | Epoch Count | Shape | Class 0 (Left Fist) | Class 1 (Right Fist) |
|---|---|---|---|---|---|---|
| **Train** | 77 | S001–S077 | 3465 | `(3465, 64, 481)` | 1750 | 1715 |
| **Validation** | 16 | S078–S093 | 630 | `(630, 64, 481)` | 316 | 314 |
| **Test (Frozen)** | 16 | S094–S109 | -- | -- | -- | -- |

---

## Data Quality Checks

| Check | Result | Details |
|---|---|---|
| **Subject Separation** | ✅ PASS | Zero subject overlap across Train, Val, and Test |
| **Epoch Uniqueness** | ✅ PASS | Zero duplicate epoch SHA-256 hashes between Train and Val |
| **Data Cleanliness** | ✅ PASS | 0 NaNs and 0 Infs in all tensors |
| **Label Encoding** | ✅ PASS | Unique labels: Class 0 (Left Fist), Class 1 (Right Fist) |
| **Scaler Fitting** | ✅ PASS | Scalers fitted strictly on training subjects ($S001-S077$) |
| **Ensemble Reproduction** | ✅ PASS | Val Ensemble (w=0.45/0.55) Val Acc = **83.02%**, Val Macro F1 = **0.8302** |
