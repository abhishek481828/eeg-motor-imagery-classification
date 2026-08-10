# Phase 1: Environment & Checkpoint Verification Audit Report

## Executive Summary
- **Audit Status**: **PASS**
- **Git Branch**: `experiments/post-final-improvements`
- **Tuned CNN Checkpoint**: `reports/experiments/new_benchmark/exp5_cnn_tuning/cnn_tuned_cfg_02_best.pt`
- **EEGNet Checkpoint**: `reports/experiments/new_benchmark/exp2_eegnet/eegnet_cfg_03_best.pt`
- **Subject Overlap**: **0** (Disjoint partitioning confirmed)
- **Validation Ensemble Accuracy**: **83.02%** (Expected: 83.02%)
- **Validation Ensemble Macro F1**: **0.8302** (Expected: 0.8302)
- **Test Set Protection**: **CONFIRMED (0 test subjects loaded or evaluated)**

---

## Subject Split Verification

| Partition | Subject Count | Subjects | Epoch Count | Shape |
|---|---|---|---|---|
| **Train** | 77 | S001–S077 | 3465 | `(3465, 64, 481)` |
| **Validation** | 16 | S078–S093 | 630 | `(630, 64, 481)` |
| **Test (Frozen)** | 16 | S094–S109 | -- | -- |
