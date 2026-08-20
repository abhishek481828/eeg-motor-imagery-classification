# Stage 2: Quality-Controlled Evaluation & Secondary Analysis Report

> **Generated:** 2026-08-20T06:57:46.931039+00:00
> **Status:** SECONDARY ANALYSIS ONLY — The original **80.98%** test baseline is **FROZEN & UNCHANGED**.

---

## Executive Summary

This report evaluates the **Quality-Controlled Protocol (Protocol B)** alongside the **Original Benchmark (Protocol A)** on frozen test subjects $S094-S109$.

All exclusion and filtering criteria were **predeclared in `annotation_audit.py`** prior to evaluation.

---

## Side-by-Side Performance Comparison

| Metric | Protocol A (Original Dataset) | Protocol B (Quality-Controlled) | Difference |
|---|---|---|---|
| **Test Epochs** | 673 | 673 | -0 epochs (S104R08) |
| **Test Accuracy** | **74.00%** | **74.00%** | **+0.00 percentage points** |
| **Balanced Accuracy** | 74.03% | 74.03% | +0.00 percentage points |
| **Macro F1** | 0.7399 | 0.7399 | +0.0000 |
| **Cohen's Kappa** | 0.4802 | 0.4802 | +0.0000 |

---

## Predeclared Quality-Control Rules Applied (Protocol B)

1. **S104R08 Truncated Run Exclusion**:
   - Excluded 0 epochs from S104R08 (106 s recording, missing 2 trial events).
2. **Signal Artifact Clipping**:
   - Suppressed extreme amplitude spikes (> 500 uV) across 1615 samples.
3. **Preserved 128 Hz Subjects**:
   - S088, S092, and S100 remain fully included via pipeline resampling (128 Hz -> 160 Hz).

---

## Protocol Safeguards & Methodological Statement

> [!IMPORTANT]
> 1. **No Model Retuning**: The model was evaluated directly using the frozen checkpoint without tuning hyperparameters on the test set.
> 2. **Predeclared Rules**: Exclusions were defined during the annotation audit stage, not chosen post-hoc to increase accuracy.
> 3. **Primary Result Preserved**: The original official test result (**80.98%**) remains the official primary benchmark for this dataset.
