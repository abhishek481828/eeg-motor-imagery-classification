# Stage 2: Quality-Controlled Evaluation & Secondary Analysis Report

> **Generated:** 2026-08-20T07:23:03.065298+00:00
> **Status:** SECONDARY ANALYSIS ONLY — The original **80.98%** test baseline is **FROZEN & UNCHANGED**.

---

## Executive Summary

This report evaluates the **Quality-Controlled Protocol (Protocol B)** alongside the **Original Benchmark (Protocol A)** on frozen test subjects $S094-S109$.

---

## Side-by-Side Performance Comparison

| Model Architecture | Protocol A (Original Dataset) | Protocol B (Quality-Controlled) | Difference |
|---|---|---|---|
| **Tuned 1D-CNN (Single Model)** | **74.00%** | **74.00%** | **+0.00 percentage points** |
| **Val-Weighted Ensemble (CNN + EEGNet)** | **80.98%** | **80.98%** | **+0.00 percentage points** |

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
> 1. **No Model Retuning**: Models were evaluated directly using frozen checkpoints without tuning hyperparameters on test data.
> 2. **Predeclared Rules**: Exclusions were defined during the annotation audit stage, not chosen post-hoc to increase accuracy.
> 3. **Primary Result Preserved**: The original official test result (**80.98%**) remains the official primary benchmark for this dataset.
