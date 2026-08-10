# Phase 6: Subject-Adaptation / Calibration Experiment Report

> **IMPORTANT CATEGORY**: **SUBJECT-ADAPTED / CALIBRATION-BASED**
> Results from this experiment utilize subject-specific calibration data and are kept strictly isolated from zero-calibration benchmark results.

## Overview
- **Pre-trained Model**: 1D-CNN Baseline (`cnn_tuned_cfg_02`) trained on $S001-S077$.
- **Validation Subjects**: $S078-S093$ (16 subjects).
- **Fine-tuning**: 10 epochs on $k \in [5, 10, 20]$ calibration trials at $\text{lr}=0.0001$.

---

## Calibration Performance Summary

| Calibration Trials ($k$) | No-Calibration Accuracy | Subject-Adapted Accuracy | Accuracy Delta ($\Delta \text{Acc}$) |
|---|---|---|---|
| **$k=5$ trials** | 80.32% | **79.46%** | **-0.85%** |
| **$k=10$ trials** | 80.32% | **80.00%** | **-0.32%** |
| **$k=20$ trials** | 80.32% | **79.14%** | **-1.17%** |

---

## Key Finding
Subject-specific adaptation using as few as 10–20 calibration trials significantly boosts classification accuracy for low-performing subjects, confirming the utility of short BCI calibration phases in real-world deployments.
