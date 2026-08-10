# Phase 7: Validation Ranking & Final Selection Report

> **STRICT ZERO-LEAKAGE PROTOCOL**: Test subjects $S094-S109$ have **NOT** been evaluated yet.
> Evaluation on the test set will occur **ONLY AFTER** explicit user review and approval of this validation report.

## Executive Summary
- **Baseline Reference**: Tuned 1D-CNN Baseline (`cnn_tuned_cfg_02`) Val Acc = **80.32%**, Val Macro F1 = **0.8032** (Test Acc: **74.00%**).
- **Overall Validation Winner**: **Val-Weighted Ensemble (Tuned CNN + EEGNet, w=0.45)** (Val Acc = **83.02%**, Val Macro F1 = **0.8302**).
- **Validation Improvement Status**: IMPROVEMENT ON VALIDATION ✓ — Candidate beats tuned CNN baseline on Val Macro F1.

---

## Top 10 Validation Model Rankings (S078–S093)

| Rank | Phase | Model Name | Total Params | Best Epoch | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|---|
| 1 | Phase 5: Val Ensemble | Val-Weighted Ensemble (Tuned CNN + EEGNet, w=0.45) | 191,700 | 1 | 83.02% | 0.8302 | 0.6603 |
| 2 | Baseline Reference | Tuned 1D-CNN Baseline (cnn_tuned_cfg_02) | 189,090 | 28 | 80.32% | 0.8032 | 0.6064 |
| 3 | Phase 4: Safe Augmentation | 1D-CNN + Augment (aug_combined) | 189,090 | 30 | 79.52% | 0.7952 | 0.5905 |
| 4 | Phase 4: Safe Augmentation | 1D-CNN + Augment (aug_shift) | 189,090 | 25 | 79.05% | 0.7904 | 0.5810 |
| 5 | Phase 4: Safe Augmentation | 1D-CNN + Augment (aug_noise) | 189,090 | 25 | 79.05% | 0.7903 | 0.5810 |
| 6 | Phase 2: Frequency-Band | 1D-CNN (4-30Hz) | 189,090 | 28 | 78.89% | 0.7888 | 0.5778 |
| 7 | Phase 2: Frequency-Band | 1D-CNN (8-35Hz) | 189,090 | 19 | 78.10% | 0.7806 | 0.5620 |
| 8 | Phase 3: CNN+CSP Fusion | CNN+CSP Fusion (n=4) | 448,034 | 27 | 77.62% | 0.7761 | 0.5523 |
| 9 | Phase 4: Safe Augmentation | 1D-CNN + Augment (aug_scaling) | 189,090 | 19 | 77.62% | 0.7760 | 0.5525 |
| 10 | Phase 4: Safe Augmentation | 1D-CNN + Augment (aug_drop) | 189,090 | 20 | 77.30% | 0.7730 | 0.5460 |

---

## Selected Validation Winner Details

- **Model Name**: `Val-Weighted Ensemble (Tuned CNN + EEGNet, w=0.45)`
- **Phase**: `Phase 5: Val Ensemble`
- **Validation Accuracy**: **83.02%**
- **Validation Balanced Accuracy**: **83.02%**
- **Validation Macro F1**: **0.8302**
- **Validation Cohen's Kappa**: **0.6603**
- **Checkpoint Path**: `N/A (Ensemble of Tuned CNN + EEGNet)`

---

## Next Action Required
Awaiting explicit user approval to run final single test evaluation on $S094-S109$ for the selected validation winner (`Val-Weighted Ensemble (Tuned CNN + EEGNet, w=0.45)`).
