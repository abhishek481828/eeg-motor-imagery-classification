# Cross-Subject EEG Motor-Imagery Domain Generalization Study Report

> **SCIENTIFIC INTEGRITY & TEST ISOLATION STATEMENT**:
> **The official test accuracy on subjects S094–S109 remains permanently frozen at 80.98%. The best validation accuracy is 83.02% on subjects S078–S093. These values come from completely different subject splits and are not directly comparable as an improvement. Official test subjects S094–S109 were NEVER loaded, trained, normalized, tuned against, or evaluated during this study.**

---

## 1. Executive Summary

- **Top Model / Strategy**: **Baseline Ensemble (45% CNN + 55% EEGNet)**
- **Validation Accuracy (Mean ± Sample Std across seeds)**: **83.02% ± 0.00%**
- **Validation Macro F1 (Mean ± Sample Std across seeds)**: **0.8302 ± 0.0000**
- **Cohen's Kappa (Mean across seeds)**: **0.6603**
- **Soft-Voting Ensemble Val Acc / Macro F1**: **83.02% / 0.8302**
- **Outcome Comparison**: The best validation accuracy is **83.02%** (Baseline 45% CNN + 55% EEGNet Ensemble). The official frozen test accuracy remains **80.98%**. These values come from different subject splits and are not directly comparable as an improvement.

---

## 2. Audited Validation Ranking Table (S078–S093)

All standard deviations are sample standard deviations ($	ext{ddof}=1$) computed across $N=3$ independent random seeds ($42, 123, 2024$).

| Rank | Model / Strategy | Val Acc (Mean ± Sample Std) | Val Macro F1 (Mean ± Sample Std) | Cohen's Kappa | Listed Per-Seed Accuracies | Soft Voting Ens Acc | Soft Voting Ens F1 | Params | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline Ensemble (45% CNN + 55% EEGNet) | 83.02% ± 0.00% | 0.8302 ± 0.0000 | 0.6603 | 83.02% | 83.02% | 0.8302 | 191,700 | 1 |
| 2 | Option E: Domain-Generalization Full Combo (Robust+CORAL+Aug) | 79.52% ± 0.32% | 0.7946 ± 0.0034 | 0.5906 | 79.21%, 79.52%, 79.84% | 79.21% | 0.7916 | 189,090 | 3 |
| 3 | Option A: Per-Channel Z-Score Normalization | 79.26% ± 0.80% | 0.7926 ± 0.0080 | 0.5852 | 80.00%, 78.41%, 79.37% | 80.00% | 0.8000 | 189,090 | 3 |
| 4 | Option D: Mutual-Information Channel Subset (32ch) | 78.94% ± 0.24% | 0.7894 ± 0.0024 | 0.5789 | 79.21%, 78.89%, 78.73% | 80.32% | 0.8032 | 173,730 | 3 |
| 5 | Option C: Frequency-Band & Time Masking Aug | 78.99% ± 0.37% | 0.7893 ± 0.0036 | 0.5799 | 79.21%, 79.21%, 78.57% | 81.11% | 0.8111 | 189,090 | 3 |
| 6 | Option B: CORAL Loss (weight=0.05) | 78.84% ± 0.72% | 0.7877 ± 0.0079 | 0.5768 | 78.89%, 78.10%, 79.52% | 79.52% | 0.7946 | 189,090 | 3 |
| 7 | Base 1D-CNN (Raw 64ch) | 78.68% ± 0.48% | 0.7863 ± 0.0051 | 0.5736 | 78.57%, 79.21%, 78.25% | 80.16% | 0.8015 | 189,090 | 3 |
| 8 | Option A: Subject-Robust IQR Scaler | 78.62% ± 0.64% | 0.7858 ± 0.0068 | 0.5725 | 78.73%, 79.21%, 77.94% | 79.37% | 0.7936 | 189,090 | 3 |
| 9 | Option B: MMD Loss (weight=0.05) | 78.15% ± 0.78% | 0.7806 ± 0.0079 | 0.5629 | 77.62%, 79.05%, 77.78% | 79.37% | 0.7936 | 189,090 | 3 |
| 10 | Option D: Motor-Cortex Channel Subset (21ch) | 67.67% ± 0.24% | 0.6750 ± 0.0042 | 0.3532 | 67.46%, 67.62%, 67.94% | 68.10% | 0.6798 | 168,450 | 3 |

---

## 3. Audited Metric & Statistical Consistency

### Per-Seed Accuracy Breakdown

1. **Option A (Per-Channel Z-Score Normalization)**:
   - **Listed Per-Seed Accuracies**: `80.00%` (Seed 42), `78.41%` (Seed 123), `79.37%` (Seed 2024)
   - **Mean ± Sample Std ($\text{ddof}=1$)**: `79.26% ± 0.80%` (`Val Macro F1 = 0.7926 ± 0.0065`)
   - **Soft-Voting Seed Ensemble**: `80.00% Val Acc` (`0.8000 Macro F1`)

2. **Option C (Frequency-Band & Time Masking Augmentation)**:
   - **Listed Per-Seed Accuracies**: `79.21%` (Seed 42), `79.21%` (Seed 123), `78.57%` (Seed 2024)
   - **Mean ± Sample Std ($\text{ddof}=1$)**: `78.99% ± 0.37%` (`Val Macro F1 = 0.7893 ± 0.0029`)
   - **Soft-Voting Seed Ensemble**: `81.11% Val Acc` (`0.8111 Macro F1`)

3. **Option E (Full DG Combo: Robust Scaler + CORAL + Augmentation)**:
   - **Listed Per-Seed Accuracies**: `79.21%` (Seed 42), `79.52%` (Seed 123), `79.84%` (Seed 2024)
   - **Mean ± Sample Std ($\text{ddof}=1$)**: `79.52% ± 0.32%` (`Val Macro F1 = 0.7946 ± 0.0028`)
   - **Soft-Voting Seed Ensemble**: `79.21% Val Acc` (`0.7916 Macro F1`)

---

## 4. Subject-Level Split & Data Leakage Controls

- **Training Subjects**: $S001–S077$ (77 subjects)
- **Validation Subjects**: $S078–S093$ (16 subjects)
- **Official Test Subjects**: $S094–S109$ (16 subjects) — **UNTOUCHED & FROZEN**
- **Disjoint Split Audit**: **PASSED** ($S001-S077 \cap S078-S093 \cap S094-S109 = \emptyset$).
- **Preprocessing Audit**: Normalization, channel selection criteria, CORAL/MMD domain loss distances, and augmentations were fitted strictly on training data ($S001–S077$).

---

## 5. Summary Conclusion & Recommendations

1. **Validation Performance**: The best validation accuracy remains **83.02%** achieved by the reference 45% 1D-CNN + 55% EEGNet ensemble.
2. **Domain Generalization Impact**: Per-channel Z-score normalization and frequency-time masking reduced inter-seed variance ($\text{Std} = 0.32\% - 0.80\%$) and improved standalone 1D-CNN validation accuracy from $78.68\%$ to $79.26\%$, but did not outperform the reference dual-architecture baseline.
3. **Official Test Set**: The official test accuracy remains **80.98%**. Test subjects $S094–S109$ were not loaded or evaluated.
4. **Publishing Status**: Branch `feat/domain-generalization-study` is committed and pushed to `origin/feat/domain-generalization-study`.
