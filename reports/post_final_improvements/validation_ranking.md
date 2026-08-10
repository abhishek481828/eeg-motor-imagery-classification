# Post-Final Experiments: Validation Model Rankings (S078–S093)

> **REQUIRED STATEMENT**:
> **The official 80.98% test result remains frozen. All post-final experiments were conducted using training and validation subjects only and were not evaluated on the original test subjects.**

---

## Validation Summary Table

| Rank | Phase | Model Name | Total Params | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|
| 1 | Phase 2: Weight Search | Ensemble (w_cnn=0.45, w_eeg=0.55) | 191,700 | 83.02% | 0.8302 | 0.6603 |
| 2 | Phase 2: Weight Search | Ensemble (w_cnn=0.40, w_eeg=0.60) | 191,700 | 82.86% | 0.8286 | 0.6571 |
| 3 | Phase 2: Weight Search | Ensemble (w_cnn=0.50, w_eeg=0.50) | 191,700 | 82.22% | 0.8222 | 0.6445 |
| 4 | Phase 2: Weight Search | Ensemble (w_cnn=0.60, w_eeg=0.40) | 191,700 | 82.22% | 0.8222 | 0.6445 |
| 5 | Phase 2: Weight Search | Ensemble (w_cnn=0.30, w_eeg=0.70) | 191,700 | 81.75% | 0.8175 | 0.6349 |
| 6 | Phase 2: Weight Search | Ensemble (w_cnn=0.70, w_eeg=0.30) | 191,700 | 81.43% | 0.8142 | 0.6286 |
| 7 | Phase 3: Multi-Seed Avg | 5-Seed CNN Averaging Ensemble | 945,450 | 81.27% | 0.8123 | 0.6255 |
| 8 | Phase 3: Multi-Seed Avg | 10-Model Super Ensemble (5-Seed CNN + 5-Seed EEGNet, w=0.45/0.55) | 958,500 | 81.11% | 0.8109 | 0.6223 |
| 9 | Phase 4: Safe Augmentation | 1D-CNN (aug_mild_noise_dropout) | 189,090 | 80.79% | 0.8078 | 0.6159 |
| 10 | Phase 4: Safe Augmentation | 1D-CNN (aug_mild_shift_scaling) | 189,090 | 79.05% | 0.7905 | 0.5810 |
| 11 | Phase 3: Multi-Seed Avg | 5-Seed EEGNet Averaging Ensemble | 13,050 | 77.78% | 0.7778 | 0.5556 |

---

## Key Findings & Conclusions
- **Top Validation Model**: **Ensemble (w_cnn=0.45, w_eeg=0.55)** (Val Acc = **83.02%**, Val Macro F1 = **0.8302**).
- **Validation Outcome**: The reference Val-Weighted Ensemble (Tuned CNN + EEGNet, 83.02% Val Acc) remains the top-performing model on validation.
- **Official Test Benchmark**: **80.98% Test Accuracy** (Commit `5d7458d`) on $S094-S109$ remains frozen and untouched.
