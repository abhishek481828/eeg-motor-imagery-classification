# Paper Study: Validation Model Rankings (S078–S093)

> **STRICT ZERO-LEAKAGE RULE**: Test subjects $S094-S109$ were **NEVER** loaded or evaluated.
> Official final test score (**80.98%**, Commit `5d7458d`) remains permanently frozen.

## Summary Table

| Rank | Phase | Model Name | Epoch Len | Params | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|---|
| 1 | Val Ensemble Reference | Val-Weighted Ensemble (Tuned CNN + EEGNet, w=0.45/0.55) | 3.0s | 191,700 | 83.02% | 0.8302 | 0.6603 |
| 2 | Phase 2: Epoch Length | Val Ensemble (3.0s Window, 481 samples) | 3.0s | 191,700 | 83.02% | 0.8302 | 0.6603 |
| 3 | Phase 7: Augmentation | Real-Only Training Baseline | 3.0s | 189,090 | 80.32% | 0.8031 | 0.6064 |
| 4 | Phase 3: Preprocessing | 1D-CNN (bandpass_4_38Hz) | 3.0s | 189,090 | 79.52% | 0.7952 | 0.5905 |
| 5 | Phase 3: Preprocessing | 1D-CNN (bandpass_0_5_40_notch) | 3.0s | 189,090 | 78.73% | 0.7872 | 0.5747 |
| 6 | Phase 7: Augmentation | Real + WGAN-GP Synthetic Augmentation | 3.0s | 189,090 | 77.62% | 0.7750 | 0.5526 |
| 7 | Phase 3: Preprocessing | 1D-CNN (bandpass_0_5_40Hz) | 3.0s | 189,090 | 76.98% | 0.7693 | 0.5398 |
| 8 | Phase 2: Epoch Length | 1D-CNN (5.0s Window, 800 samples) | 5.0s | 189,090 | 76.83% | 0.7682 | 0.5365 |
| 9 | Phase 6: CNN-LSTM | True Sequence Temporal CNN-LSTM | 3.0s | 179,074 | 66.35% | 0.6634 | 0.3270 |
| 10 | Phase 5: Riemannian Features | Riemannian Tangent Space + LDA | 3.0s | 2,081 | 57.62% | 0.5752 | 0.1527 |
| 11 | Phase 5: Riemannian Features | Riemannian Tangent Space + Logistic Regression | 3.0s | 2,081 | 57.62% | 0.5750 | 0.1527 |
| 12 | Phase 4: Wavelet Features | Wavelet (DWT db4) + LDA | 3.0s | 101 | 54.29% | 0.5424 | 0.0859 |
| 13 | Phase 4: Wavelet Features | Wavelet (DWT db4) + SVM (RBF) | 3.0s | 3,353 | 52.70% | 0.5267 | 0.0538 |
