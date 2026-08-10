# Final Benchmark Summary Report: Subject-Independent EEG Motor Imagery

## Executive Summary
- **Primary Selection Metric**: Validation Macro F1 on validation subjects $S078-S093$.
- **Validation Winner**: **CNN Tuned (cnn_tuned_cfg_02)** (Val Macro F1 = **0.8032**, Val Accuracy = **80.32%**).
- **Single Final Test Evaluation**: Test subjects $S094-S109$ evaluated **EXACTLY ONCE** on the single validation winner.
- **Winner Test Accuracy**: **74.00%** (Balanced Acc: **74.03%**, Macro F1: **0.7399**, Kappa: **0.4802**).
- **Frozen 1D-CNN Baseline Benchmark**: Test Accuracy = **72.81%**.
- **Verdict**: **IMPROVEMENT CONFIRMED ✓ — Beats frozen baseline (74.00% > 72.81%)**.

---

## Validation Ranking Table (All Experiments)

| Rank | Model Name | Exp ID | Total Params | Best Epoch | Val Acc (%) | Val Macro F1 | Val Kappa |
|---|---|---|---|---|---|---|---|
| 1 | CNN Tuned (cnn_tuned_cfg_02) | exp5_cnn_tuning | 189,090 | 28 | 80.32% | 0.8032 | 0.6064 |
| 2 | CNN Tuned (cnn_tuned_cfg_04) | exp5_cnn_tuning | 685,378 | 29 | 79.84% | 0.7979 | 0.5970 |
| 3 | CNN Tuned (cnn_tuned_cfg_01) | exp5_cnn_tuning | 27,474 | 27 | 79.37% | 0.7936 | 0.5873 |
| 4 | CNN Tuned (cnn_tuned_cfg_03) | exp5_cnn_tuning | 311,970 | 23 | 79.21% | 0.7917 | 0.5842 |
| 5 | Frozen 1D-CNN Baseline (Reference) | frozen_baseline | 65,826 | 26 | 78.57% | 0.7856 | 0.5715 |
| 6 | CNN Tuned (cnn_tuned_cfg_05) | exp5_cnn_tuning | 189,090 | 17 | 78.41% | 0.7841 | 0.5683 |
| 7 | EEGNet (eegnet_cfg_03) | exp2_eegnet | 2,610 | 28 | 77.46% | 0.7745 | 0.5492 |
| 8 | EEGNet (eegnet_cfg_04) | exp2_eegnet | 2,610 | 30 | 77.30% | 0.7729 | 0.5461 |
| 9 | EEGNet (eegnet_cfg_10) | exp2_eegnet | 2,610 | 30 | 76.35% | 0.7635 | 0.5270 |
| 10 | EEGNet (eegnet_cfg_02) | exp2_eegnet | 2,610 | 23 | 76.19% | 0.7618 | 0.5238 |
| 11 | EEGNet (eegnet_cfg_09) | exp2_eegnet | 2,610 | 30 | 76.03% | 0.7603 | 0.5207 |
| 12 | EEGNet (eegnet_cfg_01) | exp2_eegnet | 2,610 | 28 | 75.71% | 0.7571 | 0.5143 |
| 13 | EEGNet (eegnet_cfg_12) | exp2_eegnet | 2,610 | 30 | 74.44% | 0.7443 | 0.4890 |
| 14 | EEGNet (eegnet_cfg_06) | exp2_eegnet | 2,610 | 30 | 73.81% | 0.7375 | 0.4764 |
| 15 | EEGNet (eegnet_cfg_05) | exp2_eegnet | 2,610 | 30 | 73.49% | 0.7344 | 0.4700 |
| 16 | EEGNet (eegnet_cfg_11) | exp2_eegnet | 2,610 | 30 | 72.86% | 0.7285 | 0.4572 |
| 17 | EEGNet (eegnet_cfg_08) | exp2_eegnet | 2,610 | 30 | 70.79% | 0.7078 | 0.4158 |
| 18 | EEGNet (eegnet_cfg_07) | exp2_eegnet | 2,610 | 28 | 70.00% | 0.7000 | 0.4000 |
| 19 | EEGNet (eegnet_cfg_13) | exp2_eegnet | 2,610 | 30 | 67.14% | 0.6714 | 0.3428 |
| 20 | EEGNet (eegnet_cfg_14) | exp2_eegnet | 2,610 | 30 | 67.14% | 0.6714 | 0.3428 |
| 21 | EEGNet (eegnet_cfg_15) | exp2_eegnet | 2,610 | 29 | 64.29% | 0.6425 | 0.2856 |
| 22 | EEGNet (eegnet_cfg_16) | exp2_eegnet | 2,610 | 29 | 64.13% | 0.6409 | 0.2824 |
| 23 | FBCSP+LDA (k_features=20) | exp4_fbcsp_lda | 20 | 1 | 63.81% | 0.6370 | 0.2764 |
| 24 | FBCSP+LDA (k_features=10) | exp4_fbcsp_lda | 10 | 1 | 63.33% | 0.6329 | 0.2668 |
| 25 | FBCSP+LDA (k_features=all) | exp4_fbcsp_lda | 30 | 1 | 62.38% | 0.6219 | 0.2480 |
| 26 | FBCSP+LDA (k_features=25) | exp4_fbcsp_lda | 25 | 1 | 61.90% | 0.6179 | 0.2384 |
| 27 | FBCSP+LDA (k_features=15) | exp4_fbcsp_lda | 15 | 1 | 61.75% | 0.6149 | 0.2353 |
| 28 | CSP+LDA (components=16) | exp3_csp_lda | 1,024 | 1 | 58.89% | 0.5881 | 0.1776 |
| 29 | CSP+LDA (components=10) | exp3_csp_lda | 640 | 1 | 58.25% | 0.5815 | 0.1648 |
| 30 | CSP+LDA (components=12) | exp3_csp_lda | 768 | 1 | 58.10% | 0.5800 | 0.1617 |
| 31 | CSP+LDA (components=8) | exp3_csp_lda | 512 | 1 | 57.46% | 0.5746 | 0.1492 |
| 32 | CSP+LDA (components=6) | exp3_csp_lda | 384 | 1 | 57.30% | 0.5730 | 0.1460 |
| 33 | CSP+LDA (components=4) | exp3_csp_lda | 256 | 1 | 56.67% | 0.5666 | 0.1334 |

---

## Final Test Evaluation Breakdown (Unseen Subjects S094–S109)

- **Overall Test Accuracy**: **74.00%**
- **Balanced Test Accuracy**: **74.03%**
- **Test Macro F1**: **0.7399**
- **Test Cohen's Kappa**: **0.4802**
- **Per-Subject Accuracy Mean \pm Std**: **73.99% \pm 12.78%**

### Per-Subject Accuracy Table
- **S094**:  75.6%  ███████████████  (45 epochs)
- **S095**:  77.8%  ███████████████  (45 epochs)
- **S096**:  86.7%  █████████████████  (45 epochs)
- **S097**:  82.2%  ████████████████  (45 epochs)
- **S098**:  71.1%  ██████████████  (45 epochs)
- **S099**:  51.1%  ██████████  (45 epochs)
- **S100**:   0.0%    (0 epochs)
- **S101**:  84.4%  ████████████████  (45 epochs)
- **S102**:  68.9%  █████████████  (45 epochs)
- **S103**:  93.3%  ██████████████████  (45 epochs)
- **S104**:  72.1%  ██████████████  (43 epochs)
- **S105**:  91.1%  ██████████████████  (45 epochs)
- **S106**:  60.0%  ████████████  (45 epochs)
- **S107**:  68.9%  █████████████  (45 epochs)
- **S108**:  77.8%  ███████████████  (45 epochs)
- **S109**:  48.9%  █████████  (45 epochs)

---

## Methodology & Leakage Prevention Verification
1. **Disjoint Partitioning**: $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$.
2. **Train-Fitted Transformation**: Normalization parameters, CSP spatial projections, and feature selectors were fitted strictly on training subjects ($S001-S077$).
3. **Validation Selection**: Hyperparameter search and candidate model selection were conducted using validation metrics ($S078-S093$) only.
4. **Single Test Evaluation**: The test set ($S094-S109$) was evaluated strictly once for the single winning model.

---

## Artifacts Generated
- Data Integrity Report: `reports/experiments/new_benchmark/exp1_data_integrity.json`
- Experiment Validation Summary: `reports/experiments/new_benchmark/all_experiments_summary.csv`
- Winner Confusion Matrix: `reports/experiments/new_benchmark/winner_confusion_matrix.png`
- Winner Per-Subject Results: `reports/experiments/new_benchmark/winner_per_subject_results.json`
- Final Test Evaluation: `reports/experiments/new_benchmark/final_test_evaluation.json`
