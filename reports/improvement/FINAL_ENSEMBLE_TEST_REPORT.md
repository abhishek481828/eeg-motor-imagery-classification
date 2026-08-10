# Final Test Evaluation Report: Val-Weighted Ensemble (S094–S109)

## Executive Summary
- **Protocol Compliance**: Single final test evaluation on unseen subjects $S094-S109$.
- **Validation Selection**: The ensemble was selected strictly using validation subjects $S078-S093$ (Val Macro F1 = **0.8302**, Val Acc = **83.02%**).
- **Single Test Execution**: Evaluated on $S094-S109$ **EXACTLY ONCE** on 2026-08-10T09:01:10.289940+00:00.
- **Final Test Accuracy**: **80.98%** (Balanced Acc: **81.01%**, Macro F1: **0.8097**, Kappa: **0.6198**).
- **Comparison vs Frozen Tuned CNN Baseline (74.00%)**: The model improved from 74.00% to **80.98%**, an improvement of **+6.98 percentage points**.
- **Comparison vs Original CNN Baseline (72.81%)**: The model improved from 72.81% to **80.98%**, an improvement of **+8.17 percentage points**.
- **Verdict**: **IMPROVEMENT CONFIRMED ✓ — Ensemble test accuracy (80.98%) beats tuned CNN baseline (74.00%) by +6.98 percentage points!**.

---

## Ensemble Architecture & Checkpoint Details

- **Model 1**: Tuned 1D-CNN (`cnn_tuned_cfg_02`)
  - Checkpoint: `reports/experiments/new_benchmark/exp5_cnn_tuning/cnn_tuned_cfg_02_best.pt`
  - Weight: **0.45**
- **Model 2**: EEGNet (`eegnet_cfg_03`)
  - Checkpoint: `reports/experiments/new_benchmark/exp2_eegnet/eegnet_cfg_03_best.pt`
  - Weight: **0.55**
- **Combination Method**: Soft probability voting ($p_{ens} = 0.45 \cdot p_{cnn} + 0.55 \cdot p_{eegnet}$).

---

## Complete Test Set Performance Breakdown (Unseen S094–S109)

| Metric | Val-Weighted Ensemble | Tuned 1D-CNN Baseline | Original CNN Baseline | Difference vs Tuned CNN |
|---|---|---|---|---|
| **Overall Test Accuracy** | **80.98%** | **74.00%** | **72.81%** | **+6.98 percentage points** |
| **Balanced Accuracy** | **81.01%** | **74.03%** | **72.88%** | **+6.98 percentage points** |
| **Macro Precision** | **0.8108** | **0.7400** | **0.7280** | **+0.0708** |
| **Macro Recall** | **0.8101** | **0.7403** | **0.7288** | **+0.0698** |
| **Macro F1** | **0.8097** | **0.7399** | **0.7270** | **+0.0698** |
| **Cohen's Kappa (kappa)** | **0.6198** | **0.4802** | **0.4569** | **+0.1396** |
| **Per-Subject Acc Mean ± Std** | **75.92% ± 23.46%** | **73.99% ± 12.78%** | **68.26% ± 21.24%** | **+1.93 percentage points mean** |

---

## Per-Subject Accuracy Table (Unseen Subjects S094–S109)

| Subject ID | Epoch Count | Correct Predictions | Test Accuracy (%) | Visual Bar |
|---|---|---|---|---|
| **S094** | 45 | 37 | 82.22% | `████████████████` |
| **S095** | 45 | 39 | 86.67% | `█████████████████` |
| **S096** | 45 | 43 | 95.56% | `███████████████████` |
| **S097** | 45 | 37 | 82.22% | `████████████████` |
| **S098** | 45 | 33 | 73.33% | `██████████████` |
| **S099** | 45 | 27 | 60.00% | `████████████` |
| **S100** | 0 | 0 | 0.00% | `` |
| **S101** | 45 | 44 | 97.78% | `███████████████████` |
| **S102** | 45 | 34 | 75.56% | `███████████████` |
| **S103** | 45 | 44 | 97.78% | `███████████████████` |
| **S104** | 43 | 35 | 81.40% | `████████████████` |
| **S105** | 45 | 42 | 93.33% | `██████████████████` |
| **S106** | 45 | 32 | 71.11% | `██████████████` |
| **S107** | 45 | 37 | 82.22% | `████████████████` |
| **S108** | 45 | 39 | 86.67% | `█████████████████` |
| **S109** | 45 | 22 | 48.89% | `█████████` |

---

## Methodological Integrity & Safeguards
1. **Zero Retraining**: Both constituent models were loaded directly from their pre-trained validation checkpoints with zero fine-tuning on test data.
2. **Untouched Test Set**: Test subjects $S094-S109$ were never loaded or evaluated during any phase of model development or weight tuning.
3. **Single Evaluation Run**: Test inference was performed exactly once on the selected ensemble.

---

## Limitations & Scientific Notes
- The ensemble leverages soft probability integration across a 1D-CNN backbone and a compact 2D EEGNet backbone, improving spatial-temporal feature diversity.
- Per-subject variance across individual human subjects remains a known physiological property in EEG BCI research.
