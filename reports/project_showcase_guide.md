# EEG Project Technical Showcase & Presentation Guide

> **Target Audience:** Professors, Advisors, Technical Reviewers  
> **Repository:** `eeg-motor-imagery-classification`  
> **Official Unseen-Test Benchmark:** **80.98% Official Unseen-Test Accuracy** (S094–S109)  
> **Best Validation Benchmark:** **83.02% Best Validation Accuracy** (S078–S093)  

---

## 1. Five-Minute Presentation Plan

### Minute 1: Research Problem & Subject-Independent Goal
* **Key Idea:** Motor Imagery (MI) BCI systems classify imagined movements (Left vs. Right Fist).
* **The Trap:** Random window splitting mixes trials from the same subject across training and testing, causing **data leakage** and falsely inflated accuracy claims (90%+).
* **Our Rigor:** We enforce strict **subject-independent splitting**: 77 training subjects ($S001$–$S077$), 16 validation subjects ($S078$–$S093$), and 16 completely unseen test subjects ($S094$–$S109$).

### Minute 2: Data-Quality Audit
* **Key Idea:** Before evaluating models, we audited all 109 subjects and 327 binary MI runs directly from raw EDF annotations.
* **Findings:** Zero corrupt files, 153 clean `VALID` runs, and 174 `VALID_WITH_WARNINGS` runs (non-standard 128 Hz sampling rates or truncated duration).
* **No Arbitrary Deletions:** We resampled 128 Hz subjects ($S088$, $S092$, $S100$) to 160 Hz cleanly instead of throwing them away.

### Minute 3: Dual-Branch Soft Voting Ensemble
* **Architecture:** Combines a multi-scale **1D-CNN** backbone (45% weight) with **EEGNet** spatial depthwise convolutions (55% weight).
* **Soft Voting:** Predictions are integrated via soft probability addition $P_{ens} = 0.45 P_{cnn} + 0.55 P_{eegnet}$.
* **Performance:**
  * **83.02% Best Validation Accuracy** ($S078$–$S093$)
  * **80.98% Official Unseen-Test Accuracy** ($S094$–$S109$)

### Minute 4: Interactive Dashboard Demonstration (`make dashboard`)
* Launch Streamlit app (`streamlit run app.py` or `make dashboard`).
* Walk through Section 1 (Overview), Section 2 (Dataset Split), Section 4 (Results), and Section 5 (Interactive EEG Trial Explorer).
* Demonstrate trial predictions (e.g. Trial #3, Trial #4, Trial #34, Trial #456) with live confidence bars.

### Minute 5: Summary, Limitations & Q&A
* Highlight zero-leakage normalization (`TrainFittedScaler`).
* Discuss limitations: offline stored test trials vs. real-time headset streaming.

---

## 2. Professor Explanation Script

> *"We built a subject-independent EEG motor-imagery classifier. The model uses a 1D-CNN and EEGNet weighted ensemble. It achieved 80.98% on an official unseen-subject test set. The dashboard demonstrates inference on individual stored EEG trials, while the reports provide the complete evaluation metrics."*

---

## 3. Exact Commands to Run

```bash
# 1. Environment & Pre-check
python scripts/check_environment.py

# 2. Complete Data-Quality Audit (Stage 1)
make audit

# 3. Interactive CLI Demo
make demo

# 4. Interactive Streamlit Showcase Dashboard
make dashboard
# Or:
streamlit run app.py
```

---

## 4. Accurate Explanation of 80.98% vs. 83.02%

* **83.02% Best Validation Accuracy:** Evaluated strictly on validation subjects $S078$–$S093$ during model architecture selection and weight tuning.
* **80.98% Official Unseen-Test Accuracy:** Evaluated **EXACTLY ONCE** on completely unseen test subjects $S094$–$S109$.
* ⚠️ **Critical Disclaimer:** Validation and test accuracy are measured on **different subject groups**. Never say accuracy "increased" or "decreased" between validation and test.

---

## 5. Screen-by-Screen Dashboard Guide

| Section | Content & What to Highlight |
|---|---|
| **1. Project Overview** | Explains EEG/MI BCI concepts and displays the visual pipeline flowchart. |
| **2. Dataset & Subject Split** | Shows 109 subjects split table, subject-level split warning box, and audit scan summary. |
| **3. Model Architecture** | Highlights 45% 1D-CNN + 55% EEGNet soft voting ensemble structure. |
| **4. Results & Accuracy** | Displays 80.98% Test Acc vs. 83.02% Val Acc cards and per-subject breakdown chart. |
| **5. Interactive EEG Trial Explorer** | Live inference on trial indices (0–672), preset buttons, 64-channel waveform plot, MATCH/MISMATCH cards. |
| **6. Data-Quality Audit** | Summarizes 109 subject audit status (153 VALID, 174 VALID_WITH_WARNINGS, 0 corrupt). |
| **7. Limitations & Future Work** | Clarifies stored test trials notice, research prototype disclaimer, and LSL future work. |

---

## 6. Common Questions & Answers (Q&A)

**Q1: Why is subject-independent evaluation harder than standard cross-validation?**  
*A: Standard cross-validation randomly splits signal windows, so windows from the same subject appear in both training and testing. Because EEG signals contain unique subject signatures, models cheat by memorizing subject identities. Subject-independent evaluation forces the model to learn true task patterns that generalize to new human brains.*

**Q2: Did you retrain or tune hyperparameters on the test set?**  
*A: No. Test subjects S094–S109 were held out completely until final single evaluation. Zero retraining was performed.*

**Q3: Does 99% confidence on a trial mean the model is 99% accurate overall?**  
*A: No. Confidence percentage is the model's soft probability estimate for that single individual trial; overall test accuracy is 80.98% across all 673 test epochs.*

---

## 7. Backup Plan if Dashboard Fails

If Streamlit cannot be launched or a GUI is unavailable:
1. Run the interactive CLI demo:
   ```bash
   make demo
   ```
2. Display the pre-generated Markdown reports:
   ```bash
   cat reports/data_quality/eegmmidb_quality_report.md
   cat reports/data_quality/side_by_side_comparison.csv
   ```
