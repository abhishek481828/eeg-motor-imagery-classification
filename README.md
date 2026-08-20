# Reproducible EEG Motor Imagery Classification Using CNN-LSTM Deep Learning

[![CI Pipeline](https://github.com/eeg-research/eeg-motor-imagery-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/eeg-research/eeg-motor-imagery-classification/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source, production-grade research framework for **Subject-Independent EEG Motor Imagery Classification** using PyTorch, MNE-Python, and hybrid CNN-LSTM deep learning architectures applied to the **PhysioNet EEG Motor Movement/Imagery Dataset (v1.0.0)**.

---

## 1. Project Title
**Reproducible EEG Motor Imagery Classification Using CNN-LSTM Deep Learning** (`eeg-motor-imagery-classification`)

---

## 2. Project Status
> [!IMPORTANT]
> **Current status**: Full-dataset benchmark complete. All 109 PhysioNet subjects preprocessed and evaluated across 8 models using a strict subject-independent split. CNN Baseline achieves **72.81% test accuracy** on 16 completely unseen subjects (zero data leakage).

---

## 3. Research Objective
Motor Imagery (MI) EEG signals are non-stationary, low SNR, and highly variable across human subjects. Standard random signal-window splitting leads to severe **subject-level data leakage** and falsely inflated performance claims. 

This project provides an open research framework that:
1. Evaluates models exclusively on **unseen test subjects** (Subject-Independent Classification).
2. Enforces strict **zero data leakage** by fitting normalization parameters (`TrainFittedScaler`) only on training subjects.
3. Implements 1D Convolutional Neural Networks (spatial-frequency feature extraction) combined with Recurrent LSTMs (temporal dependency modeling).

---

## 4. Important Disclaimer
- **Active Research Prototype**: This codebase is an active research software prototype.
- **Results Are Empirical**: All benchmarks use real PhysioNet EDF recordings — no synthetic or mock data.
- **No Paper Reproduction Claims**: Accuracy values from the original literature are not claimed to be reproduced. Results reflect this specific preprocessing and model configuration.

---

## 5. Project Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[PhysioNet EDF Recordings S001-S109] --> B[Subject Splitter]
        B -->|70% Train Subjects| C1[Train Dataset]
        B -->|15% Val Subjects| C2[Validation Dataset]
        B -->|15% Test Subjects| C3[Test Dataset]
    end

    subgraph Preprocessing & Normalization
        C1 --> D[MNE Bandpass 7-30Hz & 60Hz Notch Filter]
        C2 --> D
        C3 --> D
        D --> E[TrainFittedScaler Normalization]
    end

    subgraph Deep Learning Architecture
        E --> H[PyTorch 1D-CNN + LSTM Hybrid]
    end

    subgraph Evaluation & Tracking
        H --> I[MLflow Tracking & Checkpointing]
        I --> J[Subject-Independent Evaluation & Reports]
    end
```

---

## 6. Repository Structure
```text
eeg-motor-imagery-classification/
├── README.md                          # Research documentation
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── CONTRIBUTING.md                    # Developer guidelines
├── CODE_OF_CONDUCT.md                 # Community guidelines
├── SECURITY.md                        # Security policy
├── pyproject.toml                     # Python dependencies & tooling configs
├── Makefile                           # Development task runner
├── .gitignore                         # Strict exclusion rules (data, models, logs)
├── .env.example                       # Safe environment variable templates
├── docker/
│   └── Dockerfile                     # Docker container spec
├── .github/
│   └── workflows/                     # CI workflows
├── configs/                           # Hydra YAML configurations
├── data/
│   ├── raw/                           # Location for downloaded PhysioNet EDF files
│   ├── processed/                     # Preprocessed data arrays
│   └── splits/                        # Subject split manifest
├── scripts/                           # CLI pipeline scripts
├── src/
│   └── eeg_mi/                        # Core Python package
├── tests/                             # Unit and integration test suite
└── reports/                           # Evaluation output directory
```

---

## 7. Environment Setup
Requires **Python 3.11+**. Using `uv` or `venv` is recommended.

```bash
# Clone repository
git clone https://github.com/eeg-research/eeg-motor-imagery-classification.git
cd eeg-motor-imagery-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"

# Verify installation
python scripts/check_environment.py
```

---

## 8. CPU-Only Setup Instructions
The framework operates seamlessly on standard CPU hardware without requiring an NVIDIA GPU or CUDA:
- Automatic selection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- CPU-safe defaults: `batch_size: 8`, `num_workers: 0`, `pin_memory: false`

```bash
# Force CPU execution explicitly
python scripts/train.py device=cpu batch_size=8 num_workers=0
```

---

## 9. Optional CUDA/GPU Instructions
If an NVIDIA GPU with CUDA is available, PyTorch will automatically utilize the GPU device:
```bash
python scripts/train.py device=cuda batch_size=32
```

---

## 10. PhysioNet Dataset Source
This project uses the **PhysioNet EEG Motor Movement/Imagery Dataset (v1.0.0)**:
- **DOI**: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)
- **URL**: [https://physionet.org/content/eegmmidb/1.0.0/](https://physionet.org/content/eegmmidb/1.0.0/)

---

## 11. Dataset Download Instructions
Raw EEG files must be downloaded manually using `wget` or `rsync`:

```bash
# Create target raw data directory
mkdir -p data/raw/physionet
cd data/raw/physionet

# Download complete dataset via wget (approx 1.2 GB)
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

---

## 12. Dataset Placement
Ensure raw EDF files are structured as follows:
```text
data/raw/physionet/
├── S001/
│   ├── S001R01.edf
│   ├── S001R02.edf
│   ├── S001R04.edf  (Motor Imagery Left vs Right Fist)
│   └── ...
├── S002/
└── S109/
```

---

## 13. Raw EEG Data Policy
> [!WARNING]
> Raw EEG recordings contain physiological biometric signals and large binary data files (`.edf`). **Raw EEG files must never be committed to Git or pushed to GitHub.** `.gitignore` strictly excludes `data/raw/` and `*.edf`.

---

## 14. Data Preprocessing
1. **Channel Selection**: Picks 64 EEG channels exclusively (`eeg=True`), discarding auxiliary non-EEG channels.
2. **Band-pass Filtering**: 7.0 Hz – 30.0 Hz FIR filter targeting Mu (8–12 Hz) and Beta (13–30 Hz) motor rhythms.
3. **Notch Filtering**: 60.0 Hz notch filter eliminating powerline interference.
4. **Window Segmentation**: Epochs segmented starting at event onset ($t_{\text{event}}$ to $t_{\text{event}} + 3.0\text{s}$).

---

## 15. Run-Specific Motor-Imagery Label Mapping
PhysioNet recordings feature run-specific task annotations. The 2-class motor imagery pipeline uses:
- **Runs 4, 8, 12 (Motor Imagery Left Fist vs Right Fist)**:
  - `T1` = Left Fist Motor Imagery $\to$ mapped to **`0`**
  - `T2` = Right Fist Motor Imagery $\to$ mapped to **`1`**
  - `T0` (Rest) events are explicitly ignored for the 2-class experiment.

---

## 16. Subject-Independent Train/Validation/Test Splitting
Fixed subject-independent split across 109 subjects:
- **Train Split (S001–S077)**: 77 subjects, 3,465 epochs
- **Validation Split (S078–S093)**: 16 subjects, 630 epochs
- **Test Split (S094–S109)**: 16 subjects, 673 epochs
- ⚠️ Note: S088, S092, S100 have no usable MI annotations in the PhysioNet source dataset.
- Generated via `data/splits/full_subject_split.json`.

---

## 17. Data-Leakage Prevention
- **Disjoint Partitioning**: $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$.
- **Train-Fitted Scaler**: Z-score normalization parameters ($\mu, \sigma$) are computed strictly on training subjects (`TrainFittedScaler`).
- **No Mock Fallback in Production**: Training and evaluation pipelines raise a clear `FileNotFoundError` if raw EDF data is missing rather than generating fake random noise.

---

## 18. Baseline Model Usage
Train a Linear Discriminant Analysis (LDA) baseline model on PSD features:
```bash
python scripts/train.py --config-name=experiments/baseline.yaml
```

---

## 19. CNN-LSTM Model Usage
Train the hybrid CNN-LSTM deep learning architecture:
```bash
python scripts/train.py experiments=cnn_lstm
```

---

## 20. GAN Augmentation Status
Conditional GAN data augmentation (`src/eeg_mi/augmentation/gan.py`) is implemented as an optional module. It is **disabled by default** (`use_gan_augmentation: false`) pending full experimental validation.

---

## 21. Training Commands
```bash
# Generate subject split manifest
python scripts/create_subject_splits.py

# Train 2-class CNN-LSTM model for 30 epochs
python scripts/train.py experiments.training.epochs=30
```

---

## 22. Evaluation Commands
Evaluate trained checkpoint on separate test subjects:
```bash
python scripts/evaluate.py --checkpoint models/checkpoints/cnn_lstm_2class_best.pt
```
Outputs `metrics_summary.json`, `metrics_summary.csv`, and `confusion_matrix.png` to `reports/experiments/`.

---

## 23. PhysioNet EEG Annotation & Data Quality Audit

A comprehensive, zero-leakage data-quality audit of all 109 PhysioNet subjects ($S001$–$S109$) and 327 binary motor-imagery runs ($R04, R08, R12$) is available in `src/eeg_mi/data_quality/`.

### Run Quality Audit (Stage 1)
```bash
python scripts/audit_eegmmidb_quality.py \
    --data-dir data/raw/physionet \
    --out-dir  reports/data_quality
```
Outputs in `reports/data_quality/`:
- `eegmmidb_quality_report.md` — Full Markdown audit report
- `eegmmidb_subject_run_audit.csv` & `.json` — Complete per-run audit database
- `invalid_or_warning_runs.csv` — Filtered anomaly manifest
- `trial_count_comparison.csv` — Expected vs usable trial counts

### Run Interactive Showcase Dashboard
```bash
streamlit run app.py
# Or:
make dashboard
```
Provides an interactive 8-section presentation dashboard:
1. **Project Overview:** High-level BCI concepts & visual architecture flowchart.
2. **Dataset & Split:** PhysioNet 109-subject breakdown, subject-level split warning, and audit scan counts.
3. **Model Architecture:** Ensemble configuration (45% 1D-CNN + 55% EEGNet) and complementary feature rationale.
4. **Results:** **80.98% Official Unseen-Test Accuracy** and **83.02% Best Validation Accuracy** performance cards and per-subject charts.
5. **Interactive EEG Trial Explorer:** Test trial selector (0–672), preset buttons, 64-channel waveform plots, MATCH/MISMATCH cards, and confidence scores.
6. **Data-Quality Audit:** Summary of 109 subjects audited and status breakdown.
7. **Limitations & Future Work:** Prototype notices and live streaming roadmap.

### Run Interactive CLI Demo
```bash
make demo
```
Allows manual trial selection (0–672) or random trial generation directly in the terminal with live probability output.

---

## 24. Testing Commands
Run unit and integration test suite:
```bash
# Run tests with pytest
pytest tests/

# Run tests with coverage
pytest --cov=src/eeg_mi tests/
```

---

## 24. Reproducibility Instructions

### Full 109-Subject Pipeline (recommended)
```bash
# Step 1: Download all 109 subjects (R04, R08, R12 only)
python scripts/download_all_subjects.py

# Step 2: Verify → Preprocess → Validate → Promote → Pre-flight summary
python scripts/prepare_full_dataset.py

# Step 3: Run 8-model benchmark
python scripts/benchmark_full_dataset.py
```

### Dev Subset (5 subjects, quick iteration)
```bash
python scripts/download_dev_subset.py
python scripts/preprocess_dataset.py
python scripts/train.py
```

---

## 25. Known Limitations
- **Subject Variability**: Inter-subject variability across 109 subjects is high, requiring domain adaptation or fine-tuning for real-world deployment.
- **Channel Density**: PhysioNet uses 64 channels; consumer BCI headsets typically feature 8–16 channels.

---

## 26. Ethical & Privacy Considerations
- EEG signals contain physiological biometric data. Raw dataset files must never be redistributed outside PhysioNet's license terms.
- Models developed in this repository are for academic research and software architecture exploration only—not for clinical medical diagnosis.

---

## 27. Future Real-Time EEG Headset Integration
The streaming prediction CLI ([`scripts/predict.py`](scripts/predict.py)) accepts raw EEG window tensors `(64, 480)` for real-time BCI integration (e.g. Lab Streaming Layer - LSL).

---

## 28. Citation Instructions
If using this software in your research, please cite:

```bibtex
@article{schalk2004bci2000,
  title={BCI2000: a general-purpose brain-computer interface (BCI) system},
  author={Schalk, Gerwin and McFarland, Dennis J and Hinterberger, Thilo and Birbaumer, Niels and Wolpaw, Jonathan R},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={51},
  number={6},
  pages={1034--1043},
  year={2004}
}
```

---

## 29. Benchmark Results (Full 109-Subject Dataset)

> All results use real PhysioNet EDF recordings. Zero mock or synthetic data.
> Training: S001–S077 (77 subjects). Validation: S078–S093. **Test: S094–S109 (never seen during training or model selection).**

### Validation Set Performance (S078–S093, 630 epochs)

| Model | Val Accuracy | Val Macro F1 | Cohen's κ | Train Time |
|---|---|---|---|---|
| Majority Baseline | 50.16% | 0.334 | 0.000 | < 1s |
| PSD + LDA | 56.51% | 0.562 | 0.131 | 9s |
| CSP + LDA | 56.67% | 0.567 | 0.133 | 28s |
| CSP + SVM | 57.94% | 0.574 | 0.158 | 20s |
| PSD + Random Forest | 58.10% | 0.580 | 0.162 | 12s |
| PSD + KNN | 50.63% | 0.505 | 0.012 | 9s |
| **CNN Baseline ★** | **78.57%** | **0.786** | **0.572** | 81s |
| CNN-LSTM | 51.75% | 0.517 | 0.035 | 124s |

### Test Set — Best Model: CNN Baseline (S094–S109, 673 epochs)

| Metric | Value |
|---|---|
| **Overall Test Accuracy** | **72.81%** |
| Balanced Accuracy | 72.88% |
| Macro F1 | 0.727 |
| Cohen's κ | 0.457 |
| Mean per-subject accuracy | 68.3% ± 21.2% |

### Controlled CNN-LSTM Hyperparameter Tuning Study (Strict 2-Phase Protocol)

> Phase 1 ranks 7 configurations on Validation Macro F1 ($S078-S093$) without loading test data.
> Phase 2 evaluates only the single winning configuration on Test ($S094-S109$) exactly once.

#### Phase 1 Validation Ranking ($S078-S093$)

| Config | Change | Params | Best Epoch | Val Macro F1 | Val Accuracy |
|---|---|---|---|---|---|
| 0 (`config_0_original`) | Reference (defaults) | 457,570 | 17 | 0.5291 | 53.02% |
| 1 (`config_1_hidden64`) | `lstm_hidden_size`: 128 → 64 | 272,226 | 6 | 0.5139 | 52.06% |
| **2 (`config_2_dropout03`) ★** | **`dropout`: 0.5 → 0.3** | **457,570** | **29** | **0.5885** | **59.05%** |
| 3 (`config_3_lr0003`) | `lr`: 0.001 → 0.0003 | 457,570 | 3 | 0.4999 | 51.59% |
| 4 (`config_4_gradclip1`) | `grad_clip`: None → 1.0 | 457,570 | 29 | 0.5411 | 55.71% |
| 5 (`config_5_lstm1layer`) | `lstm_layers`: 2 → 1 | 325,474 | 8 | 0.5148 | 51.59% |
| 6 (`config_6_patience20`) | `es_patience`: 10 → 20, max_epochs: 30 → 50 | 457,570 | 17 | 0.5291 | 53.65% |

#### Phase 2 Test Evaluation on Selected Winner (`config_2_dropout03`)

| Metric | Selected Winner (`config_2_dropout03`) | Frozen CNN Baseline |
|---|---|---|
| **Test Accuracy** | **53.49%** | **72.81%** |
| Macro F1 | 0.5325 | 0.7270 |
| Cohen's κ | 0.0713 | 0.4569 |

> **Conclusion**: While reducing dropout ($0.5 \to 0.3$) improved validation macro F1 ($0.5291 \to 0.5885$), the tuned CNN-LSTM architecture (**53.49% test acc**) remains substantially below the frozen 1D-CNN Baseline (**72.81% test acc**).


> CNN Baseline significantly outperforms all classical ML methods. CNN-LSTM did not converge on this dataset size/configuration within 30 epochs on CPU.

---

## 30. License
This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 30. Contribution Instructions
Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before submitting pull requests.
