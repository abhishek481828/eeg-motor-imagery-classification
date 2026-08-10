# Reproduction Instructions: EEG Motor Imagery Improvement Suite

To reproduce all validation experiments (Phases 1 through 7):

```bash
# 1. Run Data Audit (Phase 1)
python scripts/data_audit.py

# 2. Run Subject-Adaptation / Calibration Experiment (Phase 6)
python scripts/run_calibration_experiment.py

# 3. Run Master Benchmark Suite (Phases 2-7)
python scripts/run_improvement_suite.py
```
