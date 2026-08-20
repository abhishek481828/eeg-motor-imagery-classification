.PHONY: help install format lint test check inspect-data preprocess train evaluate audit demo dashboard clean

PYTHON ?= python3
VENV ?= .venv

help:
	@echo "EEG Motor Imagery Classification - Makefile Commands"
	@echo "---------------------------------------------------"
	@echo "make install      : Install project package and dependencies"
	@echo "make format       : Format code with Ruff"
	@echo "make lint         : Run Ruff linter and MyPy type checker"
	@echo "make test         : Run pytest test suite"
	@echo "make check        : Run environment check script"
	@echo "make audit        : Run complete Data Quality Audit & Stage 2 Evaluation"
	@echo "make demo         : Run interactive CLI demo"
	@echo "make dashboard    : Launch interactive Streamlit showcase dashboard"
	@echo "make inspect-data : Run dataset inspection script"
	@echo "make preprocess   : Run dataset preprocessing pipeline"
	@echo "make train        : Run CNN-LSTM model training"
	@echo "make evaluate     : Evaluate model checkpoint"
	@echo "make clean        : Remove build, test, and bytecode artifacts"

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ".[dev]"; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

format:
	ruff format src tests scripts

lint:
	ruff check src tests scripts
	mypy src/eeg_mi

test:
	pytest tests/

check:
	$(PYTHON) scripts/check_environment.py

inspect-data:
	$(PYTHON) scripts/inspect_dataset.py

preprocess:
	$(PYTHON) scripts/preprocess_dataset.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

audit:
	nix-shell -p "python312.withPackages(ps: with ps; [mne numpy pandas torch scipy scikit-learn matplotlib seaborn pyyaml tqdm])" --run "python3 scripts/audit_eegmmidb_quality.py --data-dir data/raw/physionet --out-dir reports/data_quality && python3 scripts/evaluate_quality_controlled.py"

demo:
	nix-shell -p "python312.withPackages(ps: with ps; [mne numpy pandas torch scipy scikit-learn matplotlib seaborn pyyaml tqdm])" --run "python3 scripts/interactive_demo.py"

dashboard:
	nix-shell -p "python312.withPackages(ps: with ps; [streamlit mne numpy pandas torch scipy scikit-learn matplotlib seaborn pyyaml tqdm])" --run "streamlit run app.py"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
