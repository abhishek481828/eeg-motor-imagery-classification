"""Baseline Machine Learning Classifier (LDA / Logistic Regression) on EEG spectral features."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.features.frequency_domain import extract_psd_features
from eeg_mi.features.time_domain import extract_time_domain_features
from eeg_mi.utils.logging import get_logger

logger = get_logger("BaselineModel")


class BaselineClassifier:
    """Baseline machine learning model wrapper using LDA or Logistic Regression."""

    def __init__(self, model_type: str = "lda", solver: str = "lsqr"):
        self.model_type = model_type.lower()
        if self.model_type == "lda":
            self.clf = LinearDiscriminantAnalysis(solver=solver, shrinkage="auto")
        elif self.model_type == "logistic_regression":
            self.clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        else:
            raise ValueError(f"Unsupported baseline model_type: {model_type}")

    def extract_features(self, X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
        """Extract combined PSD and time-domain features from EEG window shape (N, C, T)."""
        psd_feats = extract_psd_features(X, sfreq=sfreq)  # (N, C * n_bands)
        time_feats = extract_time_domain_features(X)  # (N, C * 3)
        return np.concatenate([psd_feats, time_feats], axis=-1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sfreq: float = 160.0) -> None:
        """Extract features and fit classifier on training data."""
        logger.info(f"Extracting baseline features from {len(X_train)} training windows...")
        feats_train = self.extract_features(X_train, sfreq=sfreq)
        self.clf.fit(feats_train, y_train)
        logger.info(f"Fitted {self.model_type.upper()} baseline model.")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sfreq: float = 160.0,
        class_names: list[str] | None = None,
    ) -> dict[str, any]:
        """Extract features and evaluate classifier on separate test subjects."""
        logger.info(f"Evaluating baseline on {len(X_test)} test windows...")
        feats_test = self.extract_features(X_test, sfreq=sfreq)
        y_pred = self.clf.predict(feats_test)
        metrics = compute_metrics(y_test, y_pred, class_names=class_names)
        return metrics
