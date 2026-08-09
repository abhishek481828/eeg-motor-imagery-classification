"""Subject-Independent Model Evaluation Script for 2-Class Motor Imagery (Runs 4, 8, 12).

Evaluates trained CNN-LSTM checkpoint exclusively on real test subjects.
Never uses mock or synthetic fallback data. Raises FileNotFoundError if real EDF data is missing.
Prints prediction distribution and saves confusion matrix and classification report.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_mi.evaluation.confusion_matrix import plot_confusion_matrix
from eeg_mi.evaluation.metrics import compute_metrics
from eeg_mi.evaluation.reports import save_evaluation_report
from eeg_mi.models.factory import create_model
from eeg_mi.preprocessing.pipeline import PreprocessingPipeline
from eeg_mi.scripts.train import load_real_edf_dataset
from eeg_mi.utils.device import get_device
from eeg_mi.utils.logging import get_logger

logger = get_logger("EvaluationScript")


def run_subject_level_evaluation(
    checkpoint_path: Path,
    manifest_path: Path,
    raw_data_dir: Path,
    output_dir: Path,
    allowed_runs: list[int] | None = None,
    num_channels: int = 64,
    seq_len: int = 480,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Evaluate model checkpoint exclusively on real test subjects."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_runs = allowed_runs or [4, 8, 12]

    device = get_device("auto")
    logger.info(f"Evaluating 2-class checkpoint: {checkpoint_path} on device: {device}")

    # Load test subject list from split manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    test_subjects = manifest["splits"]["test"]

    # Preprocessing setup for real EDF files
    pipeline = PreprocessingPipeline(
        l_freq=7.0,
        h_freq=30.0,
        notch_freq=60.0,
        window_duration=3.0,
        allowed_runs=allowed_runs,
    )

    # STRICT CHECK: Load real EDF test set or raise FileNotFoundError
    logger.info("Loading REAL PhysioNet test EDF files (No mock data permitted)...")
    X_test, y_test, test_meta = load_real_edf_dataset(
        raw_data_dir, test_subjects, allowed_runs, pipeline
    )

    # Instantiate model architecture
    model = create_model("cnn_lstm", num_channels=num_channels, num_classes=num_classes)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Trained model checkpoint '{checkpoint_path.resolve()}' not found!\n"
            "Train the model first using `python scripts/train.py`."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Loaded checkpoint saved at epoch {checkpoint.get('epoch', 'N/A')}")

    model.to(device)
    model.eval()

    class_names = ["Left Fist", "Right Fist"][:num_classes]

    # Predict on test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Prediction Distribution Analysis
    unique_preds, pred_counts = np.unique(preds, return_counts=True)
    pred_dist = dict(
        zip(
            [class_names[int(p)] for p in unique_preds],
            [int(cnt) for cnt in pred_counts],
            strict=False,
        )
    )

    # Compute classification metrics
    aggregate_metrics = compute_metrics(y_test, preds, class_names=class_names)
    aggregate_metrics["prediction_distribution"] = pred_dist

    # Compute per-subject metrics
    per_subject_results: dict[int, dict[str, float]] = {}
    for sub_id in test_subjects:
        sub_indices = [
            i for i, meta in enumerate(test_meta) if meta["subject_id"] == f"S{sub_id:03d}"
        ]
        if sub_indices:
            sub_y = y_test[sub_indices]
            sub_pred = preds[sub_indices]
            sub_m = compute_metrics(sub_y, sub_pred, class_names=class_names)
            per_subject_results[sub_id] = {
                "accuracy": sub_m["accuracy"],
                "balanced_accuracy": sub_m["balanced_accuracy"],
                "macro_f1": sub_m["macro_f1"],
            }

    sub_accs = [res["accuracy"] for res in per_subject_results.values()]
    sub_f1s = [res["macro_f1"] for res in per_subject_results.values()]

    aggregate_metrics["subject_level"] = {
        "num_test_subjects": len(per_subject_results),
        "mean_accuracy": float(np.mean(sub_accs)) if sub_accs else 0.0,
        "std_accuracy": float(np.std(sub_accs)) if sub_accs else 0.0,
        "mean_macro_f1": float(np.mean(sub_f1s)) if sub_f1s else 0.0,
        "std_macro_f1": float(np.std(sub_f1s)) if sub_f1s else 0.0,
        "per_subject_breakdown": per_subject_results,
    }

    # Save metrics JSON & CSV tables
    save_evaluation_report(aggregate_metrics, output_dir)

    # Save confusion matrix plot
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        np.array(aggregate_metrics["confusion_matrix"]),
        class_names=class_names,
        save_path=cm_path,
    )

    print("\n" + "=" * 70)
    print("      REAL PHYSIONET 2-CLASS TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Samples    : {len(y_test)}")
    print(f"Prediction Dist.      : {pred_dist}")
    print(f"Overall Accuracy      : {aggregate_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy     : {aggregate_metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision       : {aggregate_metrics['macro_precision']:.4f}")
    print(f"Macro Recall          : {aggregate_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score        : {aggregate_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score     : {aggregate_metrics['weighted_f1']:.4f}")
    print(f"Cohen's Kappa         : {aggregate_metrics['cohens_kappa']:.4f}")
    print("-" * 70)
    print(
        f"Per-Subject Mean Acc  : {aggregate_metrics['subject_level']['mean_accuracy']:.4f} +/- {aggregate_metrics['subject_level']['std_accuracy']:.4f}"
    )
    print(
        f"Per-Subject Mean F1   : {aggregate_metrics['subject_level']['mean_macro_f1']:.4f} +/- {aggregate_metrics['subject_level']['std_macro_f1']:.4f}"
    )
    print(f"Reports Saved To      : {output_dir.resolve()}")
    print("=" * 70)

    return aggregate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 2-Class EEG Motor Imagery Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/cnn_lstm_2class_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/splits/subject_split.json",
        help="Path to subject split manifest JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/physionet",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments",
        help="Directory to save evaluation reports and plots",
    )
    args = parser.parse_args()

    run_subject_level_evaluation(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        raw_data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
