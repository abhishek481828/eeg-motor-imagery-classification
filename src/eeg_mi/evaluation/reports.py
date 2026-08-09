"""Evaluation report generation in JSON and CSV formats."""

import json
from pathlib import Path

import pandas as pd


def save_evaluation_report(metrics: dict[str, any], output_dir: Path) -> None:
    """Save metrics evaluation summary as JSON and CSV tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Flatten top-level scalar metrics into CSV summary
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    df = pd.DataFrame([scalar_metrics])
    df.to_csv(output_dir / "metrics_summary.csv", index=False)
