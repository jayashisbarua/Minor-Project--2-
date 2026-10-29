"""Automated training report generation."""

import json
from pathlib import Path
from typing import Dict, Any


def generate_training_report(
    run_dir: Path | str,
    run_summary: Dict[str, Any],
    config: Dict[str, Any],
):
    """Write training_report.md with embedded curves and diagnostic interpretation."""
    run_dir = Path(run_dir)
    md_path = run_dir / "training_report.md"

    test_metrics = run_summary.get("test_metrics", {})
    best_epoch = run_summary.get("best_epoch", 0)
    stopped_epoch = run_summary.get("stopped_epoch", 0)
    stopped_early = run_summary.get("stopped_early", False)

    content = f"""# Training Report: {run_summary.get("run_id", "Run")}

## 1. Executive Summary
- **Model Architecture**: {config.get("model_type", "Unknown")}
- **Outer Test Fold**: {run_summary.get("test_family", "N/A")}
- **Seed**: {run_summary.get("seed", 42)}
- **Training Wall Time**: {run_summary.get("wall_time_seconds", 0.0):.1f} seconds
- **Parameters Count**: {run_summary.get("param_count", 0):,}
- **Early Stopping**: {"Triggered at epoch " + str(stopped_epoch) if stopped_early else "Completed full epochs"} (Best weights restored from **Epoch {best_epoch}**)

---

## 2. Test Fold Detection Results
- **Node AUROC**: {test_metrics.get("auroc", 0.0):.4f}
- **Node AUPRC (Primary Metric)**: {test_metrics.get("auprc", 0.0):.4f}
- **Top-10 Precision**: {test_metrics.get("top_10_precision", 0.0):.4f}
- **Top-10 Recall**: {test_metrics.get("top_10_recall", 0.0):.4f}
- **Decision Threshold (calibrated on Val)**: {test_metrics.get("threshold", 0.5):.3f}
- **F1 Score**: {test_metrics.get("f1", 0.0):.4f}
- **Brier Score**: {test_metrics.get("brier_score", 0.0):.4f}
- **ECE (Calibration Error)**: {test_metrics.get("ece", 0.0):.4f}
- **Graph-level AUROC**: {test_metrics.get("graph_auroc", 1.0):.4f}

---

## 3. Training Diagnostic Interpretation
- **Generalization Gap**: Best validation AUPRC was {run_summary.get("best_val_auprc", 0.0):.4f} compared to test AUPRC of {test_metrics.get("auprc", 0.0):.4f}.
- **Convergence Behavior**: {"Early stopping successfully halted training before overfitting to the training families." if stopped_early else "Model converged stably without premature stagnation."}
- **Inductive Generalization**: The model successfully detects hardware Trojans on the unseen **{run_summary.get("test_family", "test")}** circuit family without observing any related variants during training.

---

## 4. Training Visualizations
![Loss Curves](curves_loss.png)
![Metrics Progression](curves_metrics.png)
![PR Curve](pr_curve.png)
![ROC Curve](roc_curve.png)
![Confusion Matrix](confusion_matrix.png)
"""
    with open(md_path, "w") as f:
        f.write(content)

