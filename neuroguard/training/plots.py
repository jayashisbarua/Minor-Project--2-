"""Plotting utilities for generating training and evaluation curves."""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_training_curves(
    metrics_history: List[Dict[str, Any]],
    best_epoch: int,
    output_dir: Path | str,
):
    """Plot curves_loss.png and curves_metrics.png."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics_history]
    train_loss = [m.get("train_loss", 0.0) for m in metrics_history]
    val_loss = [m.get("val_loss", 0.0) for m in metrics_history]

    # 1. Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", lw=2)
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curves_loss.png", dpi=150)
    plt.close()

    # 2. Metrics Curve
    val_auprc = [m.get("val_auprc", 0.0) for m in metrics_history]
    val_auroc = [m.get("val_auroc", 0.0) for m in metrics_history]
    val_f1 = [m.get("val_f1", 0.0) for m in metrics_history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, val_auprc, label="Val AUPRC (Primary)", color="#2ca02c", lw=2)
    plt.plot(epochs, val_auroc, label="Val AUROC", color="#9467bd", lw=2)
    plt.plot(epochs, val_f1, label="Val F1", color="#d62728", lw=1.5, linestyle=":")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("Validation Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curves_metrics.png", dpi=150)
    plt.close()


def plot_test_evaluation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path | str,
):
    """Plot PR curve, ROC curve, and confusion matrix on the test fold."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#2ca02c", lw=2)
    plt.title("Precision-Recall Curve (Test Fold)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("ROC Curve (Test Fold)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 3. Confusion matrix
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Trojan"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

