"""Evaluation metrics for NeuroGuard-X.

Implements node-level and graph-level detection metrics:
- AUROC and AUPRC (Average Precision)
- Top-k Precision, Recall, F1 with fixed k=10 (NEVER derived from true count)
- Calibration: Brier score, Expected Calibration Error (ECE)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, brier_score_loss


def compute_calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        bin_weight = np.sum(mask) / total_samples
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_node_metrics(
    y_true: np.ndarray | List[int],
    y_score: np.ndarray | List[float],
    k: int = 10,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate comprehensive node-level detection metrics.

    Parameters
    ----------
    y_true: Ground truth binary labels (0 or 1).
    y_score: Continuous probability scores in [0, 1].
    k: Fixed integer for Top-k metrics (default 10).
    threshold: Decision threshold for hard predictions.

    Returns
    -------
    Dictionary of metric names to float values.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    metrics: Dict[str, float] = {}

    # 1. AUROC & AUPRC
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    if pos_count > 0 and neg_count > 0:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auroc"] = 0.5
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["auprc"] = float(pos_count / len(y_true))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = 0.0

    # 2. Fixed Top-K metrics (NEVER derived from true count)
    k_eval = min(k, len(y_score))
    if k_eval > 0:
        top_k_indices = np.argsort(y_score)[::-1][:k_eval]
        top_k_hits = np.sum(y_true[top_k_indices] == 1)
        metrics[f"top_{k}_precision"] = float(top_k_hits / k_eval)
        metrics[f"top_{k}_recall"] = float(top_k_hits / max(1, pos_count))
        p = metrics[f"top_{k}_precision"]
        r = metrics[f"top_{k}_recall"]
        metrics[f"top_{k}_f1"] = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0

    # 3. Thresholded metrics & calibration
    if threshold is None:
        # Default threshold that maximizes F1 or 0.5
        threshold = 0.5

    y_pred = (y_score >= threshold).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics["threshold"] = float(threshold)
    metrics["precision"] = float(p)
    metrics["recall"] = float(r)
    metrics["f1"] = float(f1)

    # Calibration
    metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
    metrics["ece"] = float(compute_calibration_ece(y_true, y_score))

    return metrics


def compute_graph_metrics(
    y_true_graph: np.ndarray | List[int],
    y_score_graph: np.ndarray | List[float],
) -> Dict[str, float]:
    """Calculate graph-level detection AUROC and AUPRC."""
    y_true = np.asarray(y_true_graph, dtype=np.int64)
    y_score = np.asarray(y_score_graph, dtype=np.float64)

    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["graph_auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["graph_auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["graph_auroc"] = 1.0
        metrics["graph_auprc"] = 1.0
    return metrics

