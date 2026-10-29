"""Statistical significance tests and bootstrap confidence intervals."""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from scipy import stats


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """Perform McNemar's test comparing model A vs model B with continuity correction."""
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b: A correct, B incorrect; c: A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    total = b + c
    if total == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # McNemar statistic with Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
    statistic = float((abs(b - c) - 1.0) ** 2 / total) if abs(b - c) > 1.0 else 0.0
    # Chi-square distribution with 1 degree of freedom
    p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "statistic": statistic,
        "p_value": p_value,
        "b_better_a": b,
        "c_better_b": c,
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical bootstrap confidence interval (mean, lower, upper)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return 0.0, 0.0, 0.0

    mean_score = float(np.mean(scores))
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))

    return mean_score, lower, upper

