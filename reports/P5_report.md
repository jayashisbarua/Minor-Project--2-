# Phase 5 Report: Splits & Evaluation Foundation

## 1. Summary
- **Leave-One-Family-Out (LOFO) Splits (`splits.py`)**:
  - Implemented 8 strictly disjoint LOFO outer folds grouped by base design: `c2670`, `c3540`, `c5315`, `c6288`, `s13207`, `s1423`, `s15850`, `s35932`.
  - **Zero Data Leakage Guarantee**: All netlists sharing a base design (across TRIT-TC, TRIT-TS, and clean TjFree baselines) are quarantined together into the same test fold.
  - **Validation Separation**: The validation set is carved strictly out of an independent training family, guaranteeing that test families are never observed during validation, early stopping, or hyperparameter tuning.
- **Evaluation Metrics (`metrics.py`)**:
  - Node-level AUROC and AUPRC (Average Precision).
  - Fixed Top-k Precision, Recall, and F1 with $k = 10$. Fixed $k$ eliminates the Oracle leakage flaw identified in legacy pipelines.
  - Model calibration: Brier score and Expected Calibration Error (ECE).
- **Statistical Tests (`stats.py`)**:
  - McNemar paired test with Edwards' continuity correction.
  - Empirical 95% bootstrap confidence intervals.

---

## 2. LOFO Split Class Balance Summary

| Fold (Test Family) | Train Netlists | Val Netlists | Test Netlists | Test Nodes | Test Trojan Nodes | Positive Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **c2670** | 710 | 111 | 101 | 108,365 | 723 | 0.667% |
| **c3540** | 710 | 111 | 101 | 123,053 | 712 | 0.579% |
| **c5315** | 660 | 151 | 111 | 290,362 | 874 | 0.301% |
| **c6288** | 719 | 92 | 111 | 276,122 | 842 | 0.305% |
| **s13207** | 620 | 151 | 151 | 377,021 | 4,655 | 1.235% |
| **s1423** | 726 | 104 | 92 | 48,675 | 2,215 | 4.551% |
| **s15850** | 670 | 101 | 151 | 479,012 | 4,872 | 1.017% |
| **s35932** | 717 | 101 | 104 | 751,132 | 2,644 | 0.352% |

---

## 3. Key Observations
- Extreme class imbalance ($0.3\%$ to $4.5\%$ positive rate) demonstrates why accuracy is misleading and why node-level **AUPRC** is the necessary primary metric for model selection and early stopping.
- Unit tests (`tests/test_splits.py`) pass. Full statistics persisted in `results/p5_split_balance.json`.

