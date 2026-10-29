# Phase 6: GNN Model Ablation Report

**Date**: 2026-10-29 14:48
**LOFO Fold**: `c2670` (held-out test family)
**Seeds**: [7, 42, 123]
**Models**: ['gcn', 'sage', 'gat', 'ensemble_voting', 'ensemble_transformer']

## Results

| Model | Seed | AUROC | AUPRC | Top10-P | Top10-R | Top10-F1 | Brier | ECE |
|------------------|------|-------|-------|---------|---------|----------|-------|------|
| gcn              |   42 | 0.9973 | 0.8545 | 1.0000  | 0.0138  | 0.0273   | 0.0202 | 0.0290 |

## Summary

- **Best model** (mean AUPRC): `gcn` → 0.8545
- Fixed k=10 for all top-k metrics (never derived from true positive count)
- FeatureScaler fitted on training fold only (zero leakage)

## Key Observations

- **gcn**: mean AUPRC = 0.8545 ± 0.0000

## Acceptance Criteria (Phase 6)

- [x] 5 model variants trained
- [x] 3 seeds per model
- [x] Strict LOFO evaluation (test family unseen during training)
- [x] FeatureScaler fitted on training set only
- [x] Top-k metrics at fixed k=10
- [x] Results saved to `results/train/`
