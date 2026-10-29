# Phase 9: Feedback Loop & Retraining Report

**Date**: 2026-10-29  
**Module**: `neuroguard.feedback` (`soft_labels.py`, `contrastive.py`, `retrain.py`)

## 1. Summary of Implementation

Phase 9 closes the human/LLM-in-the-loop active learning cycle:
- **Soft Target Distillation**:
  - Converts high-level LLM verdicts into localized node-level probability targets:
    - `trojan_detected` $\to$ $\text{confidence} / 100$
    - `likely_clean` $\to$ $(100 - \text{confidence}) / 100$
    - `inconclusive` $\to$ $0.5$
  - Nodes outside the flagged suspicious region preserve their original GNN scores.
  - Soft targets are interpolated with existing predictions via $\alpha=0.3$:
    $$\tilde{y}_i = \alpha y_i^{\text{LLM}} + (1 - \alpha) \hat{y}_i^{\text{GNN}}$$
- **InfoNCE Multimodal Contrastive Alignment**:
  - `TextProjector` maps categorical verdict metadata (confidence, verdict one-hot, evidence item counts) to the GNN hidden representation space ($d=64$).
  - Mean-pooled GNN region embeddings are pulled closer to the corresponding LLM text embeddings while pushing apart non-region representations via symmetric InfoNCE loss.
- **Feedback-Augmented Retraining**:
  - Composite loss function:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{distill}} + \gamma \mathcal{L}_{\text{contrastive}}$$
  - Default hyper-parameters: $\beta = 0.1, \gamma = 0.05$.

## 2. Experimental Verification

Running `experiments/p9_feedback_demo.py` on `c2670_T000`:
- Graph: 1065 nodes
- Injected LLM Feedback: `trojan_detected` with 85% confidence on nodes 10–25
- Soft Target Generation: Region node values set to 0.85; non-region set to 0.00
- Interpolated Target: Region mean adjusted from initial 0.65 to 0.71
- Retraining Demonstration:
  - Epoch 1: $\mathcal{L}_{\text{total}} = 0.6751$
  - Epoch 2: $\mathcal{L}_{\text{total}} = 0.6596$
  - Epoch 3: $\mathcal{L}_{\text{total}} = 0.6548$
- Unit test suite: 5/5 passing in `tests/test_feedback.py`

## 3. Acceptance Criteria Checklist

- [x] Soft label generation from LLM confidence (`verdict_to_soft_labels`)
- [x] Target merging preserving outside-region node scores (`merge_soft_labels`)
- [x] `TextProjector` mapping LLM verdict features to hidden dimensions
- [x] InfoNCE contrastive alignment loss (`info_nce_loss`)
- [x] Retraining loop with distillation and contrastive loss (`retrain_with_feedback`)
- [x] Unit test suite (`tests/test_feedback.py` - 5/5 passed)
- [x] Demonstration run artifact (`results/p9_feedback_demo.json`)

