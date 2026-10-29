"""Feedback loop package for NeuroGuard-X.

Converts LLM qualitative verdicts into soft distillation targets and InfoNCE
contrastive loss terms to retrain and continuously refine GNN detection.
"""

from neuroguard.feedback.soft_labels import verdict_to_soft_labels, merge_soft_labels
from neuroguard.feedback.contrastive import (
    TextProjector,
    info_nce_loss,
    compute_region_embeddings,
)
from neuroguard.feedback.retrain import retrain_with_feedback

__all__ = [
    "verdict_to_soft_labels",
    "merge_soft_labels",
    "TextProjector",
    "info_nce_loss",
    "compute_region_embeddings",
    "retrain_with_feedback",
]

