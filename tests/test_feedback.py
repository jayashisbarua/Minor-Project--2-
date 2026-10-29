"""Tests for Phase 9: Feedback Loop and Soft-label Distillation."""

import pytest
import torch
import torch.nn as nn

from neuroguard.feedback.soft_labels import verdict_to_soft_labels, merge_soft_labels
from neuroguard.feedback.contrastive import (
    TextProjector,
    info_nce_loss,
)


def test_soft_label_shapes():
    num_nodes = 50
    region_nodes = [3, 7, 12, 19]
    resp = {
        "verdict": "trojan_detected",
        "confidence_0_100": 80,
    }
    soft = verdict_to_soft_labels(resp, region_nodes, num_nodes)
    
    assert soft.shape == (num_nodes,)
    for idx in region_nodes:
        assert abs(soft[idx].item() - 0.8) < 1e-5
    # Nodes outside region must be 0.0
    outside = [i for i in range(num_nodes) if i not in region_nodes]
    for idx in outside:
        assert soft[idx].item() == 0.0


def test_soft_label_clean_verdict():
    num_nodes = 20
    region_nodes = [2, 5]
    resp = {
        "verdict": "likely_clean",
        "confidence_0_100": 90,
    }
    soft = verdict_to_soft_labels(resp, region_nodes, num_nodes)
    # (100 - 90)/100 = 0.1
    for idx in region_nodes:
        assert abs(soft[idx].item() - 0.1) < 1e-5


def test_merge_preserves_outside_region():
    num_nodes = 10
    gnn_scores = torch.tensor([0.2] * num_nodes)
    llm_soft = torch.zeros(num_nodes)
    llm_soft[3] = 0.9
    llm_soft[4] = 0.9

    merged = merge_soft_labels(gnn_scores, llm_soft, alpha=0.3)

    # Outside nodes should be exact original GNN score
    assert abs(merged[0].item() - 0.2) < 1e-6
    assert abs(merged[1].item() - 0.2) < 1e-6

    # Inside nodes: 0.3 * 0.9 + 0.7 * 0.2 = 0.27 + 0.14 = 0.41
    expected = 0.3 * 0.9 + 0.7 * 0.2
    assert abs(merged[3].item() - expected) < 1e-5
    assert abs(merged[4].item() - expected) < 1e-5


def test_text_projector_output_shape():
    projector = TextProjector(hidden_dim=64)
    x = torch.randn(4, 5)
    out = projector(x)
    assert out.shape == (4, 64)


def test_info_nce_loss_decreases_with_alignment():
    # Identical embeddings should have lower loss than random
    emb1 = torch.randn(3, 64)
    # Perfectly aligned text embeddings
    loss_aligned = info_nce_loss(emb1, emb1.clone())

    # Random / orthogonal text embeddings
    emb2 = torch.randn(3, 64)
    loss_random = info_nce_loss(emb1, emb2)

    assert loss_aligned.item() < loss_random.item()

