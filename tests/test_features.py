"""Unit tests for Phase 4: Feature engineering."""

import pytest
import torch
import numpy as np
from pathlib import Path

from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.graphs.features import compute_node_features, FeatureScaler


def test_node_name_invariance():
    """Verify that node features are strictly invariant to node names (Zero Leakage)."""
    parser = NetlistParser()
    c17_path = Path("data/TjFree/c17/c17.v")
    assert c17_path.exists()
    G = parser.parse(c17_path)

    # Compute features on original graph
    x_orig = compute_node_features(G, include_temporal=False)
    assert not torch.isnan(x_orig).any()
    assert not torch.isinf(x_orig).any()

    # Create a copy with renamed nodes (e.g. anonymous ids)
    mapping = {node: f"ANON_{i}" for i, node in enumerate(G.nodes())}
    import networkx as nx
    G_renamed = nx.relabel_nodes(G, mapping, copy=True)

    x_renamed = compute_node_features(G_renamed, include_temporal=False)

    # Assert exactly identical features
    assert torch.allclose(x_orig, x_renamed, atol=1e-6), (
        "Feature values changed after renaming nodes! Features must be name-invariant."
    )


def test_feature_scaler():
    """Verify FeatureScaler fits on train and transforms without NaN/inf."""
    x1 = torch.randn(50, 26) * 10.0 + 5.0
    x2 = torch.randn(30, 26) * 10.0 + 5.0

    scaler = FeatureScaler()
    scaler.fit([x1])
    x1_norm = scaler.transform(x1)
    x2_norm = scaler.transform(x2)

    assert not torch.isnan(x1_norm).any()
    assert not torch.isnan(x2_norm).any()
    assert abs(x1_norm.mean().item()) < 1e-1

