"""Tests for Phase 7: Context Expansion and Anonymization."""

import json
import re
import pytest
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

from neuroguard.context.expand import extract_suspicious_regions, _merge_overlapping
from neuroguard.context.roles import assign_role_hints
from neuroguard.context.to_json import regions_to_json, build_llm_context


@pytest.fixture
def sample_graph():
    """Create a synthetic graph with known features and edge index."""
    # 6 nodes: G0..G5
    # Let's say node 0 is high fan-in AND gate
    # node 1 is XOR gate driving nowhere
    # nodes 2..5 are logic
    edge_index = torch.tensor([
        [2, 3, 4, 0, 1],
        [0, 0, 0, 1, 5],
    ], dtype=torch.long)
    
    # 26 features per node
    x = torch.zeros((6, 26), dtype=torch.float32)
    # 0: AND -> index 0
    x[0, 0] = 1.0
    # 1: XOR -> index 11
    x[1, 11] = 1.0
    # 2: DFF -> index 3
    x[2, 3] = 1.0

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = 6
    data.node_names = [f"U_instance_{i}" for i in range(6)]
    return data


def test_no_name_leakage(sample_graph):
    """Verify that original node names are never present in generated context."""
    scores = np.array([0.95, 0.88, 0.1, 0.2, 0.15, 0.05])
    context = build_llm_context(sample_graph, scores, k=2, hops=2)
    
    json_str = json.dumps(context)
    for name in sample_graph.node_names:
        assert name not in json_str, f"Leaked instance name: {name}"

    assert "U_instance" not in json_str


def test_anonymous_gate_ids(sample_graph):
    """Verify that gates are assigned G0, G1, ... inside regions."""
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])
    context = build_llm_context(sample_graph, scores, k=2, hops=1)
    
    for reg in context["regions"]:
        for g in reg["gates"]:
            assert re.match(r"^G\d+$", g["id"]), f"Invalid gate id: {g['id']}"


def test_region_merging():
    """Test that overlapping neighborhoods merge into a single region."""
    # Seed 0 and Seed 1 share node 2
    nh0 = (0, {0, 1, 2})
    nh1 = (1, {2, 3, 4})
    merged = _merge_overlapping([nh0, nh1])
    
    assert len(merged) == 1
    seeds, nodes = merged[0]
    assert set(seeds) == {0, 1}
    assert nodes == {0, 1, 2, 3, 4}


def test_role_heuristics(sample_graph):
    """Verify role heuristics assign expected hints."""
    scores = np.array([0.95, 0.88, 0.1, 0.2, 0.15, 0.05])
    regions = extract_suspicious_regions(sample_graph, scores, k=2, hops=1)
    role_hints = assign_role_hints(sample_graph, regions[0])
    
    hints_by_orig = {rh["node_idx"]: rh["role_hint"] for rh in role_hints}
    # Node 0 has fan-in 3 and AND family -> possible_trigger
    assert hints_by_orig[0] == "possible_trigger"
    # Node 2 is DFF -> state_element
    if 2 in hints_by_orig:
        assert hints_by_orig[2] == "state_element"

