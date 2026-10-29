"""Unit tests for Phase 2: Multi-view graph construction."""

import pytest
from pathlib import Path
from neuroguard.graphs.build_views import build_multiview_graph, EDGE_TYPE_DATA, EDGE_TYPE_CONTROL, EDGE_TYPE_STATE


def test_multiview_node_alignment_combinational():
    """Verify node alignment on combinational circuit c2670_T000."""
    netlist = Path("data/TRIT-TC/c2670_T000/c2670_T000.v")
    assert netlist.exists()
    data = build_multiview_graph(netlist)

    assert data.num_nodes > 0
    # Check node alignment
    assert len(data.y_node) == data.num_nodes
    assert len(data.y_role) == data.num_nodes
    assert len(data.node_names) == data.num_nodes

    # Check structural view has valid indices
    assert data.edge_index_struct.max().item() < data.num_nodes
    assert data.edge_index_struct.min().item() >= 0
    assert data.edge_attr_struct.shape[0] == data.edge_index_struct.shape[1]

    # Check functional view has valid indices
    assert data.edge_index_func.max().item() < data.num_nodes
    assert data.edge_index_func.min().item() >= 0
    assert data.edge_type_func.shape[0] == data.edge_index_func.shape[1]

    # Check PyG validate
    data.validate(raise_on_error=True)


def test_multiview_node_alignment_sequential():
    """Verify node alignment and edge types on sequential circuit s13207_T400."""
    netlist = Path("data/TRIT-TS/s13207_T400/s13207_T400.v")
    assert netlist.exists()
    data = build_multiview_graph(netlist)

    assert data.num_nodes == 2484
    assert len(data.y_node) == data.num_nodes
    assert len(data.y_role) == data.num_nodes
    assert len(data.node_names) == data.num_nodes

    # Verify presence of state and control edges in functional view
    unique_types = set(data.edge_type_func.tolist())
    assert EDGE_TYPE_CONTROL in unique_types, "Sequential design must have control edges (CLK/SSEL)"
    assert EDGE_TYPE_STATE in unique_types, "Sequential design must have state edges (DFF output fan-out)"

    # PyG validate
    data.validate(raise_on_error=True)

