"""Multi-view graph construction for NeuroGuard-X.

Generates unified PyTorch Geometric Data objects aligned across three views:
1. Structural View: Physical gate-to-gate connections, bidirectional with direction attributes.
2. Functional View: Signal-flow and control dependencies with typed edges (data, control, state, collapsed).
3. Temporal View: Placeholder for simulation-derived activation stats (p1, toggle, entropy, rare_score).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import networkx as nx
import torch
from torch_geometric.data import Data

from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.parsing.cell_library import CANONICAL_FAMILIES
from neuroguard.graphs.features import compute_node_features

BASE_DESIGNS = ["c2670", "c3540", "c5315", "c6288", "s13207", "s1423", "s15850", "s35932"]

# Edge types in functional view:
EDGE_TYPE_DATA = 0
EDGE_TYPE_CONTROL = 1
EDGE_TYPE_STATE = 2
EDGE_TYPE_COLLAPSED = 3


def extract_base_design(name: str) -> str:
    for bd in BASE_DESIGNS:
        if name.startswith(bd):
            return bd
    return "other"


def build_multiview_graph(
    netlist_path: str | Path,
    log_path: Optional[str | Path] = None,
    parser: Optional[NetlistParser] = None,
) -> Data:
    """Convert a netlist file into an aligned multi-view PyG Data object.

    Parameters
    ----------
    netlist_path: Path to the .v file.
    log_path: Optional path to log.txt.
    parser: Optional NetlistParser instance.

    Returns
    -------
    Data object containing:
        - num_nodes
        - edge_index_struct: [2, 2 * E]
        - edge_attr_struct: [2 * E, 2]  ([1, 0] forward, [0, 1] backward)
        - edge_index_func: [2, E_func]
        - edge_type_func: [E_func] (0: data, 1: control, 2: state, 3: collapsed)
        - y_node: [num_nodes] (binary)
        - y_role: [num_nodes] (0: clean, 1: trigger, 2: body, 3: payload)
        - y_graph: [1] (binary)
        - metadata: family, trojan_type, base_design, node_names
    """
    if parser is None:
        parser = NetlistParser()

    netlist_path = Path(netlist_path)
    G = parser.parse(netlist_path, log_path)

    nodes = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # -------------------------------------------------------------
    # 1. Structural View (Bidirectional with forward/backward flag)
    # -------------------------------------------------------------
    struct_src: List[int] = []
    struct_dst: List[int] = []
    struct_attr: List[List[float]] = []

    for u, v, _ in G.edges(data=True):
        u_idx, v_idx = node2idx[u], node2idx[v]
        # Forward edge
        struct_src.append(u_idx)
        struct_dst.append(v_idx)
        struct_attr.append([1.0, 0.0])
        # Backward edge
        struct_src.append(v_idx)
        struct_dst.append(u_idx)
        struct_attr.append([0.0, 1.0])

    if struct_src:
        edge_index_struct = torch.tensor([struct_src, struct_dst], dtype=torch.long)
        edge_attr_struct = torch.tensor(struct_attr, dtype=torch.float)
    else:
        edge_index_struct = torch.empty((2, 0), dtype=torch.long)
        edge_attr_struct = torch.empty((0, 2), dtype=torch.float)

    # -------------------------------------------------------------
    # 2. Functional View (Signal flow + control + state + collapsed)
    # -------------------------------------------------------------
    func_src: List[int] = []
    func_dst: List[int] = []
    func_types: List[int] = []

    control_ports = {"CLK", "CK", "SSEL", "EB", "CLRB", "SIN", "SIN0", "SIN1"}

    for u, v, edge_data in G.edges(data=True):
        u_idx, v_idx = node2idx[u], node2idx[v]
        u_node_data = G.nodes[u]
        load_port = edge_data.get("port", "").upper()

        if load_port in control_ports:
            edge_t = EDGE_TYPE_CONTROL
        elif u_node_data.get("family") == "DFF":
            edge_t = EDGE_TYPE_STATE
        else:
            edge_t = EDGE_TYPE_DATA

        func_src.append(u_idx)
        func_dst.append(v_idx)
        func_types.append(edge_t)

    # Collapse inverter / buffer chains:
    # If node v is INV or BUF with in-degree 1 and out-degree >= 1,
    # add direct bypassed edge from each pred u to each succ w
    for v in nodes:
        v_data = G.nodes[v]
        if v_data.get("family") in ("INV", "BUF"):
            preds = list(G.predecessors(v))
            succs = list(G.successors(v))
            if len(preds) == 1 and len(succs) > 0:
                u_idx = node2idx[preds[0]]
                for w in succs:
                    w_idx = node2idx[w]
                    func_src.append(u_idx)
                    func_dst.append(w_idx)
                    func_types.append(EDGE_TYPE_COLLAPSED)

    if func_src:
        edge_index_func = torch.tensor([func_src, func_dst], dtype=torch.long)
        edge_type_func = torch.tensor(func_types, dtype=torch.long)
    else:
        edge_index_func = torch.empty((2, 0), dtype=torch.long)
        edge_type_func = torch.empty((0,), dtype=torch.long)

    # -------------------------------------------------------------
    # 3. Node Features & Labels
    x = compute_node_features(G, include_temporal=True)
    y_node = torch.tensor([G.nodes[n].get("is_trojan", 0) for n in nodes], dtype=torch.long)
    y_role = torch.tensor([G.nodes[n].get("role", 0) for n in nodes], dtype=torch.long)
    y_graph = torch.tensor([G.graph.get("is_trojan", 0)], dtype=torch.long)

    name = G.graph.get("name", netlist_path.stem)
    base = extract_base_design(name)

    # Get trojan type from log if available
    trojan_type = "clean"
    if log_path and Path(log_path).exists():
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if "Type:" in line:
                    trojan_type = line.split("Type:")[1].strip()
                    break

    # Construct PyG Data
    data = Data(
        x=x,
        num_nodes=num_nodes,
        edge_index_struct=edge_index_struct,
        edge_attr_struct=edge_attr_struct,
        edge_index_func=edge_index_func,
        edge_type_func=edge_type_func,
        y_node=y_node,
        y_role=y_role,
        y_graph=y_graph,
    )

    # PyG standard edge_index defaults to structural view for standard GNN layers
    data.edge_index = edge_index_struct

    # Store metadata (kept outside features, names never leak into x)
    data.node_names = nodes
    data.family = base
    data.base_design = base
    data.trojan_type = trojan_type
    data.netlist_name = name

    data.validate(raise_on_error=True)
    return data

