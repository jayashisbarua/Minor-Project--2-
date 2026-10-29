"""Heuristic role hints for suspicious gates (no name leakage)."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import torch

# Canonical family indices (must match neuroguard.parsing.cell_library.CANONICAL_FAMILIES)
# 0:AND  1:AOI  2:BUF  3:DFF  4:INV  5:MUX  6:NAND  7:NOR  8:OAI  9:OR  10:XNOR  11:XOR
FAMILY_NAMES = ["AND", "AOI", "BUF", "DFF", "INV", "MUX", "NAND", "NOR", "OAI", "OR", "XNOR", "XOR"]
NUM_FAMILIES = len(FAMILY_NAMES)

# Family indices considered AND/OR-like (for trigger heuristic)
_AND_OR_FAMILIES = {"AND", "OR", "NAND", "NOR", "AOI", "OAI"}
# XOR-family indices (for payload heuristic)
_XOR_FAMILIES = {"XOR", "XNOR"}
# DFF family
_DFF_FAMILIES = {"DFF"}


def _decode_family(x_row: torch.Tensor) -> str:
    """Decode the one-hot family encoding from the first 12 features."""
    one_hot = x_row[:NUM_FAMILIES]
    idx = int(torch.argmax(one_hot).item())
    if one_hot[idx].item() < 0.5:
        return "UNKNOWN"
    return FAMILY_NAMES[idx]


def _compute_fan(
    edge_index: torch.LongTensor,
    node_indices: List[int],
) -> Dict[int, Dict[str, int]]:
    """Compute fan-in / fan-out for a set of nodes from edge_index.

    Uses the *full* edge_index so we count edges to nodes outside the region too.
    """
    fan: Dict[int, Dict[str, int]] = {n: {"fan_in": 0, "fan_out": 0} for n in node_indices}
    node_set = set(node_indices)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if d in node_set:
            fan[d]["fan_in"] += 1
        if s in node_set:
            fan[s]["fan_out"] += 1
    return fan


def assign_role_hints(data: Any, region: Any) -> List[Dict[str, Any]]:
    """Assign heuristic role hints to each node in a region.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Full graph (needed for edge_index and features x).
    region : Region
        A region dataclass from expand.py.

    Returns
    -------
    list[dict]
        One dict per node in ``region.node_indices`` (same order) with keys:
        ``node_idx``, ``cell_family``, ``fan_in``, ``fan_out``, ``role_hint``.
    """
    node_indices = region.node_indices
    fan = _compute_fan(data.edge_index, node_indices)

    # Determine which nodes drive an output (have fan_out == 0 in full graph)
    # or high fan_out as a proxy for driving primary outputs.
    # We use fan_out == 0 as a simple heuristic for "drives output".
    results: List[Dict[str, Any]] = []
    for nidx in node_indices:
        family = _decode_family(data.x[nidx])
        fi = fan[nidx]["fan_in"]
        fo = fan[nidx]["fan_out"]

        # Heuristic role assignment
        if family in _DFF_FAMILIES:
            role = "state_element"
        elif family in _XOR_FAMILIES and fo == 0:
            # XOR family that drives output → possible payload
            role = "possible_payload"
        elif fi >= 3 and family in _AND_OR_FAMILIES:
            # High fan-in + AND/OR-like → possible trigger
            role = "possible_trigger"
        else:
            role = "logic"

        results.append({
            "node_idx": nidx,
            "cell_family": family,
            "fan_in": fi,
            "fan_out": fo,
            "role_hint": role,
        })

    return results
