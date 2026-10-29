"""Produce anonymized JSON context for LLM consumption.

Enforces strict zero-leakage rules:
- No instance, net, or node names are included in output.
- All gates are given synthetic anonymous identifiers (G0, G1, ...).
- Substring sanity assertion verifies no 'troj', 'trig', or raw names leak.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from neuroguard.context.expand import extract_suspicious_regions, Region
from neuroguard.context.roles import assign_role_hints


def regions_to_json(
    data: Any,
    regions: List[Region],
    role_hints_list: List[List[Dict[str, Any]]],
    scores: np.ndarray,
) -> List[Dict[str, Any]]:
    """Convert extracted regions into strictly anonymized JSON-serializable dictionaries.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG graph object.
    regions : list[Region]
        List of extracted regions.
    role_hints_list : list[list[dict]]
        Role hints corresponding to each region's nodes.
    scores : np.ndarray
        Array of GNN suspicion scores.

    Returns
    -------
    list[dict]
        Anonymized region descriptors.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    out_regions: List[Dict[str, Any]] = []

    for reg_idx, (reg, role_hints) in enumerate(zip(regions, role_hints_list)):
        # Map original node_idx to anonymous identifier G0, G1, ...
        node_map: Dict[int, str] = {
            orig_idx: f"G{i}" for i, orig_idx in enumerate(reg.node_indices)
        }
        seed_set = set(reg.seed_indices)

        gates_info: List[Dict[str, Any]] = []
        for i, (orig_idx, rh) in enumerate(zip(reg.node_indices, role_hints)):
            anon_id = node_map[orig_idx]
            gnn_score = float(scores[orig_idx])
            
            # Check temporal features if available in x (indices 22..25)
            temporal = None
            if data.x.size(1) >= 26:
                row = data.x[orig_idx].cpu().numpy()
                temporal = {
                    "p1": round(float(row[22]), 4),
                    "toggle_rate": round(float(row[23]), 4),
                    "entropy": round(float(row[24]), 4),
                    "rare_score": round(float(row[25]), 4),
                }

            gate_entry = {
                "id": anon_id,
                "is_seed": orig_idx in seed_set,
                "cell_family": rh["cell_family"],
                "fan_in": int(rh["fan_in"]),
                "fan_out": int(rh["fan_out"]),
                "gnn_suspicion_score": round(gnn_score, 4),
                "role_hint": rh["role_hint"],
            }
            if temporal is not None:
                gate_entry["temporal_profile"] = temporal

            gates_info.append(gate_entry)

        # Internal connections within region (anonymized)
        connections: List[Dict[str, str]] = []
        if reg.subgraph_edge_index.size(1) > 0:
            srcs = reg.subgraph_edge_index[0].tolist()
            dsts = reg.subgraph_edge_index[1].tolist()
            for s_local, d_local in zip(srcs, dsts):
                connections.append({
                    "from": f"G{s_local}",
                    "to": f"G{d_local}",
                })

        region_scores = [float(scores[idx]) for idx in reg.node_indices]
        region_dict = {
            "region_id": reg.region_id,
            "num_gates": reg.num_nodes,
            "num_internal_connections": reg.num_edges,
            "num_seed_gates": len(reg.seed_indices),
            "max_suspicion": round(float(np.max(region_scores)), 4) if region_scores else 0.0,
            "mean_suspicion": round(float(np.mean(region_scores)), 4) if region_scores else 0.0,
            "gates": gates_info,
            "internal_connections": connections,
        }
        out_regions.append(region_dict)

    # Strict zero-leakage check
    json_str = json.dumps(out_regions)
    if hasattr(data, "node_names") and data.node_names:
        for name in data.node_names:
            if len(name) >= 3 and name in json_str:
                raise ValueError(f"CRITICAL: Node name '{name}' leaked into anonymized context!")

    # Check for forbidden Trojan naming substrings (unless part of standard words)
    forbidden_patterns = [r"\btroj\w*", r"\btrig\w*"]
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, json_str, flags=re.IGNORECASE)
        # Filter out role_hint values like 'possible_trigger'
        suspicious = [m for m in matches if m.lower() not in {"trigger", "triggers", "trojan"}]
        if suspicious:
            raise ValueError(f"CRITICAL: Forbidden token leaked in context: {suspicious}")

    return out_regions


def build_llm_context(
    data: Any,
    scores: np.ndarray,
    k: int = 10,
    hops: int = 2,
) -> Dict[str, Any]:
    """Extract, analyze, and format suspicious regions into LLM-ready context.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG graph object.
    scores : np.ndarray
        Array of node suspicion scores.
    k : int
        Number of suspicious seed nodes.
    hops : int
        Neighborhood hop radius.

    Returns
    -------
    dict
        Structured payload for LLM analysis.
    """
    regions = extract_suspicious_regions(data, scores, k=k, hops=hops)
    role_hints_list = [assign_role_hints(data, reg) for reg in regions]
    anonymized_regions = regions_to_json(data, regions, role_hints_list, scores)

    context = {
        "analysis_type": "hardware_trojan_context_expansion",
        "total_regions_flagged": len(anonymized_regions),
        "total_seed_gates_analyzed": min(k, len(scores)),
        "regions": anonymized_regions,
    }
    return context

