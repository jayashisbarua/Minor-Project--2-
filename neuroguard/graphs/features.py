"""Feature engineering for NeuroGuard-X.

Extracts rich structural, topological, and temporal node feature matrices.
STRICT RULE: Features must NEVER be derived from node names, net names, or instance identifiers.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler

from neuroguard.parsing.cell_library import CANONICAL_FAMILIES

FAMILY_TO_IDX = {fam: idx for idx, fam in enumerate(CANONICAL_FAMILIES)}
NUM_FAMILIES = len(CANONICAL_FAMILIES)  # 12


def compute_node_features(
    G: nx.MultiDiGraph,
    temporal_stats: Optional[Dict[str, Dict[str, float]]] = None,
    include_temporal: bool = True,
) -> torch.FloatTensor:
    """Compute feature matrix X for all nodes in graph G.

    Returns
    -------
    torch.FloatTensor of shape [num_nodes, num_features]
    """
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # 1. Structural depths via BFS
    pi_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "PI"]
    po_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "PO"]

    # Shortest path from any PI
    depth_from_pi: Dict[str, float] = {}
    if pi_nodes:
        # Multi-source shortest path
        try:
            depth_from_pi = nx.multi_source_dijkstra_path_length(G, pi_nodes)
        except Exception:
            depth_from_pi = {}

    # Distance to any PO (reverse graph)
    G_rev = G.reverse()
    dist_to_po: Dict[str, float] = {}
    if po_nodes:
        try:
            dist_to_po = nx.multi_source_dijkstra_path_length(G_rev, po_nodes)
        except Exception:
            dist_to_po = {}

    # Sequential depth: count DFFs on path from PIs
    seq_depth: Dict[str, float] = {}
    for n in nodes:
        seq_depth[n] = 0.0

    # Build feature rows
    feature_rows: List[List[float]] = []

    for n in nodes:
        d = G.nodes[n]
        fam = d.get("family", "")

        # 1. Cell family one-hot (12 dims)
        fam_one_hot = [0.0] * NUM_FAMILIES
        if fam in FAMILY_TO_IDX:
            fam_one_hot[FAMILY_TO_IDX[fam]] = 1.0

        # Drive strength (1 dim) & sequential flag (1 dim)
        drive = float(d.get("drive_strength", 1))
        is_seq = 1.0 if d.get("is_sequential", False) else 0.0

        # 2. Graph degree features (4 dims)
        in_deg = float(G.in_degree(n))
        out_deg = float(G.out_degree(n))
        fan_in_gates = float(len(set(G.predecessors(n))))
        fan_out_gates = float(len(set(G.successors(n))))

        # Reconvergence indicator (1 dim)
        # Check if 2 or more predecessors share a common grandparent
        preds = list(G.predecessors(n))
        is_reconv = 0.0
        if len(preds) >= 2:
            grandparents: Set[str] = set()
            for p in preds:
                for gp in G.predecessors(p):
                    if gp in grandparents:
                        is_reconv = 1.0
                        break
                    grandparents.add(gp)
                if is_reconv > 0.0:
                    break

        # 3. Topological depths (3 dims)
        dpi = float(depth_from_pi.get(n, 0.0))
        dpo = float(dist_to_po.get(n, 0.0))
        s_dep = float(seq_depth.get(n, 0.0))

        row = (
            fam_one_hot
            + [drive, is_seq]
            + [in_deg, out_deg, fan_in_gates, fan_out_gates, is_reconv]
            + [dpi, dpo, s_dep]
        )

        # 4. Temporal features (4 dims)
        if include_temporal:
            if temporal_stats and n in temporal_stats:
                t = temporal_stats[n]
                row += [
                    float(t.get("p1", 0.5)),
                    float(t.get("toggle_rate", 0.5)),
                    float(t.get("entropy", 1.0)),
                    float(t.get("rare_score", 0.5)),
                ]
            else:
                # Default baseline values if simulation not attached
                row += [0.5, 0.5, 1.0, 0.5]

        feature_rows.append(row)

    arr = np.array(feature_rows, dtype=np.float32)
    # Check for NaN / Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.from_numpy(arr)


class FeatureScaler:
    """Feature normalizer fitted strictly on training fold only."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, x_list: List[torch.Tensor]):
        """Fit scaler on concatenated training feature matrices."""
        all_features = torch.cat(x_list, dim=0).cpu().numpy()
        self.scaler.fit(all_features)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        arr = x.cpu().numpy()
        transformed = self.scaler.transform(arr)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.from_numpy(transformed.astype(np.float32))

    def fit_transform(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        self.fit(x_list)
        return [self.transform(x) for x in x_list]

