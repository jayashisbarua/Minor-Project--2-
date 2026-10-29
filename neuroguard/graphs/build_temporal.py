# neuroguard/graphs/build_temporal.py
"""Temporal view builder stub.
The real implementation will attach per‑gate simulation statistics (p1,
toggle_rate, entropy, rare_score) computed in Phase 3.  For now it just returns an
empty ``Data`` object so that downstream code can import the function.
"""

from torch_geometric.data import Data

def build(graph_path: str) -> Data:
    """Return an empty temporal view.
    ``graph_path`` points to a cached parsed graph.
    """
    return Data()

