"""Functional view builder module."""
from pathlib import Path
from typing import Optional
from torch_geometric.data import Data
from neuroguard.graphs.build_views import build_multiview_graph

def build(netlist_path: str | Path, log_path: Optional[str | Path] = None) -> Data:
    """Build functional view Data object."""
    return build_multiview_graph(netlist_path, log_path)
