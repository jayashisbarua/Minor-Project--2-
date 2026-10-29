"""Context expansion: 2-hop neighborhood extraction around suspicious gates."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from collections import deque


@dataclass
class Region:
    """Merged neighborhood around one or more suspicious seed nodes."""
    region_id: str
    seed_indices: List[int]
    node_indices: List[int]
    seed_scores: List[float]
    subgraph_edge_index: torch.LongTensor
    num_nodes: int
    num_edges: int


def _build_adjacency(edge_index: torch.LongTensor, num_nodes: int) -> Dict[int, Set[int]]:
    """Build undirected adjacency list from edge_index."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)
    return adj


def _bfs_neighborhood(adj: Dict[int, Set[int]], seed: int, hops: int) -> Set[int]:
    """BFS-collect the k-hop neighborhood of a seed node."""
    visited: Set[int] = {seed}
    frontier = deque([(seed, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        for nbr in adj.get(node, set()):
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, depth + 1))
    return visited


def _merge_overlapping(neighborhoods: List[Tuple[int, Set[int]]]) -> List[Tuple[List[int], Set[int]]]:
    """Merge neighborhoods that share any node.

    Parameters
    ----------
    neighborhoods : list of (seed_idx, node_set) tuples

    Returns
    -------
    list of (seed_indices_list, merged_node_set) tuples
    """
    # Union-Find approach
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Assign each seed neighborhood an id (index in neighborhoods list)
    for i, _ in enumerate(neighborhoods):
        parent[i] = i

    # Build node → list of neighborhood indices
    node_to_nh: Dict[int, List[int]] = {}
    for i, (_, nodes) in enumerate(neighborhoods):
        for n in nodes:
            if n not in node_to_nh:
                node_to_nh[n] = []
            node_to_nh[n].append(i)

    # Merge neighborhoods sharing any node
    for indices in node_to_nh.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    # Group by root
    groups: Dict[int, Tuple[List[int], Set[int]]] = {}
    for i, (seed, nodes) in enumerate(neighborhoods):
        root = find(i)
        if root not in groups:
            groups[root] = ([], set())
        groups[root][0].append(seed)
        groups[root][1].update(nodes)

    return list(groups.values())


def _restrict_edge_index(edge_index: torch.LongTensor, node_set: Set[int]) -> torch.LongTensor:
    """Restrict edge_index to nodes in node_set and re-index from 0."""
    sorted_nodes = sorted(node_set)
    old_to_new = {old: new for new, old in enumerate(sorted_nodes)}

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    new_src, new_dst = [], []
    for s, d in zip(src, dst):
        if s in old_to_new and d in old_to_new:
            new_src.append(old_to_new[s])
            new_dst.append(old_to_new[d])

    if not new_src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([new_src, new_dst], dtype=torch.long)


def extract_suspicious_regions(
    data,
    scores: np.ndarray,
    k: int = 10,
    hops: int = 2,
) -> List[Region]:
    """Extract and merge neighborhoods around top-k suspicious nodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG graph with at least ``edge_index`` and ``x``.
    scores : np.ndarray
        1-D array of suspicion scores per node (length == data.num_nodes).
    k : int
        Number of most-suspicious seed nodes.
    hops : int
        BFS radius around each seed.

    Returns
    -------
    list[Region]
        Merged region descriptors sorted by max seed score descending.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else data.x.size(0)

    # Top-k seed indices (descending score)
    top_k = min(k, len(scores))
    seed_indices = np.argsort(scores)[::-1][:top_k].tolist()

    # Build adjacency
    adj = _build_adjacency(data.edge_index, num_nodes)

    # Collect neighborhoods
    neighborhoods: List[Tuple[int, Set[int]]] = []
    for seed in seed_indices:
        nh = _bfs_neighborhood(adj, seed, hops)
        neighborhoods.append((seed, nh))

    # Merge overlapping
    merged = _merge_overlapping(neighborhoods)

    # Build Region objects
    regions: List[Region] = []
    for idx, (seeds, node_set) in enumerate(merged):
        sub_ei = _restrict_edge_index(data.edge_index, node_set)
        seed_scores = [float(scores[s]) for s in seeds]
        region = Region(
            region_id=f"R{idx}",
            seed_indices=seeds,
            node_indices=sorted(node_set),
            seed_scores=seed_scores,
            subgraph_edge_index=sub_ei,
            num_nodes=len(node_set),
            num_edges=sub_ei.size(1),
        )
        regions.append(region)

    # Sort regions by max seed score descending
    regions.sort(key=lambda r: max(r.seed_scores), reverse=True)
    # Re-assign region_ids after sort
    for i, r in enumerate(regions):
        r.region_id = f"R{i}"

    return regions
