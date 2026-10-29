"""Bit-parallel logic simulator for NeuroGuard-X.

Evaluates gate-level netlists with 64-parallel vector streams using uint64 bitwise logic.
Produces temporal features:
    - p1: Probability of logic 1
    - toggle_rate: Average transition rate between consecutive vectors
    - entropy: Binary Shannon entropy of p1
    - rare_score: min(p1, 1 - p1)
"""

import math
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import networkx as nx

from neuroguard.parsing.cell_library import classify_cell

# Mask for 64-bit unsigned integers
UINT64_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
UINT64_ZERO = np.uint64(0x0000000000000000)


def popcount64(val: np.ndarray) -> np.ndarray:
    """Vectorized population count for uint64 array."""
    # NumPy 2.0+ has bitwise_count; fallback to bitwise popcount if needed
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(val)
    # 64-bit Hamming weight algorithm
    v = val.astype(np.uint64)
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = v + (v >> np.uint64(8))
    v = v + (v >> np.uint64(16))
    v = v + (v >> np.uint64(32))
    return (v & np.uint64(0x7F)).astype(np.int64)


def binary_entropy(p: float) -> float:
    """Compute binary Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


class BitParallelSimulator:
    """Simulator for combinational and sequential gate-level netlists."""

    def __init__(self, num_cycles: int = 1000, seed: int = 42):
        """
        Parameters
        ----------
        num_cycles: Number of simulation steps (each evaluates 64 parallel vectors, total = num_cycles * 64).
        seed: Random seed for deterministic reproducibility.
        """
        self.num_cycles = num_cycles
        self.seed = seed
        self.total_vectors = num_cycles * 64

    def simulate(self, G: nx.MultiDiGraph) -> Dict[str, Dict[str, float]]:
        """Simulate graph G and return temporal stats per node."""
        rng = np.random.default_rng(self.seed)

        # Identify node roles
        pi_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "PI"]
        po_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "PO"]
        const_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "CONST"]
        dff_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("type") == "GATE" and d.get("family") == "DFF"
        ]
        comb_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("type") == "GATE" and d.get("family") != "DFF"
        ]

        # Break feedback cycles at DFF outputs for combinational topological sort
        comb_subgraph = nx.DiGraph()
        for u, v in G.edges():
            u_is_dff = G.nodes[u].get("family") == "DFF"
            if not u_is_dff:
                comb_subgraph.add_edge(u, v)

        # Topological sort of combinational path
        try:
            topo_order = list(nx.topological_sort(comb_subgraph))
            eval_order = [n for n in topo_order if n in G and G.nodes[n].get("type") == "GATE" and G.nodes[n].get("family") != "DFF"]
        except nx.NetworkXUnfeasible:
            # Fallback if feedback loop exists: iterative or approximate order
            eval_order = comb_nodes

        # Initialize node values
        curr_vals: Dict[str, np.uint64] = {}
        prev_vals: Dict[str, np.uint64] = {}

        for n in G.nodes():
            curr_vals[n] = UINT64_ZERO
            prev_vals[n] = UINT64_ZERO

        # Constants
        if "CONST_0" in G:
            curr_vals["CONST_0"] = UINT64_ZERO
        if "CONST_1" in G:
            curr_vals["CONST_1"] = UINT64_ONES

        # DFF state registers: latch DIN at cycle end
        dff_state: Dict[str, np.uint64] = {n: UINT64_ZERO for n in dff_nodes}

        # Accumulators
        ones_counts = {n: 0 for n in G.nodes()}
        toggle_counts = {n: 0 for n in G.nodes()}

        for cycle in range(self.num_cycles):
            # 1. Primary Inputs: generate 64 random bits
            for pi in pi_nodes:
                val = rng.integers(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
                curr_vals[pi] = np.uint64(val)

            # 2. DFF outputs reflect previous cycle's state
            for dff in dff_nodes:
                curr_vals[dff] = dff_state[dff]

            # 3. Evaluate combinational gates in topological order
            for gate in eval_order:
                val = self._eval_gate(G, gate, curr_vals)
                curr_vals[gate] = val

            # 4. Evaluate POs
            for po in po_nodes:
                preds = list(G.predecessors(po))
                if preds:
                    curr_vals[po] = curr_vals[preds[0]]

            # 5. Latch DFF next-state (scan_enable is tied to 0 in functional mode)
            for dff in dff_nodes:
                din_val = UINT64_ZERO
                for u, _, edge_data in G.in_edges(dff, data=True):
                    port = edge_data.get("port", "").upper()
                    # In functional mode, DIN drives the DFF
                    if port in ("DIN", "DIN1", "D"):
                        din_val = curr_vals[u]
                        break
                dff_state[dff] = din_val

            # 6. Accumulate ones and toggles
            for n in G.nodes():
                val = curr_vals[n]
                ones_counts[n] += int(popcount64(np.array([val]))[0])
                if cycle > 0:
                    flips = val ^ prev_vals[n]
                    toggle_counts[n] += int(popcount64(np.array([flips]))[0])
                prev_vals[n] = val

        # Compute summary metrics
        results: Dict[str, Dict[str, float]] = {}
        v_total = max(1, self.total_vectors)
        v_toggle = max(1, (self.num_cycles - 1) * 64)

        for n in G.nodes():
            p1 = float(ones_counts[n]) / v_total
            toggle_rate = float(toggle_counts[n]) / v_toggle
            ent = binary_entropy(p1)
            rare = min(p1, 1.0 - p1)

            results[n] = {
                "p1": p1,
                "toggle_rate": toggle_rate,
                "entropy": ent,
                "rare_score": rare,
            }

        return results

    def _eval_gate(self, G: nx.MultiDiGraph, gate: str, vals: Dict[str, np.uint64]) -> np.uint64:
        """Evaluate a single combinational gate using 64-bit bitwise logic."""
        node_data = G.nodes[gate]
        family = node_data.get("family", "")

        in_edges = G.in_edges(gate, data=True)
        # Collect inputs
        inputs: List[Tuple[str, np.uint64]] = []
        for u, _, ed in in_edges:
            inputs.append((ed.get("port", ""), vals.get(u, UINT64_ZERO)))

        if not inputs:
            return UINT64_ZERO

        raw_in = [v for _, v in inputs]

        if family == "INV":
            return ~raw_in[0]
        elif family == "BUF":
            return raw_in[0]
        elif family == "AND":
            res = UINT64_ONES
            for v in raw_in:
                res &= v
            return res
        elif family == "NAND":
            res = UINT64_ONES
            for v in raw_in:
                res &= v
            return ~res
        elif family == "OR":
            res = UINT64_ZERO
            for v in raw_in:
                res |= v
            return res
        elif family == "NOR":
            res = UINT64_ZERO
            for v in raw_in:
                res |= v
            return ~res
        elif family == "XOR":
            res = UINT64_ZERO
            for v in raw_in:
                res ^= v
            return res
        elif family == "XNOR":
            res = UINT64_ZERO
            for v in raw_in:
                res ^= v
            return ~res
        elif family == "MUX":
            # Select line and data lines
            sel = UINT64_ZERO
            d0, d1 = UINT64_ZERO, UINT64_ZERO
            for p, v in inputs:
                pu = p.upper()
                if "SEL" in pu or "S" in pu or "CLK" in pu:
                    sel = v
                elif "DIN1" in pu:
                    d0 = v
                elif "DIN2" in pu:
                    d1 = v
            # 2:1 MUX: (sel & d1) | (~sel & d0)
            mux_out = (sel & d1) | (~sel & d0)
            if node_data.get("cell_type", "").startswith("mxi"):
                return ~mux_out
            return mux_out
        elif family == "AOI":
            # AND-OR-INVERT: e.g. aoi21: ~((A & B) | C)
            # Default to NAND of ORs or general approximation
            or_term = UINT64_ZERO
            for v in raw_in:
                or_term |= v
            return ~or_term
        elif family == "OAI":
            # OR-AND-INVERT: e.g. oai21: ~((A | B) & C)
            and_term = UINT64_ONES
            for v in raw_in:
                and_term &= v
            return ~and_term

        # Default fallback
        return raw_in[0]

