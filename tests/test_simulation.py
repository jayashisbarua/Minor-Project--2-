"""Unit tests for Phase 3: Bit-parallel logic simulation."""

import pytest
import networkx as nx
from pathlib import Path
from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.graphs.simulate import BitParallelSimulator


def test_handcrafted_gates():
    """Verify exact values on hand-built AND, OR, XOR circuits."""
    G = nx.MultiDiGraph()
    # 2 Primary Inputs: A, B
    G.add_node("PI_A", type="PI", cell_type="PI", family="PI")
    G.add_node("PI_B", type="PI", cell_type="PI", family="PI")

    # Gate G_AND: A & B (expected p1 = 0.25)
    G.add_node("G_AND", type="GATE", cell_type="and2s1", family="AND")
    G.add_edge("PI_A", "G_AND", port="DIN1")
    G.add_edge("PI_B", "G_AND", port="DIN2")

    # Gate G_OR: A | B (expected p1 = 0.75)
    G.add_node("G_OR", type="GATE", cell_type="or2s1", family="OR")
    G.add_edge("PI_A", "G_OR", port="DIN1")
    G.add_edge("PI_B", "G_OR", port="DIN2")

    # Gate G_XOR: A ^ B (expected p1 = 0.50)
    G.add_node("G_XOR", type="GATE", cell_type="xor2s1", family="XOR")
    G.add_edge("PI_A", "G_XOR", port="DIN1")
    G.add_edge("PI_B", "G_XOR", port="DIN2")

    sim = BitParallelSimulator(num_cycles=2000, seed=42)
    stats = sim.simulate(G)

    assert abs(stats["G_AND"]["p1"] - 0.25) < 0.015, f"Expected 0.25, got {stats['G_AND']['p1']}"
    assert abs(stats["G_OR"]["p1"] - 0.75) < 0.015, f"Expected 0.75, got {stats['G_OR']['p1']}"
    assert abs(stats["G_XOR"]["p1"] - 0.50) < 0.015, f"Expected 0.50, got {stats['G_XOR']['p1']}"


def test_c17_exhaustive_enumeration():
    """Cross-check c17 simulation against exhaustive analytical enumeration."""
    parser = NetlistParser()
    c17_path = Path("data/TjFree/c17/c17.v")
    assert c17_path.exists()
    G = parser.parse(c17_path)

    sim = BitParallelSimulator(num_cycles=2000, seed=42)
    stats = sim.simulate(G)

    # Exhaustive analytical probabilities for all 32 combinations:
    exact = {
        "NAND2_1": 0.7500,
        "NAND2_2": 0.7500,
        "NAND2_3": 0.6250,
        "NAND2_4": 0.6250,
        "NAND2_5": 0.5625,
        "NAND2_6": 0.5625,
    }
    for gate, p_exact in exact.items():
        p_sim = stats[gate]["p1"]
        diff = abs(p_sim - p_exact)
        assert diff < 0.015, f"Discrepancy for {gate}: sim {p_sim:.4f} vs exact {p_exact:.4f}"

