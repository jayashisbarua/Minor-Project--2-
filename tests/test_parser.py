"""Unit tests for Phase 1: Robust parsing & Cell Library."""

import os
import re
import glob
import random
import pytest
from pathlib import Path

from neuroguard.parsing.cell_library import classify_cell, build_library, find_dataset_netlists
from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.parsing.labels import parse_log, build_gate_labels


def test_cell_library_no_unmapped():
    """Verify zero unmapped cells across the entire dataset."""
    library = build_library("data", "results/cell_inventory.json")
    assert len(library) >= 110
    # Assert every cell belongs to one of the 12 canonical families
    canonical = {"AND", "AOI", "BUF", "DFF", "INV", "MUX", "NAND", "NOR", "OAI", "OR", "XNOR", "XOR"}
    for cell_name, info in library.items():
        assert info["function_family"] in canonical, f"Cell {cell_name} has invalid family {info['function_family']}"


def test_s13207_t400_parser_accuracy():
    """Verify s13207_T400 DFF count and PI count fixes.

    Legacy bug 1: dropped all 625 sdffs1 flip-flops due to escaped name regex.
    Legacy bug 2: ignored QN outputs, producing 568 spurious PI pseudo-nodes.
    """
    netlist_path = Path("data/TRIT-TS/s13207_T400/s13207_T400.v")
    assert netlist_path.exists(), "Test netlist s13207_T400.v not found"

    parser = NetlistParser()
    graph = parser.parse(netlist_path)

    # Count DFF cells
    sdff_count = sum(
        1 for _, d in graph.nodes(data=True)
        if d.get("type") == "GATE" and d.get("cell_type") == "sdffs1"
    )
    total_dff_count = sum(
        1 for _, d in graph.nodes(data=True)
        if d.get("type") == "GATE" and d.get("family") == "DFF"
    )
    # Exactly 625 sdffs1 instances (plus 4 trojan DFFs: dffs2 / dffles2 = 629 total)
    assert sdff_count == 625, f"Expected 625 sdffs1, got {sdff_count}"
    assert total_dff_count == 629, f"Expected 629 total DFFs, got {total_dff_count}"

    # Verify PI pseudo-nodes: should be exactly 65 (far fewer than 568)
    pi_count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "PI")
    assert pi_count == 65, f"Expected 65 PIs, got {pi_count} (legacy bug created 568)"
    assert pi_count < 100


def test_instance_count_matches_naive_grep_on_20_netlists():
    """Verify parsed instance count equals naive regex count on 20 randomly sampled netlists."""
    random.seed(42)
    all_netlists = find_dataset_netlists("data")
    assert len(all_netlists) >= 20
    sample = random.sample(all_netlists, 20)

    parser = NetlistParser()
    for netlist_path in sample:
        graph = parser.parse(netlist_path)
        parsed_gate_count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "GATE")

        # Naive count: regex for cell instantiation
        with open(netlist_path, "r", errors="ignore") as f:
            content = f.read()
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        # Matches statements with instance pattern
        naive_matches = re.findall(
            r"^\s*([a-zA-Z0-9_]+)\s+(\\\S+|\S+)\s*\(",
            content,
            flags=re.MULTILINE,
        )
        # Filter out keywords
        keywords = {"module", "input", "output", "wire", "reg", "assign", "endmodule", "always", "initial"}
        naive_gates = [m for m in naive_matches if m[0] not in keywords]

        assert parsed_gate_count == len(naive_gates), (
            f"Mismatch in {netlist_path}: parsed {parsed_gate_count} vs naive {len(naive_gates)}"
        )


def test_label_cross_check_sample():
    """Verify Trojan gate count from names matches gate count parsed from log.txt body."""
    tc_dirs = glob.glob("data/TRIT-TC/*")
    ts_dirs = [d for d in glob.glob("data/TRIT-TS/*") if Path(d).name != "original_designs"]
    all_dirs = tc_dirs + ts_dirs

    # Test on a representative subset of 50 netlists
    random.seed(42)
    sample_dirs = random.sample(all_dirs, min(50, len(all_dirs)))

    parser = NetlistParser()
    for d in sample_dirs:
        v_file = Path(d) / f"{Path(d).name}.v"
        log_file = Path(d) / "log.txt"
        if not v_file.exists() or not log_file.exists():
            continue

        log_data = parse_log(log_file)
        expected_trojan_count = len(log_data["body_instance_names"])

        graph = parser.parse(v_file, log_file)
        actual_trojan_count = sum(1 for _, data in graph.nodes(data=True) if data.get("is_trojan") == 1)

        assert actual_trojan_count == expected_trojan_count, (
            f"Trojan count mismatch in {d}: graph had {actual_trojan_count}, log had {expected_trojan_count}"
        )
