# Phase 7: Context Expansion & Anonymization Report

**Date**: 2026-10-29  
**Module**: `neuroguard.context` (`expand.py`, `roles.py`, `to_json.py`)

## 1. Summary of Implementation

Phase 7 implements the context expansion layer that bridges raw GNN suspicion logits to high-level reasoning modules (e.g. LLMs).
- **2-hop BFS Neighborhood Extraction**: Identifies the top-k suspicious nodes and extracts a 2-hop structural sub-graph surrounding each seed.
- **Neighborhood Union-Find Merging**: Disjoint or overlapping sub-graphs are automatically merged into unified connected components (`Region`).
- **Heuristic Role Hints**: Categorizes gates based purely on structural one-hot encodings and topological connectivity:
  - High fan-in AND/OR gates $\to$ `possible_trigger`
  - Floating XOR/XNOR gates $\to$ `possible_payload`
  - DFF elements $\to$ `state_element`
  - Standard gates $\to$ `logic`
- **Zero-Leakage Anonymization**: All node identifiers, instance names, and net names are converted to local synthetic gate IDs (`G0`, `G1`, ...). Strict assertion verifies that no identifier or substring from the Verilog netlist leaks into the context dictionary.

## 2. Experimental Verification

Running `experiments/p7_context_demo.py` on `c2670_T000`:
- Graph: 1065 nodes, 2810 edges
- Input: Top-10 suspicious nodes
- Merged Output: 4 regions (`R0` through `R3`)
  - Primary Region `R0`: 78 gates, 174 internal connections, max suspicion 1.0000
  - Captures trigger and payload circuitry within 2 hops.
- Verified Zero-Leakage: Automated test suite checks that none of the 1,065 original Verilog instance names exist anywhere in serialized JSON output.

## 3. Acceptance Criteria Checklist

- [x] Neighborhood extraction within 2 hops (`extract_suspicious_regions`)
- [x] Overlapping region consolidation via Union-Find (`_merge_overlapping`)
- [x] Heuristic role labeling without name leakage (`assign_role_hints`)
- [x] Strict synthetic node renaming (`G0`, `G1`, ...) with leak verification assertions
- [x] Unit test suite passing (`tests/test_context.py` - 4/4 passed)
- [x] Context generation artifact saved (`results/p7_context_demo.json`)

