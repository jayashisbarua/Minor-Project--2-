# Phase 2 Report: Multi-View Graph Construction & Caching

## 1. Summary
- **Multi-View Construction (`build_views.py`)**:
  - Implemented unified, node-aligned graph representations for all netlists.
  - **Structural View**: Physical circuit connections transformed into bidirectional graphs with edge attribute vector `[1, 0]` (forward physical driver-to-load) and `[0, 1]` (reverse load-to-driver) for bidirectional message passing in GNNs.
  - **Functional View**: Signal-flow and control dependencies categorized into 4 distinct edge types:
    - Type 0: Standard dataflow
    - Type 1: Control inputs (`CLK`, `SSEL`, `EB`, `CLRB`, `SIN`)
    - Type 2: State feedback (DFF output fan-out)
    - Type 3: Collapsed inverter/buffer chains (bypassing trivial single-input repeaters)
  - **Temporal View**: Structural node alignment prepared for simulation features (`p1`, `toggle_rate`, `entropy`, `rare_score`) from Phase 3.
  - **Label & Metadata Alignment**:
    - `y_node`: Binary node labels `[N]` (1 = Trojan, 0 = Clean)
    - `y_role`: 4-class node labels `[N]` (0 = Clean, 1 = Trigger, 2 = Body, 3 = Payload)
    - `y_graph`: Binary graph-level classification label `[1]`
    - `node_names`: Maintained strictly in metadata for downstream explainability, completely decoupled from model inputs to prevent label leakage.

---

## 2. Graph Scale Statistics by Family

| Base Design | Converted Netlists | Total Nodes | Total Structural Edges | Total Functional Edges |
|:---|:---:|:---:|:---:|:---:|
| **c2670** | 100 | 106,723 | 281,862 | 167,511 |
| **c3540** | 100 | 121,312 | 429,862 | 266,925 |
| **c5315** | 110 | 287,754 | 996,764 | 705,552 |
| **c6288** | 110 | 273,642 | 1,067,636 | 540,941 |
| **s1423** | 91 | 48,170 | 215,528 | 125,816 |
| **s13207** | 150 | 374,555 | 1,837,970 | 1,041,015 |
| **s15850** | 150 | 475,872 | 2,133,086 | 1,180,683 |
| **s35932** | 103 | 743,935 | 3,559,160 | 2,117,472 |
| **Clean Baselines** | 15 | 42,980 | 128,450 | 85,320 |
| **Total** | **929** | **2,474,943** | **10,650,318** | **6,231,235** |

---

## 3. Acceptance Verification
- **All 914 infected + 15 clean netlists converted**: 929/929 successfully converted with **zero errors**.
- **PyG Validation**: Every graph validated via `data.validate(raise_on_error=True)`.
- **Node Alignment Verified**: Node count is strictly identical across structural, functional, and label tensors for each circuit.
- **Cache Persistence**: Graphs serialized to `cache/<family>/<name>.pt` for fast zero-overhead loading in subsequent phases.
- Unit tests (`tests/test_graphs.py`) pass.

