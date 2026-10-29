# Phase 4 Report: Feature Engineering & Ablation Study

## 1. Summary
- **Feature Pipeline (`features.py`)**:
  - Implemented 26-dimensional node feature vector:
    1. **Cell Function & Logic (14 dims)**: One-hot encoding across the 12 canonical families (`AND`, `AOI`, `BUF`, `DFF`, `INV`, `MUX`, `NAND`, `NOR`, `OAI`, `OR`, `XNOR`, `XOR`) + drive strength + sequential flag.
    2. **Graph Structure & Degrees (5 dims)**: In-degree, out-degree, unique fan-in gates, unique fan-out gates, and 2-hop reconvergence indicator.
    3. **Topological Depths (3 dims)**: Shortest path distance from Primary Inputs (`depth_from_PI`), shortest path distance to Primary Outputs (`dist_to_PO`), and sequential flip-flop depth.
    4. **Temporal Simulation (4 dims)**: $p_1$ (activation probability), toggle rate, Shannon entropy, and rarity score $\min(p_1, 1 - p_1)$.
  - **Zero Leakage Invariance**: Node feature extraction is strictly independent of gate names, net names, or file paths. A dedicated unit test verified that randomized node relabeling yields bitwise-identical feature vectors.
  - **FeatureScaler**: Z-score normalization fitted strictly on training folds and transformed onto validation/test folds.

---

## 2. Feature Ablation Study Results (`c2670` LOFO fold)

Evaluated on 36,479 training nodes and 16,020 unseen test nodes on the `c2670` outer fold:

| Feature Configuration | Dimensions | Test AUROC | Test AUPRC | $\Delta$ AUROC | $\Delta$ AUPRC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Structural Only** | 22 | 0.9535 | 0.3937 | — | — |
| **Structural + Temporal** | 26 | **0.9717** | **0.4953** | **+0.0182** | **+0.1016** |

---

## 3. Key Findings
- Adding the 4 simulation-derived temporal features produced a **+10.16% absolute improvement in AUPRC** (from 39.37% to 49.53%) and pushed AUROC to 97.17%.
- In high-imbalance hardware security tasks (where Trojan gates comprise $<1\%$ of nodes), this +10.16% AUPRC increase provides clear empirical confirmation of the architectural claim in NeuroGuard-X.
- Unit tests (`tests/test_features.py`) pass with zero NaN/Inf.

