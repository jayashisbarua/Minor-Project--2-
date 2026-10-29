# NeuroGuard-X: Multi-View GNN Hardware Trojan Detection & Feedback System

NeuroGuard-X is a production-grade Hardware Trojan detection and mitigation framework for gate-level Verilog netlists. It combines multi-view Graph Neural Networks (GCN, GraphSAGE, GAT, and Transformer Fusion), bit-parallel logic simulation, zero-leakage context expansion, and LLM reasoning with closed-loop active learning feedback.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Gate-Level Verilog (.v)   |
                      +--------------+--------------+
                                     |
                [Robust Parsing & 12 Canonical Families]
                                     v
                      +-----------------------------+
                      |   Multi-View Graph (PyG)    |
                      |  - Structural (Directional) |
                      |  - Functional (Data/Control)|
                      |  - Temporal (Simulation)    |
                      +--------------+--------------+
                                     |
               [Multi-View GNN Ensembles (GCN/SAGE/GAT/Fusion)]
                                     v
                      +-----------------------------+
                      |   Suspicion Scores & Roles  |
                      +--------------+--------------+
                                     |
                   [Context Expansion: 2-hop BFS]
               [Zero-Leakage Synthetic Anonymization (G0, G1...)]
                                     v
                      +-----------------------------+
                      | LLM Reasoning & Consistency |
                      | (Ollama / Local / API Stub) |
                      +--------------+--------------+
                                     |
                 [Soft Target Distillation & InfoNCE Loss]
                                     v
                      +-----------------------------+
                      |  Feedback Retraining Loop   |
                      +-----------------------------+
```

---

## Directory Structure

```
.
├── configs/                  # Global system configuration (YAML)
├── data/                     # Raw netlists (TRIT-TC, TRIT-TS, TjFree)
├── cache/                    # Pre-processed PyTorch Geometric Data (.pt)
├── neuroguard/
│   ├── parsing/              # Cell library discovery, netlist AST parser, label parser
│   ├── graphs/               # Multi-view graph construction, bit-parallel simulator, features
│   ├── models/               # GCN, GraphSAGE, GAT, and Ensemble (Weighted & Transformer Fusion)
│   ├── training/             # Trainer, EarlyStopping, Logger, Calibration, Report generation
│   ├── eval/                 # LOFO splits, Trojan-type splits, Top-10 fixed metrics, Stats
│   ├── context/              # 2-hop neighborhood expansion, role heuristics, anonymization
│   ├── llm/                  # Backends (Stub, Ollama, API), prompts, A/B swap consistency
│   └── feedback/             # Soft-label distillation, TextProjector, InfoNCE contrastive loss
├── experiments/              # Phase-by-phase executable experiment scripts (p0..p10)
├── tests/                    # Comprehensive unit and integration test suite
├── results/                  # Generated metrics, cache statistics, logs, summaries
├── reports/                  # Phase acceptance reports (P0_report.md .. P10_report.md)
├── Makefile                  # Phased automation targets
└── requirements.txt          # Python dependencies
```

---

## Requirements & Environment

- **Python**: 3.10+ (tested on Python 3.10.22)
- **Hardware**: Compatible with macOS (Apple Silicon MPS hardware acceleration), Linux (CUDA), and CPU.
- **Dependencies**:
  - `torch >= 2.0.0`
  - `torch_geometric >= 2.4.0`
  - `numpy`, `scipy`, `scikit-learn`
  - `networkx`, `pandas`, `pyyaml`
  - `matplotlib`, `pytest`, `tqdm`

### Setup

```bash
# Clone and enter directory
cd "Minor Project 2"

# Activate your virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## Quick Start & Phased Pipeline Execution

The system is partitioned into automated phases matching the system specification. You can execute individual phases or run the entire suite using `make`:

```bash
# Execute Phase 0: Environment validation & acceleration check
make p0

# Execute Phase 1: Cell library audit (112 distinct cells across 12 canonical families)
make p1

# Execute Phase 2: Multi-view graph construction and cache generation
make p2

# Execute Phase 3: Bit-parallel simulation & toggle/entropy computation
make p3

# Execute Phase 4: Feature engineering (26 dimensions) & Random Forest ablation
make p4

# Execute Phase 5: Leave-One-Family-Out (LOFO) split generation & audit
make p5

# Execute Phase 6: GNN model training and ablation (GCN, SAGE, GAT, Ensembles)
make p6

# Execute Phase 7: Anonymized context extraction (2-hop neighborhoods, zero leakage)
make p7

# Execute Phase 8: LLM security analysis & A/B consistency protocol
make p8

# Execute Phase 9: Active learning feedback retraining & InfoNCE alignment
make p9

# Execute Phase 10: Full test suite regression
make p10
```

---

## Running Tests

Run the complete test suite across all modules:

```bash
make test
# or directly:
./.venv/bin/pytest tests/ -v
```

### Test Coverage Summary:
- `tests/test_parser.py`: Netlist parser accuracy, cell mappings, Trojan label cross-checks.
- `tests/test_graphs.py`: Multi-view node alignment, structural and functional views.
- `tests/test_simulation.py`: Analytical gate verification, c17 exhaustive truth table validation.
- `tests/test_features.py`: Node-name zero-leakage invariance, FeatureScaler fit isolation.
- `tests/test_splits.py`: LOFO strict design isolation, fixed Top-10 metrics, McNemar test.
- `tests/test_training.py`: Early stopping restore, test fold isolation, artifact generation.
- `tests/test_context.py`: Neighborhood extraction, Union-Find region merging, anonymization.
- `tests/test_llm.py`: Stub backend, markdown fence removal, A/B swap consistency.
- `tests/test_feedback.py`: Soft label generation, mask preservation, InfoNCE loss dynamics.

---

## Key Guarantees & Constraints

1. **Strict Zero-Leakage**: Feature extraction and context expansion never read, encode, or pass circuit net/instance/node names.
2. **Fixed Top-10 Metrics**: All precision, recall, and F1 at $k$ evaluate fixed $k=10$ (never scaled to true positive count).
3. **Leave-One-Family-Out (LOFO)**: Test family designs are strictly isolated from training and validation splits.
4. **Offline Resilience**: LLM modules default to deterministic `StubBackend` for local development without active external LLM endpoints.

# Minor-Project-new
# Minor-Project-new
