# Phase 1 Report: Robust Parsing & Cell Library

## 1. Objectives & Achievements
- **Cell Library (`cell_library.py`)**:
  - Scanned the entire dataset of 914+ netlists.
  - Discovered and mapped **112 distinct cell types** with **zero unmapped cells**.
  - All cells cleanly categorized into 12 canonical function families: `AND`, `AOI`, `BUF`, `DFF`, `INV`, `MUX`, `NAND`, `NOR`, `OAI`, `OR`, `XNOR`, `XOR`.
  - Exported complete dictionary to `results/cell_inventory.json`.
- **Parser Robustness (`netlist_parser.py`)**:
  - Solved **Problem A (Dropped DFFs)**: Correctly handles escaped identifiers (e.g. `\DFF_637/Q_reg `). In `TRIT-TS/s13207_T400.v`, parsed DFF count is verified at 625 `sdffs1` (629 total including 4 Trojan DFFs).
  - Solved **Problem B (Ignored QN outputs)**: Correctly treats `.QN` as an output port, eliminating spurious Primary Inputs (PI dropped from 568 down to the true 65 primary inputs).
  - Solved **Problem C (Gate type collapse)**: Rich 12-class canonical library preserves exact cell logic rather than folding into generic `buf`.
  - Accurately captures continuous `assign` constant tie-offs (`1'b0`, `1'b1`) as dedicated constant driver nodes (`CONST_0`, `CONST_1`).
  - Unit tests verified on 20 randomly sampled netlists that parsed instance counts match naive line counts exactly.
- **Labels & Ground Truth (`labels.py`)**:
  - Parsed `log.txt` ground truth (activation conditions, payload effects, Trojan body).
  - Discovered that Trojan instances use prefixes `troj`, `trojan`, and `trig` (sequential trigger gates).
  - Performed cross-check across all 914 infected netlists: **0 mismatches** between netlist Trojan gate count and `log.txt` body count.
  - Multi-class role labels implemented: `{0: clean, 1: trigger, 2: body, 3: payload}`.

---

## 2. Dataset Statistics Table

The table below summarizes all 8 base circuit families (4 combinational ISCAS-85, 4 sequential ISCAS-89) across TRIT-TC and TRIT-TS:

| Base Design | Netlists | Avg Gates | Avg Edges | Total Trojan Gates | Avg Trojan Gates | Trojan Types |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **c2670** | 100 | 693.2 | 1,409.3 | 723 | 7.23 | combinational |
| **c3540** | 100 | 1,141.1 | 2,149.3 | 712 | 7.12 | combinational |
| **c5315** | 110 | 2,314.9 | 4,530.7 | 874 | 7.95 | combinational |
| **c6288** | 110 | 2,423.7 | 4,852.9 | 842 | 7.65 | combinational |
| **s13207** | 150 | 2,277.0 | 6,126.6 | 4,655 | 31.03 | combinational, counter (2–15 states), FSM (1–15 triggers) |
| **s1423** | 91 | 503.3 | 1,184.2 | 2,215 | 24.34 | combinational, counter (2–15 states), FSM (1–15 triggers) |
| **s15850** | 150 | 2,940.5 | 7,110.3 | 4,872 | 32.48 | combinational, counter (2–15 states), FSM (1–15 triggers) |
| **s35932** | 103 | 6,863.7 | 17,277.5 | 2,644 | 25.67 | combinational, counter (2–15 states), FSM (1–15 triggers) |
| **Total** | **914** | — | — | **17,537** | **19.19** | — |

---

## 3. Acceptance Verification
- `pytest tests/test_parser.py`: All 4 test suites passed in 4.00s.
- `s13207_T400.v`: 625 `sdffs1` verified, 65 PIs verified.
- Zero unmapped cells across all 112 cell types.
- Full statistics persisted in `results/p1_dataset_stats.json`.
- Phase 1 acceptance criteria are completely satisfied.

