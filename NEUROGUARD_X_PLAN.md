# NeuroGuard-X: Implementation Plan for Antigravity

Hand this file to the agent as the project brief. Work **one phase at a time**. Do not start a phase until the previous phase's acceptance checks pass. After each phase, write a short `reports/Pn_report.md` (what was built, numbers, anything surprising).

---

## 0. Agent rules (paste into workspace rules if your Antigravity version supports them)

0. Always work inside the project's virtual environment (`.venv`, created in P0). Activate it in every new terminal before running anything, and never install packages globally.
1. Python 3.10+, PyTorch + PyTorch Geometric, networkx, numpy, scipy, scikit-learn, tqdm, pytest. No Yosys/PyVerilog/DGL unless a phase says so (see P1 for why).
2. Every script takes `--seed` and is deterministic. Default seed 42. Call `model.eval()` before any inference (the friend's project lost days to a missing `.eval()`).
3. Never use instance names, net names, or file paths as model features or LLM prompt content. They contain `troj...`, `Trigger_en`, `tempn...` and leak the label. Names may be used **only** to build ground-truth labels, and only in `labels.py`.
4. Never derive anything from the number of true Trojan gates at inference time (the friend's `top_k_flagged` uses `1.5 x true_count`; do not copy that).
5. Splits are always by **family** or by **base design**, never by random netlist (see section 1.4).
6. Every reported number comes from a script in `experiments/` and is written to `results/*.json`. No hand-typed numbers in reports.
7. Do not modify `data/`. Treat `current_work/` as read-only reference; copy code you want to reuse into `neuroguard/` and fix it.
8. Every model that is trained (GCN, SAGE, GAT, ensemble, feedback retrain) goes through the shared `Trainer` described in **section 3A**: early stopping, per-epoch metric logging, best-checkpoint saving and an auto-generated training report. No ad-hoc training loops and no fixed epoch counts without early stopping.

---

## 1. What the data actually is (verified by inspecting the zip)

### 1.1 Layout

```
data/
  TRIT-TC/   580 Trojan-inserted netlists
             c2670, c3540 (100 each), c5315, c6288 (110 each)   -> combinational ISCAS-85
             s13207, s1423, s15850, s35932 (40 each)            -> sequential ISCAS-89
  TRIT-TS/   334 Trojan-inserted netlists (all sequential)
             s13207 (110), s1423 (51), s15850 (110), s35932 (63)
             original_designs/  clean s*scan.v + .spf
  TjFree/    339 folders of clean designs, very mixed
```

Each Trojan folder holds `<name>.v` (gate-level netlist) and `log.txt`.

### 1.2 `log.txt` is a free ground-truth source

It contains: Trojan type, number of Trojans (1 in 734 netlists, 2 in 180), the **payload signal(s)** ("Effect: alters ... n796"), the **trigger signals** ("Activation Condition ..."), and the **Trojan body** as Verilog. Trojan types: 580 combinational (all of TRIT-TC), 165 FSM (TRIT-TS), 169 counter (TRIT-TS). Use this to build three role labels per gate: `trigger`, `body`, `payload`. This directly supports Stage 3's "gate roles" and gives a much more precise localization target than a binary flag.

### 1.3 Netlist dialects

- **TRIT cells** use a vendor library with drive-strength suffixes: `nnd2s3`, `nor2s1`, `xor2s3`, `i1s3`, `hi1s1`, `nb1s1`, `and3s1`, `or5s3`, `xnr2s3`, `sdffs1` (scan flip-flop), `dffs2`, `dffles2`, about 78 distinct cells. Named ports: inputs `DIN/DIN1..`, outputs `Q` (and `QN` on flip-flops). Flip-flops have escaped names like `\DFF_637/Q_reg`.
- `assign gN = 1'b1;` constant tie-offs appear.
- **TjFree** has gate-level ISCAS files (`c17, c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552, s13207, ...`) in positional primitive style (`and AND2_4 (N405, N1, N3);`), but most of the 339 folders are behavioral RTL / SystemVerilog / UVM testbenches that are **not** gate-level. Only gate-level ones are usable as clean graphs without synthesis.

### 1.4 Problems found in the friend's code (fix these, do not inherit them)

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| A | **Parser drops every flip-flop.** The instance regex requires names starting with a letter, but flip-flops have escaped names (`\DFF_637/...`). | `s13207_T400.v`: 625 `sdffs1` in text, **0** parsed. | All sequential designs lose their state elements; graphs are wrong and disconnected. |
| B | **`QN` outputs ignored.** Only port `Q` is treated as an output. | `.QN(n1431)` becomes a fake primary input. | Spurious PI nodes (568 in that file), broken fan-out. |
| C | **Gate-type vocabulary doesn't match TRIT cells.** `GATE_TYPES` has 8 generic names; `nnd2s3` is not in it and folds into `buf`. | `features._type_index` fallback. | On TRIT data the "gate type" one-hot is nearly constant, so the GNN effectively sees only degree and depth. |
| D | **Oracle leakage in the LLM stage.** Number of flagged gates is `1.5 x` true Trojan count. | `run_paired_comparison.top_k_flagged`. | Inflated LLM and heuristic scores. |
| E | **Near-duplicate leakage in splits.** The ~100 variants per family share >99% of their graph. Random netlist splits put near-identical graphs in train and test. | dataset structure | Optimistic AUC. Use LOFO as the primary metric. |
| F | **Scale mismatch in clean samples.** `c2670` infected ~866 nodes vs clean 1642; the friend's synthetic clean subgraphs are a workaround. | `parse_netlist` on both | Graph-level classification can shortcut on size. |
| G | **Inconsistent reported results.** README says LLM 89.8% region-level; `results/summary.md` for seed 42 says 76.3% (103/135). | both files | Re-run before trusting any number. |

Also keep what the friend got right: name anonymization in the LLM prompt (`anonymize()`), the A/B swap protocol (both orientations must agree), McNemar tests, and the honest finding that **the LLM did not beat a one-line `argmax(gnn_score)` heuristic** (region-level 89.8% vs 92.7%, n.s.; gate-level 10.9% vs 19.7%, p=0.036). NeuroGuard-X must be evaluated against that baseline or it proves nothing.

---

## 2. Repository layout to create

```
neuroguard/
  parsing/      cell_library.py  netlist_parser.py  labels.py
  graphs/       build_views.py   features.py        simulate.py
  models/       gcn.py sage.py gat.py fusion.py ensemble.py
  context/      expand.py        roles.py           to_json.py
  llm/          backends.py      prompts.py         reason.py   consistency.py
  feedback/     soft_labels.py   contrastive.py
  eval/         metrics.py       splits.py          stats.py
  training/     trainer.py       early_stopping.py  logger.py   report.py   plots.py
experiments/    p1_*.py ... p9_*.py
tests/
configs/default.yaml
reports/  results/  cache/
```

Cache parsed graphs as `cache/<family>/<name>.pt` (PyG `Data`) so nothing is reparsed.

---

## 3. Phases

### P0. Environment setup  (do this first, before any code)

1. Create an isolated virtual environment in the project root and activate it:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - (or conda: `conda create -n neuroguard python=3.10 && conda activate neuroguard`)
2. Upgrade pip, then install PyTorch first, using the build that matches the machine (CPU-only or the correct CUDA version from pytorch.org), then PyTorch Geometric following its install guide for that exact torch version. Then install the rest: `numpy scipy scikit-learn networkx matplotlib pandas pyyaml tqdm pytest`, plus optional `tensorboard`.
3. Write `requirements.txt` with pinned versions, add `.venv/` to `.gitignore`, and save `results/environment.txt` (`pip freeze`).
4. Run a smoke test script `experiments/p0_check_env.py` that prints Python, torch and PyG versions, whether CUDA is available, and builds a tiny `Data` object and runs one `SAGEConv` forward pass.
5. Confirm `data/` is present and readable (count the folders in `TRIT-TC`, `TRIT-TS`, `TjFree`; expect 581, 335 and 339 entries).
6. Every later command in this plan must run **inside this environment**. If Antigravity opens a new terminal, it must activate `.venv` first.

Acceptance: `python experiments/p0_check_env.py` passes with no errors, and `pytest` runs (even with zero tests).

### P1. Robust parsing + cell library  (Module 1, part 1)

**Why not PyVerilog/Yosys here:** the inputs are already gate-level netlists in a fixed regular format. A strict custom parser is faster and more controllable. Add PyVerilog/Yosys only in P9 for RTL clean designs.

Tasks
1. `cell_library.py`: map every cell name to `(function_family, n_inputs, drive_strength, is_sequential, output_ports)`. Derive by regex from the name (`nnd`->NAND, `nor`->NOR, `and`, `or`, `xor`, `xnr`->XNOR, `i1/hi1/ib1/nb1`->INV/BUF, `sdff/dff/dffle`->DFF variants, etc.). Scan the **entire dataset** to list all distinct cell names first, write them to `results/cell_inventory.json`, and fail loudly on any unmapped cell. Also support the positional primitives `and, nand, or, nor, xor, xnor, not, buf`.
2. `netlist_parser.py`: tokenize by statement, support **escaped identifiers** (`\name` up to whitespace), named and positional ports, multiple outputs (`Q`, `QN`), `assign` tie-offs (constant nodes), multi-line instances. Produce a directed graph: node = gate instance, edge = driver -> load, with edge attributes (net id, load port name such as `DIN2`, `SSEL`, `CLK`). Keep PI/PO pseudo-nodes.
3. `labels.py` (the only place names may be used): per-gate `is_trojan` from instance-name prefix `troj`; then cross-check against `log.txt`: parse the Trojan body instances and payload/trigger signals, and emit role labels `{0:clean, 1:trigger, 2:body, 3:payload}` (payload = Trojan gate whose output feeds a non-Trojan gate, or the XOR driving the "Effect" signal).

Acceptance
- `pytest tests/test_parser.py`: for `TRIT-TS/s13207_T400.v` parsed DFF count == 625 and PI pseudo-nodes are far fewer than 568; instance count equals a naive count of `^\s*<cell>\s` lines for 20 randomly chosen netlists.
- Zero unmapped cells across all 914 infected + selected clean netlists.
- Label cross-check: for every netlist, Trojan gate count from names == gate count parsed from `log.txt` body (report mismatches, do not silently fix).
- Write `reports/P1_report.md` with dataset statistics table (nodes, edges, gates, Trojan gates per family).

### P2. Multi-view graph construction  (Module 1, part 2)

All three views share the **same node set** so fusion is node-aligned.

- **Structural view**: edges = physical connections (as in P1), made bidirectional for message passing but with a direction flag feature on each edge.
- **Functional view**: signal-flow / control dependencies. Add edges from flip-flop outputs to their fan-out as *state* edges, edges from `CLK`/`SSEL`/enable pins as *control* edges (typed), and collapse inverter/buffer chains. Output: `edge_index`, `edge_type`.
- **Temporal view**: edge weights or node attributes from simulation (P3). Build in P3, attach in P4.

Store a `torch_geometric.data.Data` with `x`, `edge_index_struct`, `edge_index_func`, `edge_type_func`, `y_node` (binary), `y_role` (4-class), `y_graph`, `family`, `trojan_type`, `base_design`, `node_names` (kept **only** in metadata, never in `x`).

Acceptance: all 914 netlists convert; `data.validate()`; node alignment test (same `num_nodes` for every view).

### P3. Logic simulation for temporal features  (Module 2, simulation part)

No testbenches or traces are shipped, so build a simulator.

- Python **bit-parallel** simulator with numpy `uint64` words (64 vectors per word), evaluating gates in topological order using the cell function table from P1. Flip-flops: treat `sdffs1` with scan-enable tied to 0 (functional mode), update state each cycle. Constants from `assign`.
- Stimulus: random primary inputs, many cycles for sequential designs (>=10^4 cycles x 64 parallel streams; make it configurable). Seed everything.
- Per gate output record: `p1` (probability of logic 1), `toggle_rate`, `entropy` (binary entropy of `p1`), and `rare_score = min(p1, 1-p1)`.

Acceptance
- Unit tests on tiny hand-built circuits (AND/OR/XOR/DFF) with known exact values.
- Cross-check against a reference: for ISCAS `c17` compare against exhaustive enumeration.
- Runtime budget: report seconds for the largest netlist (`s13207` clean, 8.5k gates). Cache results.
- Sanity result to report: distribution of `p1`/`rare_score` for Trojan trigger gates vs all other gates (this is the key evidence the temporal view is useful; if there's no separation, say so).

### P4. Feature engineering  (Module 2)

Node features, **computed without names**:
1. Cell function family one-hot (about 12 classes) + drive strength + sequential flag. This replaces the broken `GATE_TYPES`.
2. `in_degree`, `out_degree`, fan-in/fan-out (gate-level, not net-level), reconvergence indicator.
3. `depth_from_PI`, `dist_to_PO`, and for sequential designs the **sequential depth** (flip-flops crossed).
4. Temporal: `p1`, `toggle_rate`, `entropy`, `rare_score` (from P3).
5. Optional (flag in config, default **off**): 1-hop / 2-hop neighbourhood type histograms. The friend's ablation showed these hurt a SAGE model; keep them behind an ablation switch only.
6. Optional (default **off**): CodeBERT/LLM contextual embedding of the gate's Verilog snippet. It is listed in NeuroGuard-X Stage 1. Because TRIT netlists are gate-level, snippets carry little semantics. Implement but evaluate before enabling.

Normalization: fit scalers on the training fold only.

Acceptance: feature matrix has no NaN/inf; a test asserts no feature column is derived from names; `experiments/p4_feature_ablation.py` prints AUC for {structural only, +temporal}.

### P5. Splits and metrics (build this before models)  (evaluation foundation)

- `splits.py`:
  - **LOFO** (leave one family out): 8 folds. Primary. Group by **base design name** (`c2670, c3540, c5315, c6288, s13207, s1423, s15850, s35932`) and put a design's netlists from **both** TRIT-TC and TRIT-TS, plus its clean TjFree version, in the same fold. The `s*` designs appear in both datasets, so splitting by dataset folder would leak the same base circuit into train and test.
  - **Leave-one-Trojan-type-out**: train on combinational (TC), test on FSM/counter (TS), and the reverse where meaningful.
  - Seeds {7, 42, 123}; report mean +/- std.
- `metrics.py`: node-level AUROC and AUPRC (the positives are rare, ~0.1% to 1% of nodes, so AUPRC matters more than AUROC), top-k precision/recall/F1 where **k is a fixed constant (10) or a fixed fraction, never the true count**, per-family breakdown, graph-level AUROC.
- `stats.py`: McNemar and bootstrap CIs.

Acceptance: tests that no netlist/base design appears on both sides of any split; a report with class balance per fold.

### P6. Multi-view GNN ensemble  (Module 3)

1. Three encoders, each takes node features + its edge set: `GCN` on the structural view (global topology), `GraphSAGE` on the structural view (inductive), `GAT` on the functional view (neighbor importance, uses `edge_type`). Two layers, hidden 64, dropout 0.2 as the starting point.
2. `fusion.py`: implement **both** (a) learned softmax-weighted voting over per-model logits and (b) a small transformer-fusion layer over the three node embeddings. Config switch.
3. Heads: per-node suspicion score + graph-level Trojan probability (readout = attention or max pooling over node scores). Multi-task loss: BCE with `pos_weight` for nodes (or focal loss), optional 4-class role loss, graph-level BCE.
4. Training: use the shared `Trainer` from **section 3A** (early stopping on validation AUPRC, per-epoch metric logging, best-checkpoint restore, auto-generated training report and curves).

**Mandatory ablations** (the whole point of an ensemble claim): single SAGE (the friend's model) vs each single model vs ensemble-voting vs ensemble-transformer, on the same LOFO folds and seeds. Report honestly even if the ensemble does not win.

Acceptance: ensemble runs on CPU in reasonable time, reproducible to within 1e-4 across two runs with the same seed on CPU, every run has the section 3A artifacts (metrics.csv, curves, run_summary.json, training_report.md), and `reports/P6_report.md` contains the ablation table plus `reports/training_results.md`.

### 3A. Training protocol, early stopping and result documentation (applies to P6 and P9)

Build this as `neuroguard/training/` **before** training any model in P6, and reuse it everywhere.

**A. Validation set (needed for early stopping)**
- For each outer split (a LOFO fold or Trojan-type split from P5), carve a validation set out of the **training families only**, by base design / family (e.g. hold out one training family, or 15% of variants chosen by base design). The test fold is never used for early stopping, threshold choice, or hyperparameter tuning.
- Save the exact train/val/test netlist lists to `results/train/<run_id>/split.json`.

**B. Early stopping (`early_stopping.py`)**
- Monitor: validation **AUPRC** (node-level), mode `max`. Secondary tie-break: validation loss.
- Config (all in `configs/default.yaml`): `max_epochs: 200`, `patience: 20`, `min_delta: 1e-4`, `warmup_epochs: 5` (no stopping before this), `restore_best: true`.
- Also use `ReduceLROnPlateau` on validation loss (factor 0.5, patience about patience/3) and gradient clipping (`max_norm: 1.0`).
- On stop, reload the best-epoch weights; the final test evaluation uses the **best** checkpoint, not the last.
- Record in the run summary: `best_epoch`, `stopped_epoch`, `stopped_early` (true/false), `best_val_auprc`.
- Run one **fixed-epoch baseline** (e.g. 30 epochs, no early stopping) per model on the same fold for comparison, and report both.

**C. Metrics logged every epoch (`logger.py`)**
- Train: total loss and each loss component (node BCE/focal, role loss, graph loss, and later soft-label / contrastive terms), learning rate, gradient norm, epoch time.
- Validation: loss, node AUROC, **AUPRC**, top-k F1 (fixed k), precision, recall, F1 at the validation-chosen threshold, graph-level AUROC, per-family AUPRC.
- Write to `results/train/<run_id>/metrics.csv` (one row per epoch) and `metrics.jsonl`; optional TensorBoard if installed.

**D. Threshold and calibration**
- Choose the decision threshold on the **validation** set (maximize F1 or fix a target recall), store it with the checkpoint, and apply it unchanged to the test fold.
- Report the Brier score and ECE of the node probabilities on the test fold.

**E. Checkpoints (`best.pt`, `last.pt`)**
- Each contains: model state dict, config, seed, epoch, validation metrics, the feature scaler (fit on training data only), the cell-function vocabulary, the selected threshold, git commit hash (if any), and library versions.
- Inference code must load everything from the checkpoint, so results can be reproduced without retraining.

**F. Artifacts generated automatically per run (`report.py`, `plots.py`)** in `results/train/<run_id>/`:
- `curves_loss.png` (train vs val loss, with best epoch marked)
- `curves_metrics.png` (val AUPRC / AUROC / F1 over epochs, early-stop point marked)
- `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png` (test fold at the saved threshold)
- `per_family_metrics.csv` (test fold)
- `run_summary.json` (best/stopped epoch, wall time, parameters count, final test metrics, seed)
- `training_report.md`: a readable summary with those figures embedded, the config used, and a 3 to 5 line interpretation (is it overfitting? did early stopping trigger? how large is the train/val gap?).

**G. Multi-seed aggregation (`experiments/aggregate_training.py`)**
- Collect all runs into one table: model x fold x seed with test AUROC, AUPRC, top-k F1, graph AUROC, best epoch, stopped epoch. Report mean +/- std, and write `reports/training_results.md` with the final comparison across GCN, SAGE, GAT, ensemble-voting and ensemble-transformer (this is the table the P6 ablation uses).

**H. Requirements**
- Packages: `torch`, `torch_geometric`, `scikit-learn`, `matplotlib`, `pandas`, `pyyaml`, `tqdm`, `pytest`; optional `tensorboard`. Write `requirements.txt` and `results/environment.txt` (`pip freeze`) for every run.
- Determinism: seed Python, numpy and torch; use CPU for the reported numbers (or document GPU nondeterminism); `model.eval()` and `torch.no_grad()` for all validation and test passes.
- Class imbalance: node positives are very rare, so use `pos_weight` or focal loss, and select by AUPRC rather than accuracy. Never report accuracy alone.
- Training must be resumable: periodic `last.pt` saves and a `--resume` flag.

**Acceptance for 3A**
- `pytest tests/test_training.py`: (1) early stopping triggers on a synthetic plateauing metric at the expected epoch; (2) best weights are restored; (3) the test fold is never read during training (assert on dataloader indices); (4) two runs with the same seed and CPU produce identical `metrics.csv` within 1e-4.
- A smoke run (2 epochs on a tiny subset) produces every artifact listed in F.
- The first real run's `training_report.md` is reviewed before proceeding to the ablations.

### P7. Context expansion + semantic encoding  (Module 4 / Stage 3)

- `expand.py`: take top-k nodes (k configurable, default 10; also an adaptive-radius option). Take the 2-hop neighborhood on the structural graph; merge overlapping regions into connected suspicious regions.
- `roles.py`: heuristic gate-role hints for the LLM (not ground truth): "converges many low-probability signals" (trigger-like, high fan-in tree of rare nodes), "XOR/MUX reconnecting to a PO path" (payload-like), "state-holding" (FSM/counter-like). Also signal path to the nearest PO, activation rarity.
- `to_json.py`: **anonymized** output (`G1, G2, w1...`) per region: cell family, fan-in/out, `p1`, `toggle_rate`, `dist_to_output`, neighbor list with families, the GNN suspicion score, and a compact netlist snippet in generic form. Keep the schema from Implementation.pdf, extended with `roles` and `snippet`.

Acceptance: a test greps every generated JSON/prompt for the strings `troj`, `Trigger`, `tempn`, and original net names and requires zero hits; region size stats are reported.

### P8. LLM reasoning + bias/consistency control  (Modules 5 and 6)

1. `backends.py`: pluggable. `OllamaBackend` (default `qwen3:14b`, as in the friend's runs), plus an API backend (Anthropic/OpenAI) selected by env var. A second model for cross-model checks (e.g. a different local model). Fixed seed, `temperature` configurable, JSON-mode output with strict parsing and retries.
2. `prompts.py` (structured output): `{verdict: trojan|benign|uncertain, confidence_0_100, evidence: [...], explanation, fix}`. The fix field is only evaluated qualitatively.
3. Two evaluation tasks (reuse the friend's protocol so numbers are comparable):
   - **Paired region task**: infected region vs same-size clean region from the family's clean baseline, A/B swapped; counts as correct only if both orientations agree with truth.
   - **Single-region verdict task** (matches Implementation.pdf Module 5): one region, verdict + confidence. Include clean regions (hard negatives) so the false-positive rate is measurable.
4. `consistency.py` (Module 6): orientation A/B swap; cross-model agreement (two LLMs) and cross-temperature agreement; confidence score = agreement across runs combined with self-reported confidence. Produce the three categories from NeuroGuard-X Stage 5. **Note:** "correct vs incorrect" requires ground truth, so these categories exist only at evaluation time; at deployment you only have {consistent -> accept, inconsistent -> flag for review}.
5. Baselines, mandatory and run through the same harness: (a) `argmax(gnn_score)` region heuristic, (b) GNN thresholding with no LLM, (c) LLM with no consistency control, (d) full NeuroGuard-X. Compare with McNemar (paired) and bootstrap CIs.
6. **Calibration** (the friend's open thread): ECE and a reliability diagram for `confidence_0_100`, and a selective-prediction curve (accuracy vs coverage when abstaining on low confidence). This is the most likely place the LLM adds real value.

Acceptance: harness is resumable (caches every LLM call by hash of prompt+model+seed so reruns are free); a dry-run mode with a stub backend exists for tests; baselines reproduce the friend's heuristic number within noise when given the same regions (else report the discrepancy from problem G/D).

### P9. Feedback loop  (Module 7 / Stage 6)

Implement the simplest version that can be tested cleanly, then extend.

1. **Soft labels**: from LLM outputs that pass consistency, build per-node soft targets `s = confidence x verdict` for the region's nodes. Retrain the GNN with `L = L_hard + lambda x L_soft`.
   - Leakage rule: soft labels may only be produced for nodes in the **training** folds' unlabeled pool or for a held-out pool; the test fold is never touched. Compare three arms on the same test fold: no feedback, feedback with ground-truth-verified LLM labels (upper bound), feedback with consistency-only filtering (the realistic deployment case, since you will not have ground truth).
2. **Contrastive alignment**: embed the LLM explanation text with a sentence encoder; add an InfoNCE loss pulling the GNN region embedding toward the embedding of its explanation, with other regions as negatives. Ablate on/off.
3. Retrain through the same `Trainer` (section 3A) with the extra loss terms logged separately; report whether GNN-only performance after feedback improves on LOFO AUPRC, and whether LLM calls needed to reach a target accuracy drop (the "reduces reliance on the LLM" claim).
4. (Optional) Clean-design expansion: synthesize a few RTL clean designs from `TjFree` with Yosys + `abc` to a gate library to add more clean negatives. This is where PyVerilog/Yosys finally appear. Skip unless time permits.

Acceptance: three-arm table with seeds and CIs. If feedback does not help, report that.

### P10. Packaging

- `run_all.sh` / `Makefile` targets per phase; `configs/default.yaml`; `README.md` with exact commands and expected runtimes.
- Final `reports/FINAL.md`: training summary (curves, early-stopping epochs, multi-seed table from `reports/training_results.md`), dataset audit (section 1.4 table with before/after), ablation tables, LLM-vs-baseline comparison, calibration, feedback results, limitations.

---

## 4. Claims to be careful about

The NeuroGuard-X comparison table asserts advantages over the friend's pipeline (bias control, scalability, adaptivity, feature depth, integration). Treat each as a **hypothesis** and fill the table with measured results from P6 to P9. Specifically:

- "Multi-view ensemble beats single GNN": only if the P6 ablation shows it.
- "Temporal features help": only if the P3/P4 separation result shows it.
- "Cross-model consistency helps": only if the P8 baselines show it beyond the swap protocol alone.
- "LLM adds value": the friend's data says it did not beat the heuristic; the credible upside is calibration/abstention and explanations, so measure those.

## 5. Suggested order if time is short

P1 -> P5 -> P4 (structural only) -> P6 (SAGE vs ensemble) -> P3 -> P7 -> P8 (paired task + baselines + calibration) -> P9 (soft labels only). Fixing problems A to C alone will likely change the GNN results materially, so re-run the friend's baseline on the fixed parser first and use that as your new reference number.