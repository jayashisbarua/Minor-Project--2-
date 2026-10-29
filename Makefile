# NeuroGuard-X Makefile
# Controls execution across all phases (Phase 0 to Phase 10)

.PHONY: all p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 test clean

PYTHON = ./.venv/bin/python
PYTEST = ./.venv/bin/pytest

all: p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 test

# Phase 0: Environment sanity check
p0:
	@echo "=== Running Phase 0: Environment Check ==="
	$(PYTHON) experiments/p0_check_env.py

# Phase 1: Cell library & dataset statistics
p1:
	@echo "=== Running Phase 1: Cell Library & Parsing ==="
	$(PYTHON) experiments/p1_dataset_stats.py

# Phase 2: Multi-view graph construction & caching
p2:
	@echo "=== Running Phase 2: Multi-View Graph Construction ==="
	$(PYTHON) experiments/p2_build_graphs.py

# Phase 3: Logic simulation & toggle statistics
p3:
	@echo "=== Running Phase 3: Bit-Parallel Logic Simulation ==="
	$(PYTHON) experiments/p3_simulation.py

# Phase 4: Feature engineering & ablation
p4:
	@echo "=== Running Phase 4: Feature Engineering ==="
	$(PYTHON) experiments/p4_feature_ablation.py

# Phase 5: LOFO evaluation splits & statistics
p5:
	@echo "=== Running Phase 5: Split Generation & Audit ==="
	$(PYTHON) experiments/p5_split_stats.py

# Phase 6: GNN model ablations & ensemble training
p6:
	@echo "=== Running Phase 6: GNN Ablations & Ensemble ==="
	$(PYTHON) experiments/p6_ablations.py --models gcn sage gat ensemble_voting ensemble_transformer

# Phase 7: Context expansion & anonymization
p7:
	@echo "=== Running Phase 7: Context Expansion & Anonymization ==="
	$(PYTHON) experiments/p7_context_demo.py

# Phase 8: LLM integration & consistency protocols
p8:
	@echo "=== Running Phase 8: LLM Integration & Consistency ==="
	$(PYTHON) experiments/p8_llm_demo.py

# Phase 9: Active feedback loop & retraining
p9:
	@echo "=== Running Phase 9: Feedback Loop & Distillation ==="
	$(PYTHON) experiments/p9_feedback_demo.py

# Phase 10: Packaging & full pipeline regression
p10:
	@echo "=== Running Phase 10: Full Test Suite Regression ==="
	$(PYTEST) tests/ -v

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf __pycache__ tests/__pycache__ neuroguard/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
