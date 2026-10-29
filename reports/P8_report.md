# Phase 8: LLM Integration & Consistency Report

**Date**: 2026-10-29  
**Module**: `neuroguard.llm` (`backends.py`, `prompts.py`, `consistency.py`)

## 1. Summary of Implementation

Phase 8 integrates Large Language Model reasoning into the Trojan detection loop:
- **Pluggable Backends**:
  - `StubBackend`: Fully offline mock returning structured verdicts for deterministic testing.
  - `OllamaBackend`: Local deployment running `qwen3:14b` or similar via Ollama REST API (`/api/generate`) with automatic retry and timeouts.
  - `APIBackend`: Standard OpenAI-compatible API interface configured via environment variables.
- **Strictly Formatted Prompts**:
  - Encapsulates anonymized circuit topology, rare-activation statistics, and heuristic role hints.
  - Requests structured JSON schema: `verdict`, `confidence_0_100`, `evidence`, `explanation`, `fix`.
  - Zero-leakage sanity check verifies that no netlist identifiers or raw instance names are ever passed to the LLM.
- **Robust Response Parser**:
  - Strips markdown code blocks and pre/postamble natural language text.
  - Validates all required JSON fields and verdict enums (`trojan_detected`, `likely_clean`, `inconclusive`).
- **A/B Swap Consistency Protocol**:
  - Randomly permutes gate identifiers (`G0`, `G1`, ...) and topological connection listings across multiple trials to measure verdict invariance.
  - Computes majority agreement ratio and confidence distribution.

## 2. Experimental Verification

Running `experiments/p8_llm_demo.py`:
- Backend: `StubBackend` (offline security analyst simulation)
- Prompt size: ~60 KB of anonymized circuit context from `c2670_T000`
- A/B Swap Results:
  - Trials: 3 independent permutations
  - Majority verdict: `trojan_detected`
  - Agreement ratio: 100.0%
  - Mean confidence: 75.0 / 100
- Unit tests: 5/5 passing in `tests/test_llm.py`

## 3. Acceptance Criteria Checklist

- [x] Pluggable backend architecture (`StubBackend`, `OllamaBackend`, `APIBackend`)
- [x] JSON-only output prompt formulation with zero-leakage enforcement
- [x] Resilient markdown/JSON parser with schema validation
- [x] A/B gate permutation consistency checking (`ab_swap_test`)
- [x] Full unit test suite (`tests/test_llm.py` - 5/5 passed)
- [x] Demonstration run artifact (`results/p8_llm_demo.json`)

