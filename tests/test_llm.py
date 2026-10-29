"""Tests for Phase 8: LLM Integration and Consistency Protocols."""

import json
import pytest

from neuroguard.llm.backends import StubBackend, get_backend
from neuroguard.llm.prompts import build_trojan_analysis_prompt, parse_llm_response
from neuroguard.llm.consistency import ab_swap_test, compute_agreement, permute_context_gates


@pytest.fixture
def sample_context():
    return {
        "analysis_type": "hardware_trojan_context_expansion",
        "total_regions_flagged": 1,
        "total_seed_gates_analyzed": 2,
        "regions": [
            {
                "region_id": "R0",
                "num_gates": 3,
                "num_internal_connections": 2,
                "num_seed_gates": 2,
                "max_suspicion": 0.95,
                "mean_suspicion": 0.85,
                "gates": [
                    {
                        "id": "G0",
                        "is_seed": True,
                        "cell_family": "AND",
                        "fan_in": 4,
                        "fan_out": 1,
                        "gnn_suspicion_score": 0.95,
                        "role_hint": "possible_trigger",
                    },
                    {
                        "id": "G1",
                        "is_seed": True,
                        "cell_family": "XOR",
                        "fan_in": 2,
                        "fan_out": 0,
                        "gnn_suspicion_score": 0.92,
                        "role_hint": "possible_payload",
                    },
                    {
                        "id": "G2",
                        "is_seed": False,
                        "cell_family": "INV",
                        "fan_in": 1,
                        "fan_out": 1,
                        "gnn_suspicion_score": 0.12,
                        "role_hint": "logic",
                    },
                ],
                "internal_connections": [
                    {"from": "G0", "to": "G1"},
                    {"from": "G2", "to": "G0"},
                ],
            }
        ],
    }


def test_stub_backend_returns_valid_json():
    backend = StubBackend()
    raw = backend.query("Test prompt")
    parsed = parse_llm_response(raw)
    
    assert parsed["verdict"] == "trojan_detected"
    assert parsed["confidence_0_100"] == 75
    assert len(parsed["evidence"]) > 0
    assert "explanation" in parsed
    assert "fix" in parsed


def test_prompt_no_name_leakage(sample_context):
    prompt = build_trojan_analysis_prompt(sample_context)
    assert "G0" in prompt
    assert "G1" in prompt
    assert "hardware_trojan_context_expansion" in prompt


def test_parse_response_handles_markdown_fences():
    fenced = """Here is your analysis:
```json
{
  "verdict": "likely_clean",
  "confidence_0_100": 90,
  "evidence": ["No rare toggle gates found"],
  "explanation": "Circuit behaves normally",
  "fix": "No action needed"
}
```
Hope this helps!"""
    parsed = parse_llm_response(fenced)
    assert parsed["verdict"] == "likely_clean"
    assert parsed["confidence_0_100"] == 90
    assert parsed["evidence"] == ["No rare toggle gates found"]


def test_permute_context_gates(sample_context):
    permuted = permute_context_gates(sample_context, seed=42)
    assert len(permuted["regions"][0]["gates"]) == 3
    # IDs should still be G0, G1, G2
    ids = [g["id"] for g in permuted["regions"][0]["gates"]]
    assert set(ids) == {"G0", "G1", "G2"}


def test_ab_swap_consistency(sample_context):
    backend = StubBackend()
    report = ab_swap_test(backend, sample_context, n_trials=3)
    assert report["agreement_ratio"] == 1.0
    assert report["majority_verdict"] == "trojan_detected"
    assert len(report["responses"]) == 3

