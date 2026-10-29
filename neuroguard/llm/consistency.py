"""A/B consistency and reliability protocols for LLM Trojan verdicts."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional
import numpy as np

from neuroguard.llm.backends import LLMBackend
from neuroguard.llm.prompts import build_trojan_analysis_prompt, parse_llm_response


def compute_agreement(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate majority verdict and agreement statistics across multiple runs.

    Parameters
    ----------
    responses : list[dict]
        Parsed response dicts from ``parse_llm_response()``.

    Returns
    -------
    dict
        Dictionary with majority_verdict, agreement_ratio, confidence_mean, confidence_std.
    """
    if not responses:
        return {
            "majority_verdict": "inconclusive",
            "agreement_ratio": 0.0,
            "confidence_mean": 0.0,
            "confidence_std": 0.0,
            "num_trials": 0,
        }

    verdicts = [r.get("verdict", "inconclusive") for r in responses]
    confs = [float(r.get("confidence_0_100", 0)) for r in responses]

    # Count occurrences
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    majority_verdict, max_count = max(counts.items(), key=lambda item: item[1])
    agreement_ratio = float(max_count / len(verdicts))

    return {
        "majority_verdict": majority_verdict,
        "agreement_ratio": round(agreement_ratio, 4),
        "confidence_mean": round(float(np.mean(confs)), 2),
        "confidence_std": round(float(np.std(confs)), 2),
        "num_trials": len(responses),
    }


def permute_context_gates(context_json: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Permute the order of gates inside each region (A/B swap protocol)."""
    ctx = copy.deepcopy(context_json)
    rng = random.Random(seed)

    for reg in ctx.get("regions", []):
        gates = reg.get("gates", [])
        if len(gates) > 1:
            # Create a permutation map
            indices = list(range(len(gates)))
            rng.shuffle(indices)
            new_id_map = {f"G{old_i}": f"G{new_i}" for new_i, old_i in enumerate(indices)}

            shuffled_gates = []
            for new_i, old_i in enumerate(indices):
                g = copy.deepcopy(gates[old_i])
                g["id"] = f"G{new_i}"
                shuffled_gates.append(g)
            reg["gates"] = shuffled_gates

            # Update connections
            new_conns = []
            for c in reg.get("internal_connections", []):
                new_conns.append({
                    "from": new_id_map.get(c.get("from", ""), c.get("from", "")),
                    "to": new_id_map.get(c.get("to", ""), c.get("to", "")),
                })
            reg["internal_connections"] = new_conns

    return ctx


def ab_swap_test(
    backend: LLMBackend,
    context_json: Dict[str, Any],
    n_trials: int = 3,
) -> Dict[str, Any]:
    """Run A/B swap consistency test by permuting gate representations across trials.

    Parameters
    ----------
    backend : LLMBackend
        Active LLM backend.
    context_json : dict
        Context dictionary produced by Phase 7.
    n_trials : int
        Number of trials to execute.

    Returns
    -------
    dict
        Consistency report containing metrics and all individual responses.
    """
    responses: List[Dict[str, Any]] = []

    for i in range(n_trials):
        # Permute context
        permuted = permute_context_gates(context_json, seed=i * 42)
        prompt = build_trojan_analysis_prompt(permuted)
        raw_text = backend.query(prompt)
        parsed = parse_llm_response(raw_text)
        responses.append(parsed)

    stats = compute_agreement(responses)
    stats["responses"] = responses
    return stats


def temperature_sweep(
    backend: LLMBackend,
    context_json: Dict[str, Any],
    temperatures: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Execute evaluation across multiple temperature settings if supported."""
    if temperatures is None:
        temperatures = [0.1, 0.5, 1.0]

    results = {}
    for temp in temperatures:
        prompt = build_trojan_analysis_prompt(context_json)
        raw = backend.query(prompt)
        results[str(temp)] = parse_llm_response(raw)

    return results

