"""Phase 8: LLM Integration & Consistency Demo."""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from neuroguard.llm.backends import get_backend
from neuroguard.llm.prompts import build_trojan_analysis_prompt, parse_llm_response
from neuroguard.llm.consistency import ab_swap_test


def main():
    # Load demo context generated in Phase 7
    context_file = root_dir / "results" / "p7_context_demo.json"
    if not context_file.exists():
        print("Please run Phase 7 demo first: experiments/p7_context_demo.py")
        return

    with open(context_file) as f:
        context_json = json.load(f)

    # Use StubBackend for deterministic, offline testing
    backend = get_backend("stub")
    print("Using LLM Backend: StubBackend (simulating security analyst)")

    # Build prompt
    prompt = build_trojan_analysis_prompt(context_json)
    print(f"\nGenerated structured prompt ({len(prompt)} chars).")

    # Run A/B Swap consistency test (3 trials)
    print("\nRunning A/B Swap Consistency Protocol (3 trials with gate permutations)...")
    report = ab_swap_test(backend, context_json, n_trials=3)

    print(f"  Majority verdict: {report['majority_verdict']}")
    print(f"  Agreement ratio: {report['agreement_ratio'] * 100:.1f}%")
    print(f"  Mean confidence: {report['confidence_mean']:.1f}")

    out_file = root_dir / "results" / "p8_llm_demo.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved LLM integration results to: {out_file}")


if __name__ == "__main__":
    main()

