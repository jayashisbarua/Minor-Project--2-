"""Phase 7: Context Expansion & Anonymization Demo."""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
from neuroguard.context.to_json import build_llm_context


def main():
    graph_path = root_dir / "cache" / "c2670" / "c2670_T000.pt"
    if not graph_path.exists():
        print(f"File not found: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)
    print(f"Loaded: {data.netlist_name} ({data.num_nodes} nodes, {data.edge_index.size(1)} edges)")

    # Simulate realistic suspicious scores using y_node + random noise
    rng = np.random.default_rng(42)
    base_scores = data.y_node.cpu().numpy().astype(float)
    noise = rng.normal(0, 0.05, size=len(base_scores))
    scores = np.clip(base_scores * 0.9 + noise + 0.05, 0.0, 1.0)

    # Context expansion with k=10, 2-hop radius
    context = build_llm_context(data, scores, k=10, hops=2)

    out_file = root_dir / "results" / "p7_context_demo.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(context, f, indent=2)

    print(f"\nContext expansion summary:")
    print(f"  Flagged regions: {context['total_regions_flagged']}")
    print(f"  Seed gates analyzed: {context['total_seed_gates_analyzed']}")
    for reg in context["regions"]:
        print(f"  - Region {reg['region_id']}: {reg['num_gates']} gates, "
              f"{reg['num_internal_connections']} connections, "
              f"max suspicion: {reg['max_suspicion']:.4f}")

    print(f"\nSaved demonstration output to: {out_file}")


if __name__ == "__main__":
    main()

