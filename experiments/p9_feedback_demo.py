"""Phase 9: Feedback Loop & Retraining Demo."""

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
from neuroguard.models.ensemble import NeuroGuardModel
from neuroguard.feedback.soft_labels import verdict_to_soft_labels, merge_soft_labels
from neuroguard.feedback.contrastive import TextProjector, info_nce_loss
from neuroguard.feedback.retrain import retrain_with_feedback


def main():
    graph_path = root_dir / "cache" / "c2670" / "c2670_T000.pt"
    if not graph_path.exists():
        print(f"File not found: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)
    print(f"Loaded: {data.netlist_name} ({data.num_nodes} nodes)")

    # 1. Simulate LLM verdict on suspicious region
    # Assume gates 10..25 form a detected Trojan region
    region_nodes = list(range(10, 26))
    llm_response = {
        "verdict": "trojan_detected",
        "confidence_0_100": 85,
        "evidence": ["High fan-in trigger pattern", "Low toggle rate on gate G12"],
        "region_node_indices": region_nodes,
        "evidence_items": ["E1", "E2"],
    }

    # 2. Convert to soft labels
    soft_labels = verdict_to_soft_labels(llm_response, region_nodes, data.num_nodes)
    print(f"\nGenerated soft labels tensor: shape {soft_labels.shape}")
    print(f"  Region nodes value: {soft_labels[10].item():.2f}")
    print(f"  Non-region nodes value: {soft_labels[0].item():.2f}")

    # 3. Simulate GNN predictions & merge
    mock_gnn_probs = torch.full((data.num_nodes,), 0.15)
    mock_gnn_probs[region_nodes] = 0.65
    merged_targets = merge_soft_labels(mock_gnn_probs, soft_labels, alpha=0.3)
    print(f"\nMerged soft targets: region mean = {merged_targets[region_nodes].mean().item():.4f}")

    # 4. Demonstrate contrastive projection
    projector = TextProjector(hidden_dim=64)
    feat = torch.tensor([[0.85, 1.0, 0.0, 0.0, 0.2]])  # [1, 5]
    text_emb = projector(feat)
    print(f"\nProjected LLM text embedding: shape {text_emb.shape}")

    # 5. Quick retrain demonstration (3 epochs on single graph)
    model = NeuroGuardModel(in_channels=26, hidden_dim=64, model_type="gcn")
    device = torch.device("cpu")
    config = {
        "beta": 0.1,
        "gamma": 0.05,
        "feedback_epochs": 3,
        "learning_rate": 1e-4,
        "hidden_dim": 64,
        "use_contrastive": True,
    }

    print("\nRunning feedback-augmented retraining (3 epochs demo)...")
    retrained_model, metrics = retrain_with_feedback(
        model=model,
        train_data=[data],
        val_data=[data],
        llm_responses=[llm_response],
        config=config,
        device=device,
    )

    for record in metrics["history"]:
        print(f"  Epoch {record['epoch']}: total_loss={record['total_loss']:.4f}, "
              f"distillation_loss={record['distillation_loss']:.4f}, "
              f"contrastive_loss={record['contrastive_loss']:.4f}")

    out_file = root_dir / "results" / "p9_feedback_demo.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved feedback demo results to: {out_file}")


if __name__ == "__main__":
    main()

