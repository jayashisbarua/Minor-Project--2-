#!/usr/bin/env python3
"""
Phase 5 Experiment: Evaluation Splits & Class Balance Audit.
Generates the 8 LOFO folds and reports the exact number of netlists,
clean nodes, Trojan nodes, and positive class balance per fold.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from neuroguard.eval.splits import create_lofo_splits, create_trojan_type_splits

def main():
    folds = create_lofo_splits(seed=42)
    split_stats = {}

    print("Analyzing class balance for 8 LOFO folds...")
    print(f"{'Fold (Test Family)':<18} | {'Train':<6} | {'Val':<5} | {'Test':<5} | {'Test Gates':<10} | {'Test Troj':<9} | {'Pos Rate (%)'}")
    print("-" * 80)

    for fold_name, fold_data in folds.items():
        test_paths = fold_data["test"]
        test_nodes = 0
        test_troj = 0

        for p in test_paths:
            data = torch.load(p, weights_only=False)
            test_nodes += data.num_nodes
            test_troj += int(torch.sum(data.y_node).item())

        pos_rate = (test_troj / max(1, test_nodes)) * 100.0

        split_stats[fold_name] = {
            "train_count": len(fold_data["train"]),
            "val_count": len(fold_data["val"]),
            "test_count": len(fold_data["test"]),
            "test_family": fold_name,
            "val_family": fold_data["val_family"],
            "test_nodes": test_nodes,
            "test_trojan_nodes": test_troj,
            "test_pos_rate_pct": round(pos_rate, 3),
        }

        print(f"{fold_name:<18} | {len(fold_data['train']):<6} | {len(fold_data['val']):<5} | {len(fold_data['test']):<5} | {test_nodes:<10} | {test_troj:<9} | {pos_rate:.3f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/p5_split_balance.json", "w") as fp:
        json.dump(split_stats, fp, indent=2)

if __name__ == "__main__":
    main()

