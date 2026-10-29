#!/usr/bin/env python3
"""
Phase 4 Experiment: Feature Ablation Study.
Evaluates detection performance on LOFO splits comparing:
1. Structural-only features (22 dimensions)
2. Structural + Temporal features (26 dimensions)
Outputs AUROC and AUPRC for both feature sets to results/p4_feature_ablation.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import glob
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.graphs.features import compute_node_features, FeatureScaler
from neuroguard.graphs.simulate import BitParallelSimulator
from neuroguard.eval.splits import create_lofo_splits

def main():
    parser = NetlistParser()
    sim = BitParallelSimulator(num_cycles=200, seed=42)

    # Use fold 'c2670' (or representative LOFO fold)
    folds = create_lofo_splits(seed=42)
    fold_c2670 = folds["c2670"]

    # Select representative subset of train and test
    train_paths = fold_c2670["train"][:15]
    test_paths = fold_c2670["test"][:15]

    print(f"Loading and extracting features for {len(train_paths)} train and {len(test_paths)} test netlists...")

    def extract_dataset(paths):
        x_struct_list = []
        x_full_list = []
        y_list = []

        for p in paths:
            # Load netlist
            v_name = Path(p).stem
            # Find raw .v file in data/
            candidates = list(Path("data").glob(f"**/{v_name}.v"))
            if not candidates:
                continue
            v_file = candidates[0]
            log_file = v_file.parent / "log.txt"

            G = parser.parse(v_file, log_file if log_file.exists() else None)
            temp_stats = sim.simulate(G)

            # 1. Structural only
            x_s = compute_node_features(G, include_temporal=False)
            # 2. Structural + Temporal
            x_f = compute_node_features(G, temporal_stats=temp_stats, include_temporal=True)

            y = torch.tensor([G.nodes[n].get("is_trojan", 0) for n in G.nodes()], dtype=torch.long)

            x_struct_list.append(x_s)
            x_full_list.append(x_f)
            y_list.append(y)

        X_s = torch.cat(x_struct_list, dim=0).numpy()
        X_f = torch.cat(x_full_list, dim=0).numpy()
        Y = torch.cat(y_list, dim=0).numpy()
        return X_s, X_f, Y

    X_train_s, X_train_f, Y_train = extract_dataset(train_paths)
    X_test_s, X_test_f, Y_test = extract_dataset(test_paths)

    print(f"Train samples: {len(Y_train)} (Positives: {np.sum(Y_train == 1)})")
    print(f"Test samples: {len(Y_test)} (Positives: {np.sum(Y_test == 1)})")

    # 1. Evaluate Structural Only
    rf_struct = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf_struct.fit(X_train_s, Y_train)
    prob_s = rf_struct.predict_proba(X_test_s)[:, 1]

    auc_s = roc_auc_score(Y_test, prob_s)
    auprc_s = average_precision_score(Y_test, prob_s)

    # 2. Evaluate Structural + Temporal
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf_full.fit(X_train_f, Y_train)
    prob_f = rf_full.predict_proba(X_test_f)[:, 1]

    auc_f = roc_auc_score(Y_test, prob_f)
    auprc_f = average_precision_score(Y_test, prob_f)

    results = {
        "fold": "c2670",
        "train_samples": int(len(Y_train)),
        "test_samples": int(len(Y_test)),
        "structural_only": {
            "num_features": X_train_s.shape[1],
            "auroc": round(float(auc_s), 4),
            "auprc": round(float(auprc_s), 4),
        },
        "structural_plus_temporal": {
            "num_features": X_train_f.shape[1],
            "auroc": round(float(auc_f), 4),
            "auprc": round(float(auprc_f), 4),
        },
        "auprc_improvement": round(float(auprc_f - auprc_s), 4),
        "auroc_improvement": round(float(auc_f - auc_s), 4),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/p4_feature_ablation.json", "w") as fp:
        json.dump(results, fp, indent=2)

    print("\nFeature Ablation Results:")
    print(f"Structural Only ({results['structural_only']['num_features']} dims):  AUROC = {results['structural_only']['auroc']:.4f}, AUPRC = {results['structural_only']['auprc']:.4f}")
    print(f"Structural + Temporal ({results['structural_plus_temporal']['num_features']} dims): AUROC = {results['structural_plus_temporal']['auroc']:.4f}, AUPRC = {results['structural_plus_temporal']['auprc']:.4f}")
    print(f"Delta AUPRC: {results['auprc_improvement']:+.4f}, Delta AUROC: {results['auroc_improvement']:+.4f}")

if __name__ == "__main__":
    main()

