"""Unit tests for Phase 5: Evaluation splits and metrics."""

import pytest
from pathlib import Path
from neuroguard.eval.splits import create_lofo_splits, create_trojan_type_splits, get_base_design
from neuroguard.eval.metrics import compute_node_metrics, compute_calibration_ece
from neuroguard.eval.stats import mcnemar_test, bootstrap_ci


def test_lofo_splits_no_leakage():
    """Verify that no base design or netlist appears in both train/val and test."""
    folds = create_lofo_splits(seed=42)
    assert len(folds) == 8, f"Expected 8 LOFO folds, got {len(folds)}"

    for fold_name, fold_data in folds.items():
        train_paths = fold_data["train"]
        val_paths = fold_data["val"]
        test_paths = fold_data["test"]

        train_set = set(train_paths)
        val_set = set(val_paths)
        test_set = set(test_paths)

        # 1. No overlapping filepaths
        assert len(train_set & test_set) == 0, f"Fold {fold_name}: train and test overlap!"
        assert len(val_set & test_set) == 0, f"Fold {fold_name}: val and test overlap!"
        assert len(train_set & val_set) == 0, f"Fold {fold_name}: train and val overlap!"

        # 2. No base design leakage
        test_families = {get_base_design(p) for p in test_paths}
        train_families = {get_base_design(p) for p in train_paths}
        val_families = {get_base_design(p) for p in val_paths}

        assert test_families == {fold_name}, f"Fold {fold_name} test contains wrong family: {test_families}"
        assert fold_name not in train_families, f"Fold {fold_name} leaked into train!"
        assert fold_name not in val_families, f"Fold {fold_name} leaked into val!"
        assert len(train_families & val_families) == 0, f"Fold {fold_name} train/val share family!"


def test_metrics_fixed_top_k():
    """Verify metrics calculation uses fixed k and handles edge cases."""
    y_true = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0] # 2 positives
    y_score = [0.1, 0.2, 0.05, 0.9, 0.3, 0.4, 0.85, 0.15, 0.25, 0.0]
    metrics = compute_node_metrics(y_true, y_score, k=5)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "top_5_precision" in metrics
    assert metrics["top_5_precision"] == 2 / 5
    assert metrics["top_5_recall"] == 2 / 2
    assert 0.0 <= metrics["ece"] <= 1.0


def test_mcnemar_statistical_test():
    """Verify McNemar test calculation on known contingency table."""
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    pred_a = [1, 1, 1, 0, 0, 0, 1, 0]
    pred_b = [1, 1, 0, 0, 0, 0, 0, 0]
    res = mcnemar_test(y_true, pred_a, pred_b)
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0

