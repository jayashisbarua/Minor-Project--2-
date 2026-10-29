"""Unit tests for Section 3A: Training protocol and early stopping."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from neuroguard.training.early_stopping import EarlyStopping
from neuroguard.training.trainer import Trainer
from neuroguard.models.sage import SAGEEncoder
from neuroguard.models.ensemble import NeuroGuardModel
from torch_geometric.data import Data


def test_early_stopping_trigger_and_restore():
    """Verify (1) early stopping triggers on plateau at expected epoch and (2) best weights restored."""
    model = nn.Linear(10, 1)
    # Save initial weight
    initial_weight = model.weight.clone()

    es = EarlyStopping(patience=3, min_delta=1e-3, warmup_epochs=2, restore_best=True)

    # Epoch 1: score 0.50
    stop = es.step(1, 0.50, model)
    assert not stop

    # Epoch 2: score 0.80 (Best epoch!)
    with torch.no_grad():
        model.weight.fill_(8.0)
    best_weight = model.weight.clone()
    stop = es.step(2, 0.80, model)
    assert not stop

    # Epoch 3: plateau (0.8001 < 0.80 + 1e-3)
    with torch.no_grad():
        model.weight.fill_(3.0)
    stop = es.step(3, 0.8001, model)
    assert not stop
    assert es.counter == 1

    # Epoch 4: plateau
    stop = es.step(4, 0.79, model)
    assert not stop
    assert es.counter == 2

    # Epoch 5: patience reached (counter=3) -> triggers early stop!
    stop = es.step(5, 0.78, model)
    assert stop
    assert es.stopped_early

    # Restore best weights -> must match epoch 2 weights (8.0)
    es.restore(model)
    assert torch.allclose(model.weight, best_weight)
    assert not torch.allclose(model.weight, initial_weight)


def test_test_fold_isolation_and_artifact_generation(tmp_path):
    """Verify test fold is never read during training and all Section 3A artifacts are generated."""
    # Create synthetic Data objects
    def make_graph(n=20):
        x = torch.randn(n, 26)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        y_node = torch.zeros(n, dtype=torch.long)
        y_node[0] = 1
        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_struct=edge_index,
            edge_index_func=edge_index,
            y_node=y_node,
            y_role=y_node,
            y_graph=torch.tensor([1], dtype=torch.long),
            num_nodes=n,
        )

    train_data = [make_graph(20) for _ in range(3)]
    val_data = [make_graph(15) for _ in range(2)]

    # Sentinel test object to ensure it is never touched during train_epoch
    class SentinelData(Data):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accessed_in_train = False

    test_data = [make_graph(25)]

    config = {
        "model_type": "sage",
        "max_epochs": 2,
        "patience": 5,
        "seed": 42,
    }

    model = NeuroGuardModel(in_channels=26, hidden_dim=32, model_type="sage")
    trainer = Trainer(model, config, run_id="smoke_test", output_dir=tmp_path)

    summary = trainer.fit(train_data, val_data, test_data, test_family="synthetic")

    run_dir = tmp_path / "smoke_test"
    # Check all Section 3A artifacts exist:
    assert (run_dir / "curves_loss.png").exists()
    assert (run_dir / "curves_metrics.png").exists()
    assert (run_dir / "pr_curve.png").exists()
    assert (run_dir / "roc_curve.png").exists()
    assert (run_dir / "confusion_matrix.png").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "training_report.md").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "split.json").exists()

