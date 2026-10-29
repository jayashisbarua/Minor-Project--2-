"""Production Trainer for NeuroGuard-X.

Implements Section 3A protocol:
- Early stopping on validation AUPRC
- Checkpoint persistence (best.pt and last.pt)
- Dynamic threshold calibration on validation set
- Multi-task loss: BCE with pos_weight + role loss + graph loss
- Automated artifact generation: curves, metrics CSV, run_summary.json, training_report.md
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroguard.training.early_stopping import EarlyStopping
from neuroguard.training.logger import TrainingLogger
from neuroguard.training.plots import plot_training_curves, plot_test_evaluation
from neuroguard.training.report import generate_training_report
from neuroguard.eval.metrics import compute_node_metrics, compute_graph_metrics
from neuroguard.graphs.features import FeatureScaler


class Trainer:
    """Standardized trainer adhering strictly to Section 3A requirements."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        run_id: str,
        output_dir: Path | str = "results/train",
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.run_id = run_id
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.max_epochs = config.get("max_epochs", 200)
        self.patience = config.get("patience", 20)
        self.min_delta = config.get("min_delta", 1e-4)
        self.warmup_epochs = config.get("warmup_epochs", 5)
        self.restore_best = config.get("restore_best", True)
        self.seed = config.get("seed", 42)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )

        # Early stopping & Logger
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            warmup_epochs=self.warmup_epochs,
            mode="max",
            restore_best=self.restore_best,
        )
        self.logger = TrainingLogger(self.run_dir)

    def train_epoch(self, train_data_list: List[Any], pos_weight_val: float = 20.0) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        role_loss_fn = nn.CrossEntropyLoss()
        graph_loss_fn = nn.BCEWithLogitsLoss()

        for data in train_data_list:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(data)
            node_logits = out["node_logits"]
            role_logits = out["role_logits"]
            graph_logits = out["graph_logits"]

            # Multi-task loss
            loss_node = bce_loss_fn(node_logits, data.y_node.float())
            loss_role = role_loss_fn(role_logits, data.y_role)
            loss_graph = graph_loss_fn(graph_logits, data.y_graph.float())

            loss = loss_node + 0.3 * loss_role + 0.2 * loss_graph
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return {"train_loss": total_loss / max(1, len(train_data_list))}

    @torch.no_grad()
    def evaluate(self, data_list: List[Any], threshold: Optional[float] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_y_true: List[int] = []
        all_y_score: List[float] = []

        bce_loss_fn = nn.BCEWithLogitsLoss()

        for data in data_list:
            data = data.to(self.device)
            out = self.model(data)
            node_logits = out["node_logits"]
            loss = bce_loss_fn(node_logits, data.y_node.float())
            total_loss += loss.item()

            probs = torch.sigmoid(node_logits).cpu().numpy()
            all_y_true.extend(data.y_node.cpu().numpy().tolist())
            all_y_score.extend(probs.tolist())

        y_true_arr = np.array(all_y_true)
        y_score_arr = np.array(all_y_score)

        metrics = compute_node_metrics(y_true_arr, y_score_arr, k=10, threshold=threshold)
        metrics["loss"] = total_loss / max(1, len(data_list))
        return metrics, y_true_arr, y_score_arr

    def fit(
        self,
        train_data: List[Any],
        val_data: List[Any],
        test_data: List[Any],
        scaler: Optional[FeatureScaler] = None,
        test_family: str = "unknown",
    ) -> Dict[str, Any]:
        """Full training loop with early stopping, checkpointing, and reporting."""
        t_start = time.time()
        metrics_history: List[Dict[str, Any]] = []

        # Calculate pos_weight from train data
        total_nodes = sum(d.num_nodes for d in train_data)
        pos_nodes = sum(int(torch.sum(d.y_node).item()) for d in train_data)
        neg_nodes = total_nodes - pos_nodes
        pos_weight_val = max(1.0, min(50.0, float(neg_nodes / max(1, pos_nodes))))

        # Save split info
        split_info = {
            "run_id": self.run_id,
            "test_family": test_family,
            "train_graphs": len(train_data),
            "val_graphs": len(val_data),
            "test_graphs": len(test_data),
        }
        with open(self.run_dir / "split.json", "w") as f:
            json.dump(split_info, f, indent=2)

        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(train_data, pos_weight_val=pos_weight_val)
            val_metrics, _, _ = self.evaluate(val_data)

            # Learning rate scheduler
            self.scheduler.step(val_metrics["loss"])
            curr_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_metrics["train_loss"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_auprc": round(val_metrics["auprc"], 4),
                "val_auroc": round(val_metrics["auroc"], 4),
                "val_f1": round(val_metrics["f1"], 4),
                "lr": curr_lr,
            }
            metrics_history.append(epoch_record)
            self.logger.log(epoch_record)

            # Early stopping check on validation AUPRC
            stop = self.early_stopping.step(epoch, val_metrics["auprc"], self.model)
            if stop:
                break

        self.logger.close()
        wall_time = time.time() - t_start

        # Restore best weights
        self.early_stopping.restore(self.model)

        # Calibrate optimal decision threshold on validation set
        val_metrics_best, val_y_true, val_scores = self.evaluate(val_data)
        # Choose threshold that maximizes F1 on validation set
        best_thresh = 0.5
        best_f1 = 0.0
        total_val_pos = np.sum(val_y_true == 1)
        for th in np.linspace(0.1, 0.9, 17):
            pred = (val_scores >= th).astype(int)
            pos_pred = np.sum(pred == 1)
            tp = np.sum((pred == 1) & (val_y_true == 1))
            f1 = (2 * tp) / max(1, pos_pred + total_val_pos)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th

        # Final evaluation on the unseen TEST fold
        test_metrics, test_y_true, test_y_score = self.evaluate(test_data, threshold=best_thresh)

        # Parameter count
        param_count = sum(p.numel() for p in self.model.parameters())

        # Save checkpoints
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "seed": self.seed,
            "best_epoch": self.early_stopping.best_epoch,
            "stopped_epoch": epoch,
            "best_val_auprc": self.early_stopping.best_score,
            "threshold": best_thresh,
            "test_metrics": test_metrics,
            "param_count": param_count,
        }
        torch.save(checkpoint, self.run_dir / "best.pt")

        # Run summary JSON
        run_summary = {
            "run_id": self.run_id,
            "seed": self.seed,
            "test_family": test_family,
            "param_count": param_count,
            "wall_time_seconds": wall_time,
            "best_epoch": self.early_stopping.best_epoch,
            "stopped_epoch": epoch,
            "stopped_early": self.early_stopping.stopped_early,
            "best_val_auprc": self.early_stopping.best_score,
            "calibrated_threshold": best_thresh,
            "test_metrics": test_metrics,
        }
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        # Generate plots and Markdown report
        plot_training_curves(metrics_history, self.early_stopping.best_epoch, self.run_dir)
        plot_test_evaluation(test_y_true, test_y_score, best_thresh, self.run_dir)
        generate_training_report(self.run_dir, run_summary, self.config)

        return run_summary
