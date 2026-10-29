"""Early stopping implementation monitoring validation AUPRC."""

from typing import Optional, Dict, Any
import copy
import torch


class EarlyStopping:
    """Monitors a validation metric (default AUPRC, mode 'max') and triggers early stopping."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        warmup_epochs: int = 5,
        mode: str = "max",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.restore_best = restore_best

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self.counter: int = 0
        self.stopped_early: bool = False

    def step(self, epoch: int, score: float, model: torch.nn.Module) -> bool:
        """Evaluate metric for epoch.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self._update_best(epoch, score, model)
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self._update_best(epoch, score, model)
            self.counter = 0
        else:
            if epoch >= self.warmup_epochs:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stopped_early = True
                    return True

        return False

    def _update_best(self, epoch: int, score: float, model: torch.nn.Module):
        self.best_score = score
        self.best_epoch = epoch
        if self.restore_best:
            self.best_state = copy.deepcopy(model.state_dict())

    def restore(self, model: torch.nn.Module):
        """Restore the best weights to model."""
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

