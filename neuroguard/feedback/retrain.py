"""Feedback-augmented retraining loop."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from neuroguard.feedback.soft_labels import verdict_to_soft_labels, merge_soft_labels
from neuroguard.feedback.contrastive import (
    TextProjector,
    info_nce_loss,
    compute_region_embeddings,
)


def _build_soft_label_tensor(
    llm_response: dict,
    num_nodes: int,
) -> torch.Tensor:
    """Build a soft-label tensor from a single LLM response.

    The response is expected to include ``region_node_indices`` alongside
    the standard verdict/confidence fields.
    """
    region_indices = llm_response.get("region_node_indices", [])
    return verdict_to_soft_labels(llm_response, region_indices, num_nodes)


def retrain_with_feedback(
    model: nn.Module,
    train_data: list,
    val_data: list,
    llm_responses: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> tuple:
    """Retrain a GNN model with feedback-augmented losses.

    The total loss combines the original supervised loss with two
    auxiliary terms:

    ``total = original_loss + beta * distillation_loss + gamma * contrastive_loss``

    Parameters
    ----------
    model : nn.Module
        The GNN model to retrain (modified in-place).
    train_data : list
        List of PyG ``Data`` objects for training.
    val_data : list
        List of PyG ``Data`` objects for validation.
    llm_responses : list[dict]
        Each dict must contain ``"verdict"``, ``"confidence_0_100"``,
        ``"region_node_indices"`` and optionally ``"evidence_items"``
        (list) for the contrastive head.
    config : dict
        Training configuration.  Recognised keys:

        - ``beta`` (float): weight for distillation loss (default 0.1)
        - ``gamma`` (float): weight for contrastive loss (default 0.05)
        - ``feedback_epochs`` (int): retraining epochs (default 30)
        - ``learning_rate`` (float): LR for retraining (default 1e-4)
        - ``hidden_dim`` (int): model hidden dim (default 64)
        - ``use_contrastive`` (bool): enable contrastive loss (default True)
    device : torch.device
        Device to run on.

    Returns
    -------
    tuple[nn.Module, dict]
        Retrained model and a metrics dict with per-epoch losses.
    """
    beta = config.get("beta", 0.1)
    gamma = config.get("gamma", 0.05)
    feedback_epochs = config.get("feedback_epochs", 30)
    lr = config.get("learning_rate", 1e-4)
    hidden_dim = config.get("hidden_dim", 64)
    use_contrastive = config.get("use_contrastive", True)

    model = model.to(device)
    model.train()

    # Build per-graph soft labels from LLM responses
    # Map llm_responses to training graphs (one per graph, wrap-around if fewer)
    soft_labels_per_graph: List[Optional[torch.Tensor]] = []
    for i, data in enumerate(train_data):
        if i < len(llm_responses):
            sl = _build_soft_label_tensor(llm_responses[i], data.num_nodes)
            soft_labels_per_graph.append(sl.to(device))
        else:
            soft_labels_per_graph.append(None)

    # Optional contrastive projector
    text_projector = None
    if use_contrastive and llm_responses:
        text_projector = TextProjector(hidden_dim=hidden_dim).to(device)

    # Optimiser covers model + optional projector parameters
    params = list(model.parameters())
    if text_projector is not None:
        params += list(text_projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    metrics_history: List[Dict[str, float]] = []

    for epoch in range(1, feedback_epochs + 1):
        epoch_loss = 0.0
        epoch_distill = 0.0
        epoch_contrastive = 0.0

        for i, data in enumerate(train_data):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            node_logits = out["node_logits"]

            # --- Original supervised loss ---
            original_loss = bce_loss_fn(node_logits, data.y_node.float())

            # --- Distillation loss (KL-div with soft labels) ---
            distillation_loss = torch.tensor(0.0, device=device)
            if soft_labels_per_graph[i] is not None:
                soft = soft_labels_per_graph[i]
                # Merge soft labels with current predictions
                pred_probs = torch.sigmoid(node_logits).detach()
                merged = merge_soft_labels(pred_probs, soft, alpha=0.3)

                # KL-divergence between model output distribution and merged targets
                log_pred = F.log_softmax(
                    torch.stack([node_logits, -node_logits], dim=-1), dim=-1
                )
                target_dist = torch.stack([merged, 1 - merged], dim=-1)
                distillation_loss = F.kl_div(
                    log_pred, target_dist, reduction="batchmean"
                )

            # --- Contrastive loss ---
            contrastive_loss = torch.tensor(0.0, device=device)
            if (
                use_contrastive
                and text_projector is not None
                and i < len(llm_responses)
            ):
                resp = llm_responses[i]
                region_indices = resp.get("region_node_indices", [])
                if region_indices:
                    # Build verdict feature vector
                    confidence = resp.get("confidence_0_100", 50) / 100.0
                    verdict = resp.get("verdict", "inconclusive")
                    evidence_items = resp.get("evidence_items", [])

                    feat = torch.tensor(
                        [
                            confidence,
                            float(verdict == "trojan_detected"),
                            float(verdict == "likely_clean"),
                            float(verdict == "inconclusive"),
                            len(evidence_items) / 10.0,
                        ],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)  # [1, 5]

                    text_emb = text_projector(feat)  # [1, hidden_dim]

                    # Region embedding from current node embeddings
                    node_emb = out["node_embeddings"]
                    idx_t = torch.tensor(
                        region_indices, dtype=torch.long, device=device
                    )
                    region_emb = node_emb[idx_t].mean(dim=0, keepdim=True)  # [1, hidden_dim]

                    # Need at least 2 regions for InfoNCE; use
                    # non-region as implicit negative
                    all_indices = set(range(data.num_nodes))
                    non_region = list(all_indices - set(region_indices))
                    if non_region:
                        neg_idx = torch.tensor(
                            non_region[:len(region_indices)],
                            dtype=torch.long,
                            device=device,
                        )
                        neg_emb = node_emb[neg_idx].mean(dim=0, keepdim=True)

                        region_embs = torch.cat([region_emb, neg_emb], dim=0)
                        # Negative text embedding: zero vector
                        neg_text = torch.zeros_like(text_emb)
                        text_embs = torch.cat([text_emb, neg_text], dim=0)

                        contrastive_loss = info_nce_loss(region_embs, text_embs)

            # --- Total loss ---
            total = original_loss + beta * distillation_loss + gamma * contrastive_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total.item()
            epoch_distill += distillation_loss.item()
            epoch_contrastive += contrastive_loss.item()

        n = max(1, len(train_data))
        metrics_history.append(
            {
                "epoch": epoch,
                "total_loss": round(epoch_loss / n, 6),
                "distillation_loss": round(epoch_distill / n, 6),
                "contrastive_loss": round(epoch_contrastive / n, 6),
            }
        )

    metrics = {
        "feedback_epochs": feedback_epochs,
        "beta": beta,
        "gamma": gamma,
        "use_contrastive": use_contrastive,
        "history": metrics_history,
    }
    return model, metrics
