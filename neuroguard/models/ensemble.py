"""Unified Multi-View GNN Ensemble Architecture for NeuroGuard-X.

Combines:
- GCN (structural view: global topology)
- GraphSAGE (structural view: inductive learning)
- GAT (functional view: neighbor importance)
With:
- Learnable weighted voting OR Transformer-based embedding fusion
- Multi-task heads: per-node suspicion, 4-class gate roles, and graph-level likelihood
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroguard.models.gcn import GCNEncoder
from neuroguard.models.sage import SAGEEncoder
from neuroguard.models.gat import GATEncoder
from neuroguard.models.fusion import WeightedVotingFusion, TransformerFusion


class NeuroGuardModel(nn.Module):
    """Multi-view GNN model supporting single-model baselines and full ensemble."""

    def __init__(
        self,
        in_channels: int = 26,
        hidden_dim: int = 64,
        model_type: str = "ensemble_voting",
        dropout: float = 0.2,
    ):
        """
        model_type:
            - 'gcn': Single GCN on structural view
            - 'sage': Single GraphSAGE on structural view (friend's baseline)
            - 'gat': Single GAT on functional view
            - 'ensemble_voting': Full 3-model ensemble with softmax-weighted logit voting
            - 'ensemble_transformer': Full 3-model ensemble with Transformer embedding fusion
        """
        super().__init__()
        self.model_type = model_type
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Initialize encoders
        if model_type in ("gcn", "ensemble_voting", "ensemble_transformer"):
            self.gcn = GCNEncoder(in_channels, hidden_dim, hidden_dim, dropout=dropout)
        if model_type in ("sage", "ensemble_voting", "ensemble_transformer"):
            self.sage = SAGEEncoder(in_channels, hidden_dim, hidden_dim, dropout=dropout)
        if model_type in ("gat", "ensemble_voting", "ensemble_transformer"):
            self.gat = GATEncoder(in_channels, hidden_dim, hidden_dim, dropout=dropout)

        # Fusion layer
        if model_type == "ensemble_voting":
            self.fusion = WeightedVotingFusion(num_models=3)
            # Separate node heads per encoder
            self.head_gcn = nn.Linear(hidden_dim, 1)
            self.head_sage = nn.Linear(hidden_dim, 1)
            self.head_gat = nn.Linear(hidden_dim, 1)
        elif model_type == "ensemble_transformer":
            self.fusion = TransformerFusion(embed_dim=hidden_dim, num_heads=4, num_views=3)

        # Unified prediction heads
        self.node_head = nn.Linear(hidden_dim, 1)
        self.role_head = nn.Linear(hidden_dim, 4)  # 4 classes: clean, trigger, body, payload
        self.graph_head = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> dict[str, torch.Tensor]:
        x = data.x
        edge_struct = data.edge_index_struct
        edge_func = data.edge_index_func

        if self.model_type == "gcn":
            h = self.gcn(x, edge_struct)
            node_logits = self.node_head(h).squeeze(-1)
        elif self.model_type == "sage":
            h = self.sage(x, edge_struct)
            node_logits = self.node_head(h).squeeze(-1)
        elif self.model_type == "gat":
            h = self.gat(x, edge_func)
            node_logits = self.node_head(h).squeeze(-1)
        elif self.model_type == "ensemble_voting":
            h_gcn = self.gcn(x, edge_struct)
            h_sage = self.sage(x, edge_struct)
            h_gat = self.gat(x, edge_func)

            logit_gcn = self.head_gcn(h_gcn)
            logit_sage = self.head_sage(h_sage)
            logit_gat = self.head_gat(h_gat)

            fused_logits = self.fusion([logit_gcn, logit_sage, logit_gat])
            node_logits = fused_logits.squeeze(-1)
            h = (h_gcn + h_sage + h_gat) / 3.0
        elif self.model_type == "ensemble_transformer":
            h_gcn = self.gcn(x, edge_struct)
            h_sage = self.sage(x, edge_struct)
            h_gat = self.gat(x, edge_func)

            h = self.fusion([h_gcn, h_sage, h_gat])
            node_logits = self.node_head(h).squeeze(-1)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Multi-task heads
        role_logits = self.role_head(h)
        # Graph-level readout: max pooling over node suspicion embeddings
        graph_embed = torch.max(h, dim=0, keepdim=True)[0]
        graph_logits = self.graph_head(graph_embed).squeeze(-1)

        return {
            "node_logits": node_logits,
            "role_logits": role_logits,
            "graph_logits": graph_logits,
            "node_embeddings": h,
        }

