"""Multi-view fusion layers for NeuroGuard-X.

Supports:
1. WeightedVotingFusion: Softmax-weighted voting over individual model output logits.
2. TransformerFusion: Multi-head self-attention fusion over node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedVotingFusion(nn.Module):
    """Learnable softmax-weighted ensemble voting over logits from 3 models."""

    def __init__(self, num_models: int = 3):
        super().__init__()
        # Initialized with equal weights
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, logits_list: list[torch.Tensor]) -> torch.Tensor:
        """Combine list of [N, 1] logits into single [N, 1] fused logit."""
        alpha = F.softmax(self.weights, dim=0)
        # Stack along dim -1: [N, 1, 3]
        stacked = torch.stack(logits_list, dim=-1)
        # Weighted sum: [N, 1]
        fused = torch.sum(stacked * alpha, dim=-1)
        return fused


class TransformerFusion(nn.Module):
    """Multi-head cross-view self-attention fusion over node embeddings."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_views: int = 3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse list of [N, D] view embeddings into [N, D] unified embedding.

        embeddings_list has length num_views (e.g. [GCN, SAGE, GAT]).
        """
        # Stack views as sequence: [N, num_views, D]
        seq = torch.stack(embeddings_list, dim=1)
        # Self-attention across views for each node
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + attn_out)
        out = self.mlp(seq)
        # Mean pooling across views: [N, D]
        fused = torch.mean(out, dim=1)
        return fused

