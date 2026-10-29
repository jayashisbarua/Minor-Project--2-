"""GAT Encoder on the functional view (learns neighbor importance via attention)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """2-layer GAT encoder with multi-head attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h

