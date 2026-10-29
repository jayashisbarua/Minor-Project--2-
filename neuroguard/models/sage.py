"""GraphSAGE Encoder on the structural view (inductive representation)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder with mean aggregation."""

    def __init__(self, in_channels: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h

