"""InfoNCE contrastive loss for GNN-LLM alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    region_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss between region and text embeddings.

    Positive pairs are same-region GNN + text embeddings; negatives are
    all cross-region pairs within the batch.

    Parameters
    ----------
    region_embeddings : torch.Tensor
        Shape ``[N_regions, hidden_dim]`` – GNN mean-pooled region embeddings.
    text_embeddings : torch.Tensor
        Shape ``[N_regions, hidden_dim]`` – projected LLM verdict embeddings.
    temperature : float
        Softmax temperature (default ``0.07``).

    Returns
    -------
    torch.Tensor
        Scalar InfoNCE loss.
    """
    # L2-normalise both sets of embeddings
    region_norm = F.normalize(region_embeddings, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)

    # Cosine similarity matrix [N, N]
    logits = torch.matmul(region_norm, text_norm.T) / temperature

    # Positive pairs lie on the diagonal
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric InfoNCE: average of region→text and text→region losses
    loss_rt = F.cross_entropy(logits, labels)
    loss_tr = F.cross_entropy(logits.T, labels)

    return (loss_rt + loss_tr) / 2.0


class TextProjector(nn.Module):
    """MLP that maps a fixed-size verdict feature vector to ``hidden_dim``.

    Verdict feature vector (dim=5):
        ``[confidence/100, is_trojan (0/1), is_clean (0/1),
          is_inconclusive (0/1), num_evidence_items/10]``
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, 5]`` – batch of verdict feature vectors.

        Returns
        -------
        torch.Tensor
            Shape ``[B, hidden_dim]``.
        """
        return self.net(x)


def compute_region_embeddings(
    model: nn.Module,
    data,
    region_indices_list: list,
) -> torch.Tensor:
    """Extract mean-pooled region embeddings from a GNN model.

    Parameters
    ----------
    model : nn.Module
        A :class:`NeuroGuardModel` (or compatible) that returns a dict
        with key ``"node_embeddings"``.
    data
        A PyG ``Data`` object.
    region_indices_list : list[list[int]]
        Each inner list contains node indices for one region.

    Returns
    -------
    torch.Tensor
        Shape ``[N_regions, hidden_dim]`` – mean-pooled embeddings.
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
    node_embeddings = out["node_embeddings"]  # [num_nodes, hidden_dim]

    region_embeds = []
    for indices in region_indices_list:
        if not indices:
            # Empty region: zero vector
            region_embeds.append(
                torch.zeros(node_embeddings.size(1), device=node_embeddings.device)
            )
        else:
            idx = torch.tensor(indices, dtype=torch.long, device=node_embeddings.device)
            region_embeds.append(node_embeddings[idx].mean(dim=0))

    return torch.stack(region_embeds)
