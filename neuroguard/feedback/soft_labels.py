"""Convert LLM verdicts into soft training labels."""

import torch


def verdict_to_soft_labels(
    llm_response: dict,
    region_node_indices: list,
    num_nodes: int,
) -> torch.Tensor:
    """Convert a parsed LLM response into a soft-label tensor.

    Parameters
    ----------
    llm_response : dict
        Must contain ``"verdict"`` (one of ``"trojan_detected"``,
        ``"likely_clean"``, or ``"inconclusive"``) and
        ``"confidence_0_100"`` (int 0–100).
    region_node_indices : list[int]
        Node indices that belong to the analysed region.
    num_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Shape ``[num_nodes]`` with soft labels.  Nodes outside the
        region are ``0.0``.
    """
    verdict = llm_response.get("verdict", "inconclusive")
    confidence = llm_response.get("confidence_0_100", 50)

    if verdict == "trojan_detected":
        region_value = confidence / 100.0
    elif verdict == "likely_clean":
        region_value = (100 - confidence) / 100.0
    else:  # inconclusive
        region_value = 0.5

    soft = torch.zeros(num_nodes, dtype=torch.float32)
    for idx in region_node_indices:
        if 0 <= idx < num_nodes:
            soft[idx] = region_value
    return soft


def merge_soft_labels(
    gnn_scores: torch.Tensor,
    llm_soft: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Merge GNN suspicion scores with LLM-derived soft labels.

    Weighted combination is applied **only** where ``llm_soft > 0``
    (i.e. within analysed regions).  Nodes outside the region keep
    their original GNN score unchanged.

    Parameters
    ----------
    gnn_scores : torch.Tensor
        Shape ``[num_nodes]`` – raw GNN suspicion scores.
    llm_soft : torch.Tensor
        Shape ``[num_nodes]`` – soft labels from :func:`verdict_to_soft_labels`.
    alpha : float
        Weight given to the LLM soft labels (default ``0.3``).

    Returns
    -------
    torch.Tensor
        Shape ``[num_nodes]`` – merged soft targets.
    """
    merged = gnn_scores.clone()
    mask = llm_soft > 0
    merged[mask] = alpha * llm_soft[mask] + (1 - alpha) * gnn_scores[mask]
    return merged
