from typing import Union, Tuple, Dict

import torch
from torch import Tensor
from torch_geometric.utils import degree


@torch.no_grad()
def modularity(edge_index: Tensor, clusters: Tensor, vectorized: bool = True) -> float:
    r"""Computes the modularity of clusters in a graph.
    This implementation is adapted from: `google-research/dmon
    <https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py#L93>`_.

    Args:
        edge_index (torch.Tensor): :obj:`(2, num_edges)`, undirected edge list.
        clusters (torch.Tensor): :obj:`(num_nodes,)`, cluster assignments.
        vectorized (bool, optional): Whether to use vectorized computation.
            (default: :obj:`True`)

    Returns:
        Modularity score.
    """
    row, col = edge_index
    num_edges = row.shape[0]
    num_nodes = clusters.shape[0]
    device = clusters.device

    deg = degree(row, num_nodes=num_nodes).to(device)

    if vectorized:
        # Vectorized computation
        same_cluster = (clusters[row] == clusters[col])
        edges_within = same_cluster.sum().float()

        # Compute degree sum per cluster
        num_clusters = clusters.max().item() + 1
        deg_per_cluster = torch.zeros(num_clusters, dtype=torch.float32, device=device)
        deg_per_cluster.scatter_add_(0, clusters, deg)

        expected_edges = (deg_per_cluster ** 2).sum() / num_edges
        mod = edges_within - expected_edges
    else:
        # Original loop-based computation
        mod = 0.0
        for c in torch.unique(clusters):
            mask = (clusters == c)

            submask = mask[row] & mask[col]
            edges_in_c = submask.sum()

            deg_c = deg[mask].sum()

            mod += edges_in_c - (deg_c ** 2) / num_edges

    return float(mod / num_edges)


@torch.no_grad()
def conductance(edge_index: Tensor, clusters: Tensor, vectorized: bool = True) -> float:
    r"""Computes the average conductance of clusters in a graph.
    This implementation is adapted from: `google-research/dmon
    <https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py#L115>`_.

    Args:
        edge_index (torch.Tensor): :obj:`(2, num_edges)`, undirected edge list.
        clusters (torch.Tensor): :obj:`(num_nodes,)`, cluster assignments.
        vectorized (bool, optional): Whether to use vectorized computation.
            (default: :obj:`True`)

    Returns:
        Average conductance of all clusters.
    """
    row, col = edge_index

    if vectorized:
        # Vectorized computation
        same_cluster = (clusters[row] == clusters[col])
        intra = same_cluster.sum().float()
        inter = (~same_cluster).sum().float()
    else:
        # Original loop-based computation
        inter = 0  # Number of inter-cluster edges.
        intra = 0  # Number of intra-cluster edges.

        for cluster_id in torch.unique(clusters):
            mask = (clusters == cluster_id)
            out_cluster = (mask[row] ^ mask[col])  # only one endpoint in cluster
            in_cluster = mask[row] & mask[col]  # both endpoints in cluster

            inter += out_cluster.sum() * 0.5
            intra += in_cluster.sum()

    return float(inter / (inter + intra))


VALID_METRICS = ('Mod', 'Cond')


def structure_metrics(
        edge_index: Tensor,
        clusters: Tensor,
        metrics: Union[str, Tuple[str, ...]] = ('Mod', 'Cond'),
        vectorized: bool = True
) -> Dict[str, float]:
    r"""Computes structural clustering metrics on a graph.

    Supports modularity and conductance measures for evaluating the structural
    quality of clustering on graphs.

    Args:
        edge_index (torch.Tensor): Undirected edge index tensor of shape :obj:`(2, num_edges)`.
        clusters (torch.Tensor): Cluster assignments of shape :obj:`(num_nodes,)`.
        metrics (str or tuple of str, optional): Structural metrics to compute.
            Valid options are: :obj:`'Mod'`, :obj:`'Cond'`.
            (default: :obj:`('Mod', 'Cond')`)
        vectorized (bool, optional): Whether to use vectorized computation for better performance.
            (default: :obj:`True`)

    Returns:
        Dictionary mapping metric names to their computed values.

    Example:
        >>> # Fast vectorized version (default)
        >>> result = structure_metrics(edge_index, clusters, metrics=('Mod', 'Cond'))
        >>> print(result)
        {'Mod': 0.45, 'Cond': 0.32}

        >>> # Loop-based version
        >>> result = structure_metrics(edge_index, clusters, vectorized=False)
    """
    if isinstance(metrics, str):
        metrics = (metrics,)

    invalid = [m for m in metrics if m not in VALID_METRICS]
    if invalid:
        raise ValueError(f"Invalid metric(s): {', '.join(invalid)}. "
                         f"Valid metrics are: {', '.join(VALID_METRICS)}.")

    results = {}

    if 'Mod' in metrics:
        results['Mod'] = modularity(edge_index, clusters, vectorized=vectorized)
    if 'Cond' in metrics:
        results['Cond'] = conductance(edge_index, clusters, vectorized=vectorized)

    return results
