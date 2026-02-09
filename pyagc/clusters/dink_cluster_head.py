from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pyagc.clusters import BaseClusterHead
from pyagc.utils import off_diagonal, pairwise_squared_distance


class DinkClusterHead(BaseClusterHead):
    r"""
    Neural Clustering Layer proposed in the `"Dink-Net: Neural Clustering on Large Graphs"
    <https://arxiv.org/abs/2305.18405>`_ paper (Liu et al., ICML 2023).

    This layer models cluster centers as learnable parameters and optimizes clustering
    assignments via cluster dilation and shrink losses:

    .. math::
        \mathcal{L}_\text{dilation} = \frac{-1}{(K-1)K} \sum_{i=1}^{K} \sum_{j=1, j\neq i}^{K} \| \mathbf{C}_i - \mathbf{C}_j \|_2^2

    .. math::
        \mathcal{L}_\text{shrink} = \frac{1}{BK} \sum_{i=1}^{B} \sum_{j=1}^{K} \| \mathbf{Z}_i - \mathbf{C}_j \|_2^2

    where :math:`\mathbf{C}_i` is the i-th cluster center,
    :math:`\mathbf{Z}_i` is the i-th embedding in a batch,
    :math:`B` is the batch size, and :math:`K` is the number of clusters.
    The cluster dilation loss pushes cluster centers away from each other to expand the clustering space,
    while the cluster shrink loss pulls node embeddings toward all cluster centers to avoid confirmation bias.

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of the input.
    """

    def __init__(self, n_clusters: int, n_features: int):
        super(DinkClusterHead, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

    def reset_cluster_centers(self, cluster_centers: Tensor = None) -> None:
        r"""
        Manually sets the cluster centers.

        Args:
            cluster_centers (torch.Tensor, optional):
                Tensor of shape :obj:`(n_clusters, n_features)` to initialize
                the cluster centers. If None, use Xavier uniform initialization.
        """
        if cluster_centers is not None:
            assert cluster_centers.shape == (self.n_clusters, self.n_features)
            with torch.no_grad():
                self.cluster_centers.copy_(cluster_centers)
        else:
            nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the clustering loss.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: dilation_loss and shrink_loss
        """
        n = z.shape[0]

        # Dilation loss
        # Push different cluster centers apart
        dist_cc = pairwise_squared_distance(self.cluster_centers, self.cluster_centers)
        dilation_loss = -off_diagonal(dist_cc).mean()

        # Shrink loss
        # Pull samples close to cluster centers
        dist_zc = pairwise_squared_distance(z, self.cluster_centers)
        shrink_loss = dist_zc.mean()

        return dilation_loss, shrink_loss

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""
        Predicts cluster assignments.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.
            soft (bool, optional):
                If True, returns the soft assignment matrix;
                if False, returns hard cluster assignments. (default: :obj:`False`)

        Returns:
            - If :obj:`soft` is False, :obj:`(n_samples,)` tensor of cluster indices.
            - If :obj:`soft` is True, :obj:`(n_samples, n_clusters)` tensor of probabilities.
        """
        dist = pairwise_squared_distance(z, self.cluster_centers)  # (n_samples, n_clusters)

        if soft:
            return (-dist.sqrt()).softmax(dim=-1)  # smaller distance => higher score
        else:
            return dist.argmin(dim=-1)  # assign to nearest cluster center
