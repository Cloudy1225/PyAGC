import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from pyagc.clusters import BaseClusterHead


class INCClusterHead(BaseClusterHead):
    r"""
    Neural Clustering Layer proposed in the `"XAI Beyond Classification: Interpretable Neural Clustering"
    <https://jmlr.org/papers/v23/19-497.html>`_ paper (Peng et al., JMLR 2022).

    This layer implements a differentiable k-means reformulation,
    where cluster centers are learnable parameters.

    The cluster centers :math:`\mathbf{\Omega}_j` and weights :math:`\mathbf{W}_j` are related via:

    .. math::
        \mathbf{W}_j = 2 \alpha \mathbf{\Omega}_j

    The loss is defined as:

    .. math::
        L = \frac{1}{\alpha} \sum_{i} (2\alpha - \mathbf{W}_{c(i)}^T \mathbf{z}_i)

    where :math:`c(i)` is the nearest cluster to sample :math:`i`.

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of the input.
        alpha (float): Scaling factor for cluster centers. Default is 0.001.
    """

    def __init__(self, n_clusters: int, n_features: int, alpha: float = 0.001):
        super(INCClusterHead, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha

        # Cluster centers (normalized to unit norm)
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

    def reset_cluster_centers(self, cluster_centers: Tensor = None) -> None:
        r"""
        Manually sets the cluster centers.

        Args:
            cluster_centers (torch.Tensor, optional):
                Tensor of shape (n_clusters, n_features) to initialize the cluster centers.
                If None, use Xavier uniform initialization.
        """
        if cluster_centers is not None:
            assert cluster_centers.shape == (self.n_clusters, self.n_features)
            with torch.no_grad():
                self.cluster_centers.copy_(F.normalize(cluster_centers, dim=-1))
        else:
            nn.init.xavier_uniform_(self.cluster_centers)
            with torch.no_grad():
                self.cluster_centers.copy_(F.normalize(self.cluster_centers, dim=-1))

    def normalize_cluster_centers(self) -> None:
        r"""
        Normalize cluster centers and apply alpha scaling.
        """
        with torch.no_grad():
            normalized = F.normalize(self.cluster_centers, dim=-1)
            self.cluster_centers.copy_(normalized)

    def compute_cluster_center(self) -> Tensor:
        r"""
        Compute the actual cluster centers (scaled by alpha).

        Returns:
            Scaled cluster centers.
        """
        return (0.5 / self.alpha) * self._get_weight()

    def _get_weight(self) -> Tensor:
        r"""
        Compute scaled cluster weight.

        Returns:
            Weight tensor.
        """
        return 2.0 * self.alpha * F.normalize(self.cluster_centers, dim=-1)

    def forward(self, z: Tensor) -> Tensor:
        r"""
        Compute the clustering loss.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.

        Returns:
            Scalar loss tensor.
        """
        z = F.normalize(z, dim=-1)
        W = self._get_weight()

        similarity = F.linear(z, W)
        assignments = similarity.argmax(dim=-1)

        selected_similarity = similarity.gather(1, assignments.unsqueeze(1)).squeeze(1)

        loss = (1. / self.alpha) * (2.0 * self.alpha - selected_similarity).mean()
        return loss

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
        z = F.normalize(z, dim=-1)
        W = self._get_weight()

        similarity = F.linear(z, W)

        if soft:
            return similarity.softmax(dim=-1)
        else:
            return similarity.argmax(dim=-1)

    def gradient_normalize(self) -> None:
        r"""
        Normalize the gradient of the cluster centers.
        """
        if self.cluster_centers.grad is not None:
            with torch.no_grad():
                normed_grad = F.normalize(self.cluster_centers.grad, dim=-1)
                self.cluster_centers.grad.copy_(normed_grad * (0.2 * self.alpha))
