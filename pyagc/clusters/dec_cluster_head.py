import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from pyagc.clusters import BaseClusterHead
from pyagc.utils import pairwise_squared_distance


class DECClusterHead(BaseClusterHead):
    r"""
    Neural Clustering Layer proposed in the `"Unsupervised Deep Embedding for Clustering Analysis"
    <https://arxiv.org/abs/1511.06335>`_ paper (Xie et al., ICML 2016).

    This layer learns cluster centers and computes soft assignment of
    input samples to clusters using Student's t-distribution.

    Specifically, the probability :math:`q_{ij}` that a sample :math:`i`
    belongs to cluster :math:`j` is given by:

    .. math::
        q_{ij} = \frac{(1 + \|z_i - \mu_j\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}
                      {\sum_{j'} (1 + \|z_i - \mu_{j'}\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}

    where :math:`z_i` is the embedded point and :math:`\mu_j` is the j-th
    cluster center, and :math:`\alpha` is the degrees of freedom of the
    Student's t-distribution (default is 1).

    The target distribution :math:`p_{ij}` is computed as:

    .. math::
        p_{ij} = \frac{q_{ij}^2 / \sum_i q_{ij}}{\sum_j (q_{ij}^2 / \sum_i q_{ij})}

    The loss is the KL divergence between the soft assignments :math:`q`
    and the target distribution :math:`p`:

    .. math::
        L = \text{KL}(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of the input.
        alpha (float, optional): Degrees of freedom for Student's t-distribution.
            Default is 1.
    """

    def __init__(self, n_clusters: int, n_features: int, alpha: float = 1.0):
        super(DECClusterHead, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

        # Cache for target distribution P to enable delayed updates
        self._cached_target = None

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

    def forward(self, z: Tensor, update_target: bool = True) -> Tensor:
        r"""
        Computes the KL divergence loss between the soft assignments and
        the target distribution.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.
            update_target (bool, optional): Whether to recompute the target
                distribution P. If :obj:`False`, uses the cached distribution.
                This is useful for maintaining training stability by updating
                the target less frequently. (default: :obj:`True`)

        Returns:
            Scalar loss tensor.
        """
        # Compute soft assignment Q
        q = self._soft_assign(z)  # (n_samples, n_clusters)

        # Conditionally update target distribution P
        # Recompute if: (1) update_target is True, or (2) cache is empty, or (3) batch size changed
        if update_target or self._cached_target is None or self._cached_target.size(0) != z.size(0):
            p = self._target_distribution(q)  # (n_samples, n_clusters)
            self._cached_target = p.detach()  # Cache and detach from computation graph
        else:
            p = self._cached_target

        # KL Divergence loss: KL(P || Q)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

    def _soft_assign(self, z: Tensor) -> Tensor:
        r"""
        Computes soft assignment using Student's t-distribution.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.

        Returns:
            Soft assignment matrix of shape :obj:`(n_samples, n_clusters)`.
        """
        dist = pairwise_squared_distance(z, self.cluster_centers)  # Squared Euclidean distance
        numerator = (1.0 + dist / self.alpha).pow(-(self.alpha + 1) / 2)
        q = numerator / numerator.sum(dim=-1, keepdim=True)
        return q

    def _target_distribution(self, q: Tensor) -> Tensor:
        r"""
        Computes the target distribution p.

        Args:
            q (torch.Tensor): Soft assignment matrix of shape :obj:`(n_samples, n_clusters)`.

        Returns:
            Target distribution matrix of shape :obj:`(n_samples, n_clusters)`.
        """
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=-1, keepdim=True)
        return p.detach()  # Detach to stop gradients through p

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
        if soft:
            return self._soft_assign(z)
        else:
            # Hard assignment: directly use distance for argmin
            dist = pairwise_squared_distance(z, self.cluster_centers)
            return dist.argmin(dim=-1)
