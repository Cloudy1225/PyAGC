from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree
from pyagc.clusters import BaseClusterHead

EPS = 1e-15

class DMoNClusterHead(BaseClusterHead):
    r"""
    Deep Modularity Network (DMoN) Clustering Head proposed in the
    `"Graph Clustering with Graph Neural Networks"
    <https://arxiv.org/abs/2006.16904>`_ paper (Tsitsulin et al., JMLR 2023).

    This layer learns a soft cluster assignment matrix :math:`\mathbf{S}`
    by projecting node embeddings :math:`\mathbf{Z}` into :math:`K` clusters
    using a linear transformation followed by a softmax. It optimizes the
    clustering structure with two objectives:

    **(1) Spectral modularity loss:**

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \mathrm{Tr}(\mathbf{S}^\top \mathbf{B} \mathbf{S})

    where :math:`\mathbf{B} = \mathbf{A} - \frac{\mathbf{d}\mathbf{d}^\top}{2m}`
    is the modularity matrix, and :math:`m` is the total number of edges.

    **(2) Collapse regularization loss:**

    .. math::
        \mathcal{L}_c = \frac{\sqrt{K}}{N} \left\| \sum_i \mathbf{S}_i^\top \right\|_F - 1

    which prevents unbalanced cluster sizes.

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of input node embeddings.
    """
    def __init__(self, n_clusters: int, n_features: int):
        super(DMoNClusterHead, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

        # Buffer for pre-computed graph structures
        self.register_buffer('adj_sparse', None, persistent=False)
        self.register_buffer('deg', None, persistent=False)
        self.register_buffer('m', None, persistent=False)

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

    def _prepare_graph(self, edge_index: Tensor, N: int):
        """Pre-computes and caches sparse adjacency and degree statistics."""
        if self.adj_sparse is not None and self.adj_sparse.size(0) == N:
            return

        device = edge_index.device
        # 1. Pre-compute degree and total edges
        deg = degree(edge_index[0], N, dtype=torch.float).view(-1, 1)
        m = deg.sum() / 2.0

        # 2. Pre-compute Sparse COO Adjacency Matrix
        # Using coalesce() is vital for optimized sparse-dense matmul
        val = torch.ones(edge_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(edge_index, val, (N, N)).coalesce()

        self.adj_sparse = adj
        self.deg = deg
        self.m = m

    def forward(self, z: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Computes DMoN clustering objectives using node embeddings and graph structure.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: modularity_loss and collapse_loss
        """
        N = z.size(0)
        K = self.n_clusters

        # 0. Ensure graph stats are cached
        self._prepare_graph(edge_index, N)

        # 1. Compute Soft Assignments
        # Materializing (N, K) is often the bottleneck; we use a simple linear projection
        # for better memory management than raw matmul if possible.
        S = torch.matmul(z, self.cluster_centers.t()).softmax(dim=-1)  # (N, K)

        # 2. Spectral Modularity Loss
        # Instead of B = A - (dd^T)/2m, we compute Tr(S^T A S) and Tr(S^T dd^T S) separately.
        # Tr(S^T A S) using sparse multiplication
        # AS calculation: (N, N) @ (N, K) -> (N, K)
        AS = torch.sparse.mm(self.adj_sparse, S)
        # Tr(S^T AS) = sum(S * AS) element-wise product followed by sum
        tr_SAS = torch.sum(S * AS)

        # Tr(S^T (dd^T / 2m) S) = (S^T d)(d^T S) / 2m
        # S^T d is (K, 1), d^T S is (1, K)
        St_d = torch.matmul(S.t(), self.deg)  # (K, 1)
        tr_SddS = torch.sum(St_d ** 2) / (2 * self.m + EPS)

        modularity_loss = -(tr_SAS - tr_SddS) / (2 * self.m + EPS)

        # 3. Collapse Loss (Memory Efficient)
        # L_c = (sqrt(K)/N) * ||sum_i S_i||_F - 1
        cluster_sizes = S.sum(dim=0)  # (K,)
        collapse_loss = (cluster_sizes.norm() * (K ** 0.5) / N) - 1

        return modularity_loss, collapse_loss

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
        sim = torch.matmul(z, self.cluster_centers.t())

        if soft:
            return sim.softmax(dim=-1)
        else:
            return sim.argmax(dim=-1)
