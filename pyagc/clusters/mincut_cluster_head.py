from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree

from pyagc.clusters import BaseClusterHead

EPS = 1e-15


class MinCutClusterHead(BaseClusterHead):
    r"""
    MinCut Clustering Head proposed in
    `"Spectral Clustering in Graph Neural Networks for Graph Pooling"
    <https://arxiv.org/abs/1907.00481>`_ (Bianchi et al., ICML 2019).

    This layer learns a **soft cluster assignment matrix** :math:`\mathbf{S}`
    by projecting node embeddings :math:`\mathbf{Z}` into :math:`K` clusters.
    It jointly optimizes two objectives:

    **(1) MinCut loss:**

    .. math::
        \mathcal{L}_{\text{mincut}} = - \frac{\mathrm{Tr}(\mathbf{S}^\top \mathbf{A} \mathbf{S})}
        {\mathrm{Tr}(\mathbf{S}^\top \mathbf{D} \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix.

    **(2) Orthogonality loss:**

    .. math::
        \mathcal{L}_{\text{ortho}} =
        \left\| \frac{\mathbf{S}^\top \mathbf{S}}{\|\mathbf{S}^\top \mathbf{S}\|_F}
        - \frac{\mathbf{I}_K}{\sqrt{K}} \right\|_F

    which encourages near-orthogonal cluster assignments.

    Args:
        n_clusters (int): Number of clusters :math:`K`.
        n_features (int): Feature dimension of node embeddings :math:`F`.
        temperature (float, optional): Softmax temperature. (default: 1.0)
    """

    def __init__(self, n_clusters: int, n_features: int, temperature: float = 1.0):
        super(MinCutClusterHead, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.temperature = temperature

        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

        # Buffer for pre-computed graph structures
        self.register_buffer('adj_sparse', None, persistent=False)
        self.register_buffer('deg', None, persistent=False)

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

        # 1. Pre-compute degree
        deg = degree(edge_index[0], N, dtype=torch.float)

        # 2. Pre-compute Sparse COO Adjacency Matrix
        # Using coalesce() is vital for optimized sparse-dense matmul
        val = torch.ones(edge_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(edge_index, val, (N, N)).coalesce()

        self.adj_sparse = adj
        self.deg = deg

    def forward(self, z: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Compute MinCut and Orthogonality losses given node embeddings and graph structure.

        Args:
            z (torch.Tensor): Node embeddings :math:`(N, F)`.
            edge_index (torch.Tensor): Edge indices :math:`(2, E)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mincut_loss and ortho_loss
        """
        N = z.size(0)
        K = self.n_clusters

        # === Step 0: Cache graph ===
        self._prepare_graph(edge_index, N)

        # === Step 1: Soft assignment ===
        S = torch.matmul(z, self.cluster_centers.t())  # (N, K)
        S = torch.softmax(S / self.temperature, dim=-1)

        # === Step 2: Compute MinCut Loss ===
        # Tr(S^T A S) using sparse multiplication
        # AS calculation: (N, N) @ (N, K) -> (N, K)
        AS = torch.sparse.mm(self.adj_sparse, S)
        # Tr(S^T AS) = sum(S * AS) element-wise product followed by sum
        tr_SAS = torch.sum(S * AS)

        # Tr(S^T D S) where D is diagonal degree matrix
        # = sum_i d_i * (S_i^T S_i) = sum_i d_i * ||S_i||^2
        # More efficiently: deg^T (S ⊙ S) where ⊙ is element-wise product
        # = sum over all elements of: deg[:, None] * S * S
        S_squared = S * S  # (N, K)
        tr_SDS = torch.sum(self.deg.view(-1, 1) * S_squared)

        mincut_loss = -tr_SAS / (tr_SDS + EPS)

        # === Step 3: Orthogonality Loss ===
        # S^T S is (K, K)
        StS = torch.matmul(S.t(), S)

        # # Normalize by Frobenius norm
        # StS_norm = StS / (torch.norm(StS, p='fro') + EPS)
        # # Target: I_K / sqrt(K)
        # I_normalized = torch.eye(K, device=z.device, dtype=z.dtype) / (K ** 0.5)
        # ortho_loss = torch.norm(StS_norm - I_normalized, p='fro')

        StS_norm = torch.norm(StS, p='fro')
        trace_StS = torch.trace(StS)
        ortho_loss = 2.0 * (1.0 - trace_StS / (StS_norm * (K ** 0.5) + EPS))

        return mincut_loss, ortho_loss

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""
        Predict cluster assignments.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, F).
            soft (bool, optional): If True, return soft assignment probabilities.

        Returns:
            torch.Tensor: Hard cluster indices or soft assignment matrix.
        """
        sim = torch.matmul(z, self.cluster_centers.t())

        if soft:
            return (sim / self.temperature).softmax(dim=-1)
        else:
            return sim.argmax(dim=-1)
